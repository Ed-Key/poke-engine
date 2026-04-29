//! HTTP client for the Plan E metamon NN sidecar.
//!
//! Speaks to the Python FastAPI service that wraps Kakuna 142M (see
//! `~/Projects/metamon-spike/sidecar/nn_sidecar.py`). The contract is:
//!
//! - `POST /policy` body: `{"state": <BattleRequest JSON>, "perspective": "p1"|"p2"}`
//! - Response (`PolicyResponse`):
//!     - `probs[13]`     normalized over legal actions (sums to ~1.0)
//!     - `q_values[13]`  raw shaped-reward Q estimates per action
//!     - `v_estimate`    scalar; **NOT** comparable to `evaluate.rs`'s scale
//!                       — see verifier CRIT-2 in
//!                       `analysis/plan-e-recon/phase-4-5-spec-verification.md`
//!     - `decoded_actions[13]` informational labels like "move:earthquake"
//!
//! Index layout for `probs`/`q_values`:
//!     0..3   moves, alphabetical by `clean_no_numbers(name)`
//!     4..8   switches, alphabetical by `clean_name(species)`
//!     9..12  tera-variants of the same moves as 0..3
//!
//! All NN client calls happen INSIDE the MCTS worker thread (already a
//! `std::thread::spawn` in `analyze_stream_handler`, or `spawn_blocking` in
//! `analyze_handler`). Do NOT call from an async axum handler context — the
//! `reqwest::blocking::Client` will stall the executor.
//!
//! ## Failure mode
//!
//! Any error here MUST be non-fatal at the call site. The engine's
//! `MctsSearch::new` will log the error and fall back to uniform priors.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Sidecar's expected probability/q-value vector length (4 moves + 5 switches +
/// 4 tera variants).
pub const ACTION_DIM: usize = 13;

/// Errors that the sidecar HTTP path can produce. All variants are non-fatal:
/// callers should log + fall back to uniform priors.
#[derive(Debug, Error)]
pub enum NnClientError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("response decoding error: {0}")]
    Decode(#[from] serde_json::Error),

    #[error("bad response shape: {0}")]
    BadResponseShape(&'static str),

    #[error("sidecar returned status {0}: {1}")]
    BadStatus(u16, String),
}

/// One side's view of the sidecar — `"p1"` or `"p2"`.
///
/// The Rust engine internally tracks `Side1` / `Side2`; this enum is the
/// over-the-wire representation that Kakuna's policy head uses.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Perspective {
    P1,
    P2,
}

impl Perspective {
    pub fn as_str(self) -> &'static str {
        match self {
            Perspective::P1 => "p1",
            Perspective::P2 => "p2",
        }
    }
}

/// Sidecar `/policy` JSON response.
///
/// `probs` and `q_values` are `Vec<f32>` (not fixed-size arrays) on the wire —
/// `validate()` enforces `len() == ACTION_DIM` so callers can index safely.
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyResponse {
    pub probs: Vec<f32>,
    pub q_values: Vec<f32>,
    pub v_estimate: f32,
    pub decoded_actions: Vec<String>,
}

impl PolicyResponse {
    /// Run sanity checks the sidecar's own validators don't enforce on the wire.
    /// Returns `Err` for any case where the sidecar response cannot be safely
    /// indexed downstream.
    pub fn validate(&self) -> Result<(), NnClientError> {
        if self.probs.len() != ACTION_DIM {
            return Err(NnClientError::BadResponseShape("probs.len() != 13"));
        }
        if self.q_values.len() != ACTION_DIM {
            return Err(NnClientError::BadResponseShape("q_values.len() != 13"));
        }
        if self.decoded_actions.len() != ACTION_DIM {
            return Err(NnClientError::BadResponseShape("decoded_actions.len() != 13"));
        }
        // Sum check: sidecar already renormalizes, so a healthy response sums
        // to ~1.0. Non-positive sum signals "all actions illegal" (degenerate
        // forced-switch with no live mons) — also a fall-back-to-uniform case.
        let sum: f32 = self.probs.iter().sum();
        if sum < 0.5 {
            return Err(NnClientError::BadResponseShape("probs sum < 0.5"));
        }
        if !sum.is_finite() {
            return Err(NnClientError::BadResponseShape("probs sum is NaN/inf"));
        }
        Ok(())
    }

    /// Index of the highest-probability action — informational helper for
    /// logging / tests.
    pub fn argmax(&self) -> Option<usize> {
        self.probs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
}

#[derive(Debug, Serialize)]
struct PolicyRequestBody<'a> {
    state: &'a serde_json::Value,
    perspective: &'a str,
}

/// Synchronous HTTP client for the metamon sidecar.
///
/// Internally uses `reqwest::blocking::Client`, which is fine in an MCTS
/// worker thread (the inner search is sync) but MUST NOT be called from a
/// tokio async handler — that would stall the executor.
pub struct NnClient {
    base_url: String,
    timeout: Duration,
    client: reqwest::blocking::Client,
}

impl NnClient {
    /// Build a new client. `base_url` is something like `http://localhost:7273`
    /// (no trailing slash — paths are appended verbatim).
    pub fn new(base_url: String, timeout: Duration) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest::blocking::Client::build");
        NnClient {
            base_url,
            timeout,
            client,
        }
    }

    /// Hit the sidecar's `/healthz` endpoint. Returns `Ok(())` if the sidecar
    /// is up; an error otherwise. Useful as a startup sanity check.
    pub fn healthz(&self) -> Result<(), NnClientError> {
        let url = format!("{}/healthz", self.base_url);
        let resp = self.client.get(&url).send()?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let body = resp.text().unwrap_or_default();
            return Err(NnClientError::BadStatus(code, body));
        }
        Ok(())
    }

    /// Get a policy + value for `state_json` from `perspective`. The state
    /// argument is a `serde_json::Value` so the encoder can build it once.
    ///
    /// Returns `Err` for ANY failure — caller is expected to log + fall back.
    pub fn policy(
        &self,
        state_json: &serde_json::Value,
        perspective: Perspective,
    ) -> Result<PolicyResponse, NnClientError> {
        let url = format!("{}/policy", self.base_url);
        let body = PolicyRequestBody {
            state: state_json,
            perspective: perspective.as_str(),
        };
        let resp = self.client.post(&url).json(&body).send()?;
        let status = resp.status();
        if !status.is_success() {
            let code = status.as_u16();
            let text = resp.text().unwrap_or_default();
            return Err(NnClientError::BadStatus(code, text));
        }
        let policy: PolicyResponse = resp.json()?;
        policy.validate()?;
        Ok(policy)
    }

    /// Effective request timeout (constant-once-built). Mostly here for tests.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Base URL, for diagnostics.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

// ---------------------------------------------------------------------------
// In-crate unit tests (exhaustive coverage in tests/test_nn_client.rs).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn good_response() -> PolicyResponse {
        PolicyResponse {
            probs: vec![
                0.10, 0.05, 0.05, 0.05, // 4 moves
                0.10, 0.10, 0.10, 0.10, 0.05, // 5 switches
                0.10, 0.10, 0.05, 0.05, // 4 tera (sums to ~1.00)
            ],
            q_values: vec![0.0; ACTION_DIM],
            v_estimate: 0.0,
            decoded_actions: (0..ACTION_DIM).map(|i| format!("move:{}", i)).collect(),
        }
    }

    #[test]
    fn validate_accepts_good_response() {
        let resp = good_response();
        resp.validate().expect("should validate");
    }

    #[test]
    fn validate_rejects_short_probs() {
        let mut resp = good_response();
        resp.probs.pop();
        let err = resp.validate().unwrap_err();
        assert!(matches!(err, NnClientError::BadResponseShape(_)));
    }

    #[test]
    fn validate_rejects_zero_sum() {
        let mut resp = good_response();
        for p in resp.probs.iter_mut() {
            *p = 0.0;
        }
        let err = resp.validate().unwrap_err();
        assert!(matches!(err, NnClientError::BadResponseShape(_)));
    }

    #[test]
    fn perspective_serialization() {
        assert_eq!(Perspective::P1.as_str(), "p1");
        assert_eq!(Perspective::P2.as_str(), "p2");
    }

    #[test]
    fn argmax_returns_top_action() {
        let mut resp = good_response();
        resp.probs[7] = 0.99; // overwrite to skew
        let _ = resp.validate().is_err(); // sum > 1 invalidates -- but argmax test uses raw vec
        assert_eq!(resp.argmax(), Some(7));
    }
}
