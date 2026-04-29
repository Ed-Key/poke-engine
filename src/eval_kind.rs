//! Plan E eval-dispatch enum.
//!
//! Selects between the heuristic-only baseline (current) and the NN-augmented
//! mode where Kakuna's policy is consulted ONCE at the root and used as the
//! PUCT prior.
//!
//! Per verifier CRIT-2 (`analysis/plan-e-recon/phase-4-5-spec-verification.md`),
//! `root_eval` is ALWAYS the heuristic `evaluate(state)` — even in NN mode.
//! Kakuna's `v_estimate` is in raw shaped-reward units (~[100, 2000]) and is
//! NOT comparable to the heuristic's signed-f32 (~[-300, +300]) scale; mixing
//! them saturates the leaf sigmoid and destroys the Q-signal. So the NN's
//! contribution is **policy only** at this stage.

use std::sync::Arc;

use crate::nn_client::NnClient;

/// Which evaluator the search should consult.
///
/// `Heuristic` is the pre-Plan-E baseline. `Nn(...)` enables a single root
/// `/policy` call to the sidecar; the returned distribution becomes the PUCT
/// prior on s1's options. `s2` priors stay uniform (Kakuna is a one-sided
/// model — see Phase 0.5 finding).
#[derive(Clone)]
pub enum EvalKind {
    Heuristic,
    Nn(Arc<NnClient>),
}

impl std::fmt::Debug for EvalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalKind::Heuristic => write!(f, "EvalKind::Heuristic"),
            EvalKind::Nn(_) => write!(f, "EvalKind::Nn(<client>)"),
        }
    }
}

impl EvalKind {
    /// True iff we should consult the NN at the root for the policy prior.
    pub fn uses_nn(&self) -> bool {
        matches!(self, EvalKind::Nn(_))
    }
}
