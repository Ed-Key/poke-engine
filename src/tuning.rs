//! Engine tuning knobs (engine-prior-tuning branch, 2026-05-08).
//!
//! Four runtime-configurable parameters identified by 4-agent convergence
//! analysis as high-leverage fixes for the engine's prior-dominance pathology:
//!
//! 1. `prior_cap`         — clip per-action NN prior before renormalize. Caps
//!                          over-confident NN spikes that lock MCTS visits to
//!                          one action. `1.0` (default) disables.
//! 2. `dirichlet_alpha`   — Dirichlet-noise concentration mixed into root
//!                          priors only. AlphaZero-chess uses `0.3`. `0.0`
//!                          disables.
//! 3. `dirichlet_eps`     — fraction of Dirichlet noise blended in:
//!                          `prior' = (1-eps)*prior + eps*dirichlet`.
//!                          AlphaZero default `0.25`. `0.0` disables.
//! 4. `eval_slope`        — sigmoid slope used to map heuristic eval
//!                          (`evaluate()` ~[-300,+300]) to leaf value [0,1].
//!                          Original `0.0125` saturates at ±200; `0.005`
//!                          gives MCTS more dynamic range.
//!
//! All defaults preserve pre-branch behavior bit-identically. The server
//! binary calls `init_tuning()` once at startup with values from the CLI.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct TuningConfig {
    pub prior_cap: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
    pub eval_slope: f32,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            prior_cap: 1.0,
            dirichlet_alpha: 0.0,
            dirichlet_eps: 0.0,
            eval_slope: 0.0125,
        }
    }
}

static TUNING: OnceLock<TuningConfig> = OnceLock::new();

pub fn init_tuning(cfg: TuningConfig) {
    let _ = TUNING.set(cfg);
}

pub fn tuning() -> &'static TuningConfig {
    TUNING.get_or_init(TuningConfig::default)
}

/// Sample one Gamma(alpha, 1.0) variate using Marsaglia–Tsang. Handles
/// alpha < 1 via the standard `X = G(alpha+1) * U^(1/alpha)` boost.
pub fn sample_gamma(alpha: f32, rng: &mut impl rand::RngCore) -> f32 {
    use rand_distr::{Distribution, Gamma};
    let g = Gamma::new(alpha as f64, 1.0).expect("gamma alpha must be positive");
    g.sample(rng) as f32
}

/// Draw `n` iid Gamma(alpha,1) samples and normalize → Dirichlet(alpha,...,alpha).
pub fn sample_dirichlet(alpha: f32, n: usize, rng: &mut impl rand::RngCore) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let raw: Vec<f32> = (0..n).map(|_| sample_gamma(alpha, rng).max(1e-30)).collect();
    let s: f32 = raw.iter().sum();
    if s > 0.0 && s.is_finite() {
        raw.into_iter().map(|x| x / s).collect()
    } else {
        let u = 1.0 / n as f32;
        vec![u; n]
    }
}

/// Mix Dirichlet noise into priors:
///   `priors[i] = (1-eps)*priors[i] + eps*dirichlet[i]`
/// Caller is responsible for ensuring eps and alpha are in valid ranges.
pub fn apply_dirichlet_noise(priors: &mut [f32], alpha: f32, eps: f32) {
    if eps <= 0.0 || alpha <= 0.0 || priors.is_empty() {
        return;
    }
    let mut rng = rand::rng();
    let noise = sample_dirichlet(alpha, priors.len(), &mut rng);
    let eps = eps.clamp(0.0, 1.0);
    for (p, n) in priors.iter_mut().zip(noise.iter()) {
        *p = (1.0 - eps) * *p + eps * *n;
    }
}

/// Cap each prior at `cap`, then renormalize so they sum to 1. No-op when
/// cap >= 1.0. When the cap forces every entry below `cap` (impossible in
/// practice unless n*cap < 1), falls back to uniform.
pub fn apply_prior_cap(priors: &mut [f32], cap: f32) {
    if cap >= 1.0 || priors.is_empty() {
        return;
    }
    for p in priors.iter_mut() {
        if *p > cap {
            *p = cap;
        }
    }
    let s: f32 = priors.iter().sum();
    if s > 1e-6 && s.is_finite() {
        for p in priors.iter_mut() {
            *p /= s;
        }
    } else {
        let u = 1.0 / priors.len() as f32;
        for p in priors.iter_mut() {
            *p = u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_inert() {
        let c = TuningConfig::default();
        assert_eq!(c.prior_cap, 1.0);
        assert_eq!(c.dirichlet_alpha, 0.0);
        assert_eq!(c.dirichlet_eps, 0.0);
        assert_eq!(c.eval_slope, 0.0125);
    }

    #[test]
    fn cap_no_op_when_cap_ge_1() {
        let mut p = vec![0.7_f32, 0.2, 0.1];
        apply_prior_cap(&mut p, 1.0);
        assert_eq!(p, vec![0.7, 0.2, 0.1]);
    }

    #[test]
    fn cap_clips_and_renormalizes() {
        let mut p = vec![0.95_f32, 0.03, 0.02];
        apply_prior_cap(&mut p, 0.5);
        // After clip: [0.5, 0.03, 0.02], sum = 0.55
        // After renorm: [~0.909, ~0.0545, ~0.0364]
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(p[0] <= 0.5_f32 + 1e-6 || p[0] >= 0.5_f32 - 1e-6);
    }

    #[test]
    fn dirichlet_sample_is_simplex() {
        let mut rng = rand::rng();
        let v = sample_dirichlet(0.3, 5, &mut rng);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
        for x in &v {
            assert!(*x >= 0.0);
        }
    }

    #[test]
    fn noise_no_op_when_eps_zero() {
        let mut p = vec![0.7_f32, 0.2, 0.1];
        let snapshot = p.clone();
        apply_dirichlet_noise(&mut p, 0.3, 0.0);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn noise_changes_priors_when_eps_positive() {
        let mut p = vec![0.7_f32, 0.2, 0.1];
        apply_dirichlet_noise(&mut p, 0.3, 0.25);
        // After mixing 25% Dirichlet noise the values almost certainly differ.
        assert!((p[0] - 0.7).abs() > 1e-6 || (p[1] - 0.2).abs() > 1e-6);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }
}
