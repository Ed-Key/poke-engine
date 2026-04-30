//! Heuristic prior policy for Plan I (NN prior dampening).
//!
//! Computes a sparse 13-slot distribution that puts mass on (damage-calc
//! top move, matchup-switch). Caller blends with the NN policy to rescue
//! alternatives that Kakuna's overconfident policy would otherwise starve.
//! See docs/superpowers/specs/2026-04-30-plan-i-prior-dampening-design.md.

use crate::choices::Choices;
use crate::nn_client::ACTION_DIM;
use crate::nn_state_encoder::SidePerspective;
use crate::pokemon::PokemonName;
use crate::engine::state::MoveChoice;
use crate::state::State;

#[derive(Debug, Clone)]
pub struct HeuristicPrior {
    /// 13-element distribution over Plan E's action slots, sums to 1.0
    /// (or 0.0 when both heuristic picks were skipped — caller falls back
    /// to raw NN priors in that case).
    pub probs: [f32; ACTION_DIM],
    /// Move ID the damage-calc heuristic selected (None if skipped).
    pub damage_calc_pick: Option<Choices>,
    /// Species the matchup heuristic selected (None if skipped).
    pub matchup_switch_pick: Option<PokemonName>,
}

/// Compute the heuristic prior. Returns None when neither heuristic
/// produces a valid pick (e.g., last Pokemon alive AND locked into status
/// move). Caller falls back to raw NN priors.
pub fn compute(
    _state: &State,
    _perspective: SidePerspective,
    _options: &[MoveChoice],
    _mass_dmg: f32,
    _mass_switch: f32,
) -> Option<HeuristicPrior> {
    // Stub — filled in across Tasks 2-4.
    None
}
