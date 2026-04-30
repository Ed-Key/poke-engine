//! Heuristic prior policy for Plan I (NN prior dampening).
//!
//! Computes a sparse 13-slot distribution that puts mass on (damage-calc
//! top move, matchup-switch). Caller blends with the NN policy to rescue
//! alternatives that Kakuna's overconfident policy would otherwise starve.
//! See docs/superpowers/specs/2026-04-30-plan-i-prior-dampening-design.md.

use crate::choices::{Choice, Choices, MoveCategory};
use crate::engine::damage_calc::{calculate_damage, DamageRolls};
use crate::engine::state::MoveChoice;
use crate::nn_client::ACTION_DIM;
use crate::nn_state_encoder::SidePerspective;
use crate::pokemon::PokemonName;
use crate::state::{SideReference, State};

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

/// Pick the active's highest-expected-damage legal damaging move against
/// the opposing active. Tiebreak (within 10% of top damage): highest
/// `base_power × accuracy / 100`.
///
/// Returns None when:
///   - All legal moves are status / switch (no damaging move).
///   - All damaging moves do 0 damage (immunity).
///   - The active has no moves at all.
pub fn damage_calc_top_move(state: &State, perspective: SidePerspective) -> Option<Choices> {
    let attacking_side = match perspective {
        SidePerspective::Side1 => SideReference::SideOne,
        SidePerspective::Side2 => SideReference::SideTwo,
    };
    let active = match perspective {
        SidePerspective::Side1 => state.side_one.get_active_immutable(),
        SidePerspective::Side2 => state.side_two.get_active_immutable(),
    };

    let mut candidates: Vec<(Choices, i16, f32)> = Vec::new();
    for mv in active.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 {
            continue;
        }
        let choice: &Choice = &mv.choice;
        if choice.category == MoveCategory::Status || choice.category == MoveCategory::Switch {
            continue;
        }
        if let Some((_, max_dmg)) =
            calculate_damage(state, &attacking_side, choice, DamageRolls::Max)
        {
            if max_dmg > 0 {
                let acc = choice.accuracy;
                let score = choice.base_power * (acc / 100.0);
                candidates.push((mv.id, max_dmg, score));
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    let top_damage = candidates[0].1 as f32;
    let mut tiebreak_pool: Vec<&(Choices, i16, f32)> = candidates
        .iter()
        .filter(|c| (c.1 as f32) >= top_damage * 0.9)
        .collect();
    tiebreak_pool.sort_by(|a, b| {
        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(tiebreak_pool[0].0)
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
