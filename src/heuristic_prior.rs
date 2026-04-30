//! Heuristic prior policy for Plan I (NN prior dampening).
//!
//! Computes a sparse 13-slot distribution that puts mass on (damage-calc
//! top move, matchup-switch). Caller blends with the NN policy to rescue
//! alternatives that Kakuna's overconfident policy would otherwise starve.
//! See docs/superpowers/specs/2026-04-30-plan-i-prior-dampening-design.md.

use crate::choices::{Choice, Choices, MoveCategory};
use crate::engine::damage_calc::{calculate_damage, type_effectiveness_modifier, DamageRolls};
use crate::engine::state::MoveChoice;
use crate::nn_client::ACTION_DIM;
use crate::nn_state_encoder::SidePerspective;
use crate::pokemon::PokemonName;
use crate::state::{Pokemon, SideReference, State};

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
    let (attacking_side, active) = match perspective {
        SidePerspective::Side1 => (SideReference::SideOne, state.side_one.get_active_immutable()),
        SidePerspective::Side2 => (SideReference::SideTwo, state.side_two.get_active_immutable()),
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
        if let Some((max_dmg, _crit)) =
            calculate_damage(state, &attacking_side, choice, DamageRolls::Max)
        {
            if max_dmg > 0 {
                let acc = if choice.accuracy < 0.0 { 100.0 } else { choice.accuracy };
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

/// Foul-play `_estimate_matchup` analogue. For each non-fainted, non-active
/// bench Pokemon, compute:
///
///   score = Σ over opp.legal_damaging_moves [type_eff(opp_type, my_bench)]
///         - Σ over my_bench.legal_damaging_moves [type_eff(my_type, opp)]
///
/// Lower (more negative) is better — bench resists more than it gets
/// resisted by. Returns None when:
///   - The active is force-trapped (switching not legal).
///   - No bench Pokemon are alive (last alive).
fn matchup_score_against(opp: &Pokemon, candidate: &Pokemon) -> f32 {
    let mut incoming: f32 = 0.0;
    for mv in opp.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 {
            continue;
        }
        if mv.choice.category == MoveCategory::Status
            || mv.choice.category == MoveCategory::Switch
        {
            continue;
        }
        incoming += type_effectiveness_modifier(&mv.choice.move_type, candidate);
    }

    let mut outgoing: f32 = 0.0;
    for mv in candidate.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 {
            continue;
        }
        if mv.choice.category == MoveCategory::Status
            || mv.choice.category == MoveCategory::Switch
        {
            continue;
        }
        outgoing += type_effectiveness_modifier(&mv.choice.move_type, opp);
    }

    incoming - outgoing
}

pub fn matchup_switch_pick(
    state: &State,
    perspective: SidePerspective,
) -> Option<PokemonName> {
    let (my_side, opp_side) = match perspective {
        SidePerspective::Side1 => (&state.side_one, &state.side_two),
        SidePerspective::Side2 => (&state.side_two, &state.side_one),
    };
    if my_side.force_trapped {
        return None;
    }
    let opp = opp_side.get_active_immutable();
    let active_idx_u8 = my_side.active_index as u8;

    let mut best: Option<(PokemonName, f32)> = None;
    for (idx, pkmn) in my_side.pokemon.into_iter().enumerate() {
        if (idx as u8) == active_idx_u8 {
            continue;
        }
        if pkmn.hp <= 0 {
            continue;
        }
        let score = matchup_score_against(opp, pkmn);
        match best {
            None => best = Some((pkmn.id, score)),
            Some((_, best_score)) if score < best_score => {
                best = Some((pkmn.id, score))
            }
            _ => {}
        }
    }

    best.map(|(name, _)| name)
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
