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
use crate::nn_state_encoder::{
    active_move_ids, alpha_perm_with_norm, move_index_to_slot, move_name_norm, pokemon_name_norm,
    reserve_slot_for, reserve_species, SidePerspective,
};
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
///   - The static `force_trapped` flag is set on this side. Volatile or
///     ability-based traps (PARTIALLYTRAPPED, SHADOWTAG, ARENATRAP,
///     MAGNETPULL) are NOT detected here — `compute()` filters those
///     downstream by intersecting the pick against the legal `options`
///     list it receives.
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
///
/// ## Slot mapping
///
/// The 13-element `probs` vector uses Plan E's alphabetical-slot layout
/// (matches `nn_state_encoder::map_policy_to_options`):
///   - 0..3   active moves, alphabetical by `move_name_norm`
///   - 4..8   reserve switches, alphabetical by `pokemon_name_norm`
///   - 9..12  tera variants of slots 0..3
///
/// We resolve picks → slot indices by reusing the same `alpha_perm_with_norm`
/// helper that drives `map_policy_to_options`, so the mapping stays in
/// lockstep with the NN-prior path. We do NOT piggyback on
/// `map_policy_to_options` itself because it renormalizes its output and
/// returns `options.len()`-shaped vectors, not 13-element vectors.
///
/// ## Mass distribution
///
/// 1. Place `mass_dmg` on the dmg-pick slot (if it's in the legal set).
/// 2. Place `mass_switch` on the switch-pick slot (if legal AND distinct).
/// 3. Distribute the remaining `1 - placed` uniformly over the OTHER legal
///    slots (those present in `options` but not yet filled).
/// 4. If only the heuristic-picked slots are legal, renormalize so the
///    output sums to 1.0.
pub fn compute(
    state: &State,
    perspective: SidePerspective,
    options: &[MoveChoice],
    mass_dmg: f32,
    mass_switch: f32,
) -> Option<HeuristicPrior> {
    debug_assert!(mass_dmg >= 0.0 && mass_switch >= 0.0);
    debug_assert!(mass_dmg + mass_switch <= 1.0 + 1e-6);

    let dmg_pick = damage_calc_top_move(state, perspective);
    let switch_pick = matchup_switch_pick(state, perspective);

    if dmg_pick.is_none() && switch_pick.is_none() {
        return None;
    }

    let side = match perspective {
        SidePerspective::Side1 => &state.side_one,
        SidePerspective::Side2 => &state.side_two,
    };
    let active = side.get_active_immutable();

    // Build alphabetical perms — identical to map_policy_to_options.
    let move_ids = active_move_ids(active);
    let move_alpha_perm = alpha_perm_with_norm(&move_ids, move_name_norm);
    let switch_species = reserve_species(side);
    let switch_alpha_perm = alpha_perm_with_norm(&switch_species, pokemon_name_norm);

    // Resolve dmg_pick → 0..3 slot. Find which M0..M3 holds the pick id,
    // then read its alphabetical rank.
    let dmg_slot: Option<usize> = dmg_pick.and_then(|pick_id| {
        let mv = &active.moves;
        let m_slots = [&mv.m0, &mv.m1, &mv.m2, &mv.m3];
        for (i, m) in m_slots.iter().enumerate() {
            if m.id == pick_id {
                return move_alpha_perm.get(i).copied();
            }
        }
        None
    });

    // Resolve switch_pick → 4..8 slot. Find the reserve PokemonIndex
    // whose species matches the picked name, then look up its reserve
    // slot's alphabetical rank.
    let switch_slot: Option<usize> = switch_pick.and_then(|pick_name| {
        for (i, pkmn) in side.pokemon.into_iter().enumerate() {
            let pidx = match i {
                0 => crate::state::PokemonIndex::P0,
                1 => crate::state::PokemonIndex::P1,
                2 => crate::state::PokemonIndex::P2,
                3 => crate::state::PokemonIndex::P3,
                4 => crate::state::PokemonIndex::P4,
                _ => crate::state::PokemonIndex::P5,
            };
            if pkmn.id == pick_name {
                if let Some(reserve_slot) = reserve_slot_for(side, pidx) {
                    if let Some(rank) = switch_alpha_perm.get(reserve_slot).copied() {
                        return Some(4 + rank);
                    }
                }
            }
        }
        None
    });

    // Determine legal slots — replicate map_policy_to_options's per-option
    // resolution but record the slot index instead of the prob.
    let mut legal_slots: Vec<usize> = Vec::with_capacity(options.len());
    for opt in options {
        let slot_opt: Option<usize> = match opt {
            MoveChoice::Move(idx) => {
                let s = move_index_to_slot(*idx);
                move_alpha_perm.get(s).copied()
            }
            MoveChoice::MoveTera(idx) => {
                let s = move_index_to_slot(*idx);
                move_alpha_perm.get(s).copied().map(|r| 9 + r)
            }
            MoveChoice::MoveMega(idx) => {
                // Same approximation as map_policy_to_options: collapse
                // mega onto the base move slot.
                let s = move_index_to_slot(*idx);
                move_alpha_perm.get(s).copied()
            }
            MoveChoice::Switch(pidx) => match reserve_slot_for(side, *pidx) {
                Some(reserve_slot) => switch_alpha_perm.get(reserve_slot).copied().map(|r| 4 + r),
                None => None,
            },
            MoveChoice::None => None,
        };
        if let Some(s) = slot_opt {
            if s < ACTION_DIM && !legal_slots.contains(&s) {
                legal_slots.push(s);
            }
        }
    }

    let mut probs = [0.0_f32; ACTION_DIM];
    let mut placed_mass = 0.0_f32;

    if let Some(s) = dmg_slot {
        if legal_slots.contains(&s) {
            probs[s] = mass_dmg;
            placed_mass += mass_dmg;
        }
    }
    if let Some(s) = switch_slot {
        if legal_slots.contains(&s) && probs[s] == 0.0 {
            probs[s] = mass_switch;
            placed_mass += mass_switch;
        }
    }

    if placed_mass <= 0.0 {
        // Both picks resolved to slots not in `options` (e.g., volatile
        // trap excluded the switch, and dmg pick is somehow unmapped).
        return None;
    }

    let remaining = 1.0 - placed_mass;
    let unfilled: Vec<usize> = legal_slots
        .iter()
        .filter(|s| probs[**s] == 0.0)
        .copied()
        .collect();
    if !unfilled.is_empty() && remaining > 0.0 {
        let share = remaining / unfilled.len() as f32;
        for s in unfilled {
            probs[s] = share;
        }
    } else if remaining > 0.0 {
        // Only the heuristic-picked slots are legal; renormalize to 1.0.
        let total: f32 = probs.iter().sum();
        if total > 0.0 {
            for p in probs.iter_mut() {
                *p /= total;
            }
        }
    }

    Some(HeuristicPrior {
        probs,
        damage_calc_pick: dmg_pick,
        matchup_switch_pick: switch_pick,
    })
}
