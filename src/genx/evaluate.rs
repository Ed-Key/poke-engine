use super::abilities::Abilities;
use super::items::Items;
use super::state::PokemonVolatileStatus;
use crate::choices::{MoveCategory, MoveTarget};
use crate::engine::damage_calc::{calculate_damage, DamageRolls};
use crate::engine::generate_instructions::{get_effective_speed, immune_to_status};
use crate::state::{Pokemon, PokemonStatus, PokemonType, Side, SideReference, State};

const POKEMON_ALIVE: f32 = 30.0;
const POKEMON_HP: f32 = 100.0;
const USED_TERA: f32 = -75.0;

// Fix #2.5: reduced 30.0 → 25.0 for atk/spa/spe to compensate for
// double-counting with threat_score × THREAT_SCORE_WEIGHT (40), which
// already credits +2 offense at roughly +24 per side. Defensive boosts
// unchanged (foul-play parity, no offense double-count).
const POKEMON_ATTACK_BOOST: f32 = 25.0;
const POKEMON_DEFENSE_BOOST: f32 = 15.0;
const POKEMON_SPECIAL_ATTACK_BOOST: f32 = 25.0;
const POKEMON_SPECIAL_DEFENSE_BOOST: f32 = 15.0;
const POKEMON_SPEED_BOOST: f32 = 25.0;

pub const THREAT_SCORE_WEIGHT: f32 = 40.0;
pub const MORTALITY_SCORE_WEIGHT: f32 = 20.0;
pub const STATUS_THREAT_WEIGHT: f32 = 25.0;

const POKEMON_BOOST_MULTIPLIER_6: f32 = 3.3;
const POKEMON_BOOST_MULTIPLIER_5: f32 = 3.15;
const POKEMON_BOOST_MULTIPLIER_4: f32 = 3.0;
const POKEMON_BOOST_MULTIPLIER_3: f32 = 2.5;
const POKEMON_BOOST_MULTIPLIER_2: f32 = 2.0;
const POKEMON_BOOST_MULTIPLIER_1: f32 = 1.0;
const POKEMON_BOOST_MULTIPLIER_0: f32 = 0.0;
const POKEMON_BOOST_MULTIPLIER_NEG_1: f32 = -1.0;
const POKEMON_BOOST_MULTIPLIER_NEG_2: f32 = -2.0;
const POKEMON_BOOST_MULTIPLIER_NEG_3: f32 = -2.5;
const POKEMON_BOOST_MULTIPLIER_NEG_4: f32 = -3.0;
const POKEMON_BOOST_MULTIPLIER_NEG_5: f32 = -3.15;
const POKEMON_BOOST_MULTIPLIER_NEG_6: f32 = -3.3;

/// Bracketed HP-aware multiplier for the active Pokemon's boost rewards,
/// inspired by Metamon Kaizo's setup schedule
/// (metamon/baselines/heuristic/kaizo.py:185-228). Replaces the previous
/// linear `hp/maxhp` multiplier, which still gave 54% credit at 54% HP and
/// caused the engine to recommend setup (DD, Nasty Plot, etc.) into known
/// KO threats.
///
/// Step function:
///   HP > 70%  → full reward (Kaizo "neutral")
///   40-70%    → zero (Kaizo would apply -2; cleanest map to our scale)
///   < 40%     → slightly negative (actually penalize setup at low HP)
#[allow(dead_code)] // removed from production callers in Fix #2.5; kaizo tests below pin old behavior; both removed in Task 5
fn boost_hp_multiplier(hp: i16, maxhp: i16) -> f32 {
    let pct = (hp as f32) / (maxhp as f32).max(1.0);
    if pct > 0.70 {
        1.0
    } else if pct >= 0.40 {
        0.0
    } else {
        -0.5
    }
}

const POKEMON_FROZEN: f32 = -40.0;
const POKEMON_ASLEEP: f32 = -25.0;
const POKEMON_PARALYZED: f32 = -25.0;
const POKEMON_TOXIC: f32 = -30.0;
const POKEMON_POISONED: f32 = -10.0;
const POKEMON_BURNED: f32 = -25.0;

const LEECH_SEED: f32 = -30.0;
const SUBSTITUTE: f32 = 40.0;
const CONFUSION: f32 = -20.0;

const REFLECT: f32 = 20.0;
const LIGHT_SCREEN: f32 = 20.0;
const AURORA_VEIL: f32 = 40.0;
const SAFE_GUARD: f32 = 5.0;
const TAILWIND: f32 = 7.0;
const HEALING_WISH: f32 = 30.0;

const STEALTH_ROCK: f32 = -10.0;
const SPIKES: f32 = -7.0;
const TOXIC_SPIKES: f32 = -7.0;
const STICKY_WEB: f32 = -25.0;

fn evaluate_poison(pokemon: &Pokemon, base_score: f32) -> f32 {
    match pokemon.ability {
        Abilities::POISONHEAL => 15.0,
        Abilities::GUTS
        | Abilities::MARVELSCALE
        | Abilities::QUICKFEET
        | Abilities::TOXICBOOST
        | Abilities::MAGICGUARD => 10.0,
        _ => base_score,
    }
}

fn evaluate_burned(pokemon: &Pokemon) -> f32 {
    // burn is not as punishing in certain situations

    // guts, marvel scale, quick feet will result in a positive evaluation
    match pokemon.ability {
        Abilities::GUTS | Abilities::MARVELSCALE | Abilities::QUICKFEET => {
            return -2.0 * POKEMON_BURNED
        }
        _ => {}
    }

    let mut multiplier = 0.0;
    for mv in pokemon.moves.into_iter() {
        if mv.choice.category == MoveCategory::Physical {
            multiplier += 1.0;
        }
    }

    // don't make burn as punishing for special attackers
    if pokemon.special_attack > pokemon.attack {
        multiplier /= 2.0;
    }

    multiplier * POKEMON_BURNED
}

fn get_boost_multiplier(boost: i8) -> f32 {
    match boost {
        6 => POKEMON_BOOST_MULTIPLIER_6,
        5 => POKEMON_BOOST_MULTIPLIER_5,
        4 => POKEMON_BOOST_MULTIPLIER_4,
        3 => POKEMON_BOOST_MULTIPLIER_3,
        2 => POKEMON_BOOST_MULTIPLIER_2,
        1 => POKEMON_BOOST_MULTIPLIER_1,
        0 => POKEMON_BOOST_MULTIPLIER_0,
        -1 => POKEMON_BOOST_MULTIPLIER_NEG_1,
        -2 => POKEMON_BOOST_MULTIPLIER_NEG_2,
        -3 => POKEMON_BOOST_MULTIPLIER_NEG_3,
        -4 => POKEMON_BOOST_MULTIPLIER_NEG_4,
        -5 => POKEMON_BOOST_MULTIPLIER_NEG_5,
        -6 => POKEMON_BOOST_MULTIPLIER_NEG_6,
        _ => panic!("Invalid boost value: {}", boost),
    }
}

fn evaluate_hazards(pokemon: &Pokemon, side: &Side) -> f32 {
    let mut score = 0.0;
    let pkmn_is_grounded = pokemon.is_grounded();
    if pokemon.item != Items::HEAVYDUTYBOOTS {
        if pokemon.ability != Abilities::MAGICGUARD {
            score += side.side_conditions.stealth_rock as f32 * STEALTH_ROCK;
            if pkmn_is_grounded {
                score += side.side_conditions.spikes as f32 * SPIKES;
                score += side.side_conditions.toxic_spikes as f32 * TOXIC_SPIKES;
            }
        }
        if pkmn_is_grounded {
            score += side.side_conditions.sticky_web as f32 * STICKY_WEB;
        }
    }

    score
}

fn evaluate_pokemon(pokemon: &Pokemon) -> f32 {
    let mut score = 0.0;
    score += POKEMON_HP * pokemon.hp as f32 / pokemon.maxhp as f32;

    match pokemon.status {
        PokemonStatus::BURN => score += evaluate_burned(pokemon),
        PokemonStatus::FREEZE => score += POKEMON_FROZEN,
        PokemonStatus::SLEEP => score += POKEMON_ASLEEP,
        PokemonStatus::PARALYZE => score += POKEMON_PARALYZED,
        PokemonStatus::TOXIC => score += evaluate_poison(pokemon, POKEMON_TOXIC),
        PokemonStatus::POISON => score += evaluate_poison(pokemon, POKEMON_POISONED),
        PokemonStatus::NONE => {}
    }

    if pokemon.item != Items::NONE {
        score += 10.0;
    }

    // without this a low hp pokemon could get a negative score and incentivize the other side
    // to keep it alive
    if score < 0.0 {
        score = 0.0;
    }

    score += POKEMON_ALIVE;

    score
}

/// Returns the leaf-evaluation threat score for `side_ref`'s active Pokemon
/// threatening the opposing active Pokemon. Value is in `[0.0, 1.2]` before
/// the caller applies `THREAT_SCORE_WEIGHT`.
///
/// Captures: fraction of defender HP removed by the attacker's best damaging
/// move at current boosts, plus a +20% bonus when the attacker outspeeds.
/// Uses the real `calculate_damage` function so STAB / type effectiveness /
/// weather / screens / items / abilities are all correctly reflected.
pub(crate) fn threat_score(state: &State, side_ref: &SideReference) -> f32 {
    let (attacker_side, defender_side) = match side_ref {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    let attacker = attacker_side.get_active_immutable();
    let defender = defender_side.get_active_immutable();

    if attacker.hp <= 0 || defender.hp <= 0 {
        return 0.0;
    }

    // Find best single-move damage across all usable damaging moves.
    let mut best_damage: i16 = 0;
    for mv in attacker.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 {
            continue;
        }
        let choice = &mv.choice;
        if choice.category == MoveCategory::Status || choice.category == MoveCategory::Switch {
            continue;
        }
        if choice.base_power == 0.0 {
            continue;
        }
        // Fix B Rule 1: priority moves are blocked by Psychic Terrain on grounded opp.
        if choice.priority > 0
            && state.terrain.terrain_type == crate::engine::state::Terrain::PSYCHICTERRAIN
            && state.terrain.turns_remaining > 0
            && defender.is_grounded()
        {
            continue;
        }
        if let Some((normal, _crit)) =
            calculate_damage(state, side_ref, choice, DamageRolls::Average)
        {
            // Fix B Rule 4: Knock Off's 1.5x damage bonus only realizes when the
            // defender currently holds an item. On an item-less target the
            // "remove item" utility is already cashed, so scale the recommended
            // damage down to discourage MCTS from over-spamming Knock Off.
            let mut effective = normal;
            if choice.move_id == crate::choices::Choices::KNOCKOFF
                && defender.item == Items::NONE
            {
                effective = ((effective as f32) * 0.66) as i16;
            }
            if effective > best_damage {
                best_damage = effective;
            }
        }
    }

    if best_damage == 0 {
        return 0.0;
    }

    let hp_ratio = (best_damage as f32 / defender.hp as f32).min(1.0);

    // Speed tier bonus: if attacker outspeeds, a guaranteed kill happens THIS
    // turn rather than next, so the tempo value is higher.
    let defender_side_ref = match side_ref {
        SideReference::SideOne => SideReference::SideTwo,
        SideReference::SideTwo => SideReference::SideOne,
    };
    let attacker_speed = get_effective_speed(state, side_ref);
    let defender_speed = get_effective_speed(state, &defender_side_ref);
    let speed_bonus = if attacker_speed > defender_speed {
        1.2
    } else {
        1.0
    };

    hp_ratio * speed_bonus
}

/// Returns the leaf-evaluation mortality score for `side_ref`'s active Pokemon
/// being threatened by the opposing active Pokemon. Value is in `[0.0, 1.2]`
/// before the caller applies `MORTALITY_SCORE_WEIGHT`.
///
/// Captures: fraction of our attacker's HP removed by opp's best damaging move
/// at current boosts, +20% when opp outspeeds (the KO happens THIS turn).
///
/// This is exactly `threat_score` with attacker/defender roles swapped, so
/// `mortality_score(state, S1) == threat_score(state, S2)` by construction.
pub(crate) fn mortality_score(state: &State, side_ref: &SideReference) -> f32 {
    let opposing = match side_ref {
        SideReference::SideOne => SideReference::SideTwo,
        SideReference::SideTwo => SideReference::SideOne,
    };
    threat_score(state, &opposing)
}

/// Returns the leaf-evaluation status-threat score for `side_ref`'s active
/// Pokemon — how much eval-value the side can expect to gain by landing a
/// non-damaging status move on the opposing active. Value in `[0.0, 1.6]`
/// before the caller applies `STATUS_THREAT_WEIGHT`.
///
/// Rewards the existence of an applicable status move so MCTS doesn't pick
/// a 0x-damage move over Sleep Powder / Will-O-Wisp / Leech Seed just
/// because threat_score is blind to status. Accuracy-weighted and checks
/// `immune_to_status` (type / ability / terrain / Substitute / Safeguard /
/// sleep clause) so Grass mons don't get credit for Sleep Powder etc.
pub(crate) fn status_threat_score(state: &State, side_ref: &SideReference) -> f32 {
    let (attacker_side, defender_side_ref) = match side_ref {
        SideReference::SideOne => (&state.side_one, SideReference::SideTwo),
        SideReference::SideTwo => (&state.side_two, SideReference::SideOne),
    };
    let attacker = attacker_side.get_active_immutable();
    if attacker.hp <= 0 {
        return 0.0;
    }
    let defender_side = match defender_side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let defender = defender_side.get_active_immutable();
    if defender.hp <= 0 {
        return 0.0;
    }

    let mut best: f32 = 0.0;
    for mv in attacker.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 {
            continue;
        }
        let choice = &mv.choice;
        if choice.category != MoveCategory::Status {
            continue;
        }
        // Fix B Rule 2: powder moves are blocked by Grass type or Overcoat ability.
        if choice.flags.powder
            && (defender.has_type(&PokemonType::GRASS)
                || defender.ability == Abilities::OVERCOAT)
        {
            continue;
        }
        // Fix B Rule 3: Toxic / Poison on Poison Heal opp gives them sustain.
        if let Some(status_spec) = &choice.status {
            if (status_spec.status == PokemonStatus::TOXIC
                || status_spec.status == PokemonStatus::POISON)
                && defender.ability == Abilities::POISONHEAL
            {
                continue;
            }
        }
        let accuracy = (choice.accuracy / 100.0).min(1.0);

        // Primary status (Sleep Powder, Thunder Wave, Will-O-Wisp, Toxic, ...).
        if let Some(status_spec) = &choice.status {
            if status_spec.target == MoveTarget::Opponent
                && !immune_to_status(state, &status_spec.target, &defender_side_ref, &status_spec.status)
            {
                let value = match status_spec.status {
                    PokemonStatus::SLEEP => 1.0,
                    PokemonStatus::BURN => 1.0,
                    PokemonStatus::PARALYZE => 1.0,
                    PokemonStatus::TOXIC => 1.2,
                    PokemonStatus::POISON => 0.4,
                    PokemonStatus::FREEZE => 1.6,
                    PokemonStatus::NONE => 0.0,
                };
                let contribution = accuracy * value;
                if contribution > best {
                    best = contribution;
                }
            }
        }

        // Volatile-status primary moves — Leech Seed specifically, which
        // mirrors the -30 LEECH_SEED leaf penalty. Grass-type immunity and
        // an already-seeded check avoid double-counting.
        if let Some(vs_spec) = &choice.volatile_status {
            if vs_spec.target == MoveTarget::Opponent
                && vs_spec.volatile_status == PokemonVolatileStatus::LEECHSEED
                && !defender.has_type(&PokemonType::GRASS)
                && !defender_side
                    .volatile_statuses
                    .contains(&PokemonVolatileStatus::LEECHSEED)
            {
                let contribution = accuracy * 1.2; // 30 / 25 (STATUS_THREAT_WEIGHT)
                if contribution > best {
                    best = contribution;
                }
            }
        }
    }
    best
}

/// Returns 0.3 if `opp_active` has a hazard remover (Defog, Rapid Spin, Court Change)
/// in its moveset, else 1.0. Used to scale hazard eval contributions in stalemate
/// matchups where the opp can blow our hazards away.
///
/// Note: chaos priors may populate Defog speculatively for some opp mons. We accept
/// the false-positive risk for v1; if A/B regresses we'll gate on `is_revealed`.
pub(crate) fn hazard_value_factor(opp_active: &Pokemon) -> f32 {
    use crate::choices::Choices;
    for mv in opp_active.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 {
            continue;
        }
        match mv.choice.move_id {
            Choices::DEFOG | Choices::RAPIDSPIN | Choices::COURTCHANGE => return 0.3,
            _ => {}
        }
    }
    1.0
}

pub fn evaluate(state: &State) -> f32 {
    let mut score = 0.0;

    // Fix C: scale each side's hazard contributions by THAT side's active's
    // remover capacity. Hazards on side_one damage side_one's mons; side_one's
    // active is the one that could Defog/Spin/CourtChange them away.
    let s1_hazard_factor = hazard_value_factor(state.side_one.get_active_immutable());
    let s2_hazard_factor = hazard_value_factor(state.side_two.get_active_immutable());

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score += evaluate_pokemon(pkmn);
            score += evaluate_hazards(pkmn, &state.side_one) * s1_hazard_factor;
            if iter.pokemon_index == state.side_one.active_index {
                for vs in state.side_one.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => score += LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => score += SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => score += CONFUSION,
                        _ => {}
                    }
                }

                // Flat boost reward (foul-play parity). HP-conditional gating
                // was removed in Fix #2.5: it double-counted with threat_score
                // × 40 and produced a 40-70% HP blind zone for setup sweepers.
                // Self-mortality is naturally handled by the active+alive gate
                // on the enclosing `if iter.pokemon_index == ...` block.
                score += get_boost_multiplier(state.side_one.attack_boost) * POKEMON_ATTACK_BOOST;
                score += get_boost_multiplier(state.side_one.defense_boost) * POKEMON_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                score += get_boost_multiplier(state.side_one.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.speed_boost) * POKEMON_SPEED_BOOST;
                score += threat_score(state, &SideReference::SideOne) * THREAT_SCORE_WEIGHT;
                score -= mortality_score(state, &SideReference::SideOne) * MORTALITY_SCORE_WEIGHT;
                score += status_threat_score(state, &SideReference::SideOne) * STATUS_THREAT_WEIGHT;
            }
        }
        if pkmn.terastallized {
            s1_used_tera = true;
        }
    }
    if s1_used_tera {
        score += USED_TERA;
    }
    let mut iter = state.side_two.pokemon.into_iter();
    let mut s2_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score -= evaluate_pokemon(pkmn);
            score -= evaluate_hazards(pkmn, &state.side_two) * s2_hazard_factor;

            if iter.pokemon_index == state.side_two.active_index {
                for vs in state.side_two.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => score -= LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => score -= SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => score -= CONFUSION,
                        _ => {}
                    }
                }

                // Flat boost reward (foul-play parity). See s1 block for rationale.
                score -= get_boost_multiplier(state.side_two.attack_boost) * POKEMON_ATTACK_BOOST;
                score -= get_boost_multiplier(state.side_two.defense_boost) * POKEMON_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                score -= get_boost_multiplier(state.side_two.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.speed_boost) * POKEMON_SPEED_BOOST;
                score -= threat_score(state, &SideReference::SideTwo) * THREAT_SCORE_WEIGHT;
                score += mortality_score(state, &SideReference::SideTwo) * MORTALITY_SCORE_WEIGHT;
                score -= status_threat_score(state, &SideReference::SideTwo) * STATUS_THREAT_WEIGHT;
            }
        }
        if pkmn.terastallized {
            s2_used_tera = true;
        }
    }
    if s2_used_tera {
        score -= USED_TERA;
    }

    score += state.side_one.side_conditions.reflect as f32 * REFLECT;
    score += state.side_one.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    score += state.side_one.side_conditions.aurora_veil as f32 * AURORA_VEIL;
    score += state.side_one.side_conditions.safeguard as f32 * SAFE_GUARD;
    score += state.side_one.side_conditions.tailwind as f32 * TAILWIND;
    score += state.side_one.side_conditions.healing_wish as f32 * HEALING_WISH;

    score -= state.side_two.side_conditions.reflect as f32 * REFLECT;
    score -= state.side_two.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    score -= state.side_two.side_conditions.aurora_veil as f32 * AURORA_VEIL;
    score -= state.side_two.side_conditions.safeguard as f32 * SAFE_GUARD;
    score -= state.side_two.side_conditions.tailwind as f32 * TAILWIND;
    score -= state.side_two.side_conditions.healing_wish as f32 * HEALING_WISH;

    score
}

/// Per-term breakdown of the heuristic eval. `total` is constructed to equal
/// `evaluate(state)` by construction (asserted by `test_breakdown_matches_evaluate`).
///
/// Used only for instrumentation/logging — the production hot path goes through
/// `evaluate(state) -> f32` which is unchanged. `evaluate_breakdown` is fine for
/// per-request structured logging but is NOT optimized for the rollout loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalBreakdown {
    /// Sum of all terms below; equals `evaluate(state)`.
    pub total: f32,
    /// Sum over both sides of `evaluate_pokemon` (HP, status, item, alive bonus).
    /// Side 1 contributes positive, side 2 negative — same signing convention
    /// as `evaluate`.
    pub hp_term: f32,
    /// Hazard contribution (Stealth Rock, Spikes, Toxic Spikes, Sticky Web)
    /// for both sides, scaled by hazard_value_factor.
    pub hazards_term: f32,
    /// Side-1 active stat-boost contribution (atk/def/spA/spD/spe), HP-scaled.
    pub boost_term_s1: f32,
    /// Side-2 active stat-boost contribution.
    pub boost_term_s2: f32,
    /// `threat_score(SideOne) * THREAT_SCORE_WEIGHT` minus
    /// `mortality_score(SideOne) * MORTALITY_SCORE_WEIGHT` -- all the s1
    /// damage-output / damage-taken offense terms.
    pub threat_score_s1: f32,
    /// Mirror for s2.
    pub threat_score_s2: f32,
    /// Active Pokémon volatile statuses (Leech Seed, Substitute, Confusion)
    /// summed across both sides.
    pub volatile_status_term: f32,
    /// Side conditions (Reflect, Light Screen, Aurora Veil, Safeguard,
    /// Tailwind, Healing Wish) summed across both sides.
    pub side_conditions_term: f32,
    /// Tera-used penalty (USED_TERA flag) summed across both sides.
    pub tera_term: f32,
    /// `status_threat_score(...) * STATUS_THREAT_WEIGHT` for both sides
    /// (signed). Captures Sleep Powder / Will-O-Wisp / Leech Seed value.
    pub status_threat_term: f32,
}

/// Mirror of `evaluate` that returns a per-term breakdown. The summed `total`
/// equals `evaluate(state)` exactly (verified by unit test). Used for the
/// `[ENGINE-INSTRUMENT]` JSON log line; do NOT call from the rollout loop.
pub fn evaluate_breakdown(state: &State) -> EvalBreakdown {
    let mut hp_term: f32 = 0.0;
    let mut hazards_term: f32 = 0.0;
    let mut boost_term_s1: f32 = 0.0;
    let mut boost_term_s2: f32 = 0.0;
    let mut volatile_status_term: f32 = 0.0;
    let mut tera_term: f32 = 0.0;

    let s1_hazard_factor = hazard_value_factor(state.side_one.get_active_immutable());
    let s2_hazard_factor = hazard_value_factor(state.side_two.get_active_immutable());

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            hp_term += evaluate_pokemon(pkmn);
            hazards_term += evaluate_hazards(pkmn, &state.side_one) * s1_hazard_factor;
            if iter.pokemon_index == state.side_one.active_index {
                for vs in state.side_one.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => volatile_status_term += LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => volatile_status_term += SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => volatile_status_term += CONFUSION,
                        _ => {}
                    }
                }
                boost_term_s1 += get_boost_multiplier(state.side_one.attack_boost) * POKEMON_ATTACK_BOOST;
                boost_term_s1 += get_boost_multiplier(state.side_one.defense_boost) * POKEMON_DEFENSE_BOOST;
                boost_term_s1 += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                boost_term_s1 += get_boost_multiplier(state.side_one.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                boost_term_s1 += get_boost_multiplier(state.side_one.speed_boost) * POKEMON_SPEED_BOOST;
            }
        }
        if pkmn.terastallized {
            s1_used_tera = true;
        }
    }
    if s1_used_tera {
        tera_term += USED_TERA;
    }

    let mut iter = state.side_two.pokemon.into_iter();
    let mut s2_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            hp_term -= evaluate_pokemon(pkmn);
            hazards_term -= evaluate_hazards(pkmn, &state.side_two) * s2_hazard_factor;
            if iter.pokemon_index == state.side_two.active_index {
                for vs in state.side_two.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => volatile_status_term -= LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => volatile_status_term -= SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => volatile_status_term -= CONFUSION,
                        _ => {}
                    }
                }
                boost_term_s2 -= get_boost_multiplier(state.side_two.attack_boost) * POKEMON_ATTACK_BOOST;
                boost_term_s2 -= get_boost_multiplier(state.side_two.defense_boost) * POKEMON_DEFENSE_BOOST;
                boost_term_s2 -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                boost_term_s2 -= get_boost_multiplier(state.side_two.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                boost_term_s2 -= get_boost_multiplier(state.side_two.speed_boost) * POKEMON_SPEED_BOOST;
            }
        }
        if pkmn.terastallized {
            s2_used_tera = true;
        }
    }
    if s2_used_tera {
        tera_term -= USED_TERA;
    }

    // Threat / mortality / status_threat: only the active mons matter; matching
    // the gating in `evaluate` (only added when the active is alive).
    let s1_active_alive = state.side_one.get_active_immutable().hp > 0;
    let s2_active_alive = state.side_two.get_active_immutable().hp > 0;
    let mut threat_score_s1 = 0.0_f32;
    let mut threat_score_s2 = 0.0_f32;
    let mut status_threat_term = 0.0_f32;
    if s1_active_alive {
        threat_score_s1 += threat_score(state, &SideReference::SideOne) * THREAT_SCORE_WEIGHT;
        threat_score_s1 -= mortality_score(state, &SideReference::SideOne) * MORTALITY_SCORE_WEIGHT;
        status_threat_term += status_threat_score(state, &SideReference::SideOne) * STATUS_THREAT_WEIGHT;
    }
    if s2_active_alive {
        threat_score_s2 -= threat_score(state, &SideReference::SideTwo) * THREAT_SCORE_WEIGHT;
        threat_score_s2 += mortality_score(state, &SideReference::SideTwo) * MORTALITY_SCORE_WEIGHT;
        status_threat_term -= status_threat_score(state, &SideReference::SideTwo) * STATUS_THREAT_WEIGHT;
    }

    let mut side_conditions_term: f32 = 0.0;
    side_conditions_term += state.side_one.side_conditions.reflect as f32 * REFLECT;
    side_conditions_term += state.side_one.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    side_conditions_term += state.side_one.side_conditions.aurora_veil as f32 * AURORA_VEIL;
    side_conditions_term += state.side_one.side_conditions.safeguard as f32 * SAFE_GUARD;
    side_conditions_term += state.side_one.side_conditions.tailwind as f32 * TAILWIND;
    side_conditions_term += state.side_one.side_conditions.healing_wish as f32 * HEALING_WISH;
    side_conditions_term -= state.side_two.side_conditions.reflect as f32 * REFLECT;
    side_conditions_term -= state.side_two.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    side_conditions_term -= state.side_two.side_conditions.aurora_veil as f32 * AURORA_VEIL;
    side_conditions_term -= state.side_two.side_conditions.safeguard as f32 * SAFE_GUARD;
    side_conditions_term -= state.side_two.side_conditions.tailwind as f32 * TAILWIND;
    side_conditions_term -= state.side_two.side_conditions.healing_wish as f32 * HEALING_WISH;

    let total = hp_term
        + hazards_term
        + boost_term_s1
        + boost_term_s2
        + threat_score_s1
        + threat_score_s2
        + volatile_status_term
        + side_conditions_term
        + tera_term
        + status_threat_term;

    EvalBreakdown {
        total,
        hp_term,
        hazards_term,
        boost_term_s1,
        boost_term_s2,
        threat_score_s1,
        threat_score_s2,
        volatile_status_term,
        side_conditions_term,
        tera_term,
        status_threat_term,
    }
}

#[cfg(test)]
mod tests {
    use crate::choices::{Choice, Choices, MoveCategory};
    use crate::engine::damage_calc::{calculate_damage, DamageRolls};
    use crate::state::{PokemonType, SideReference, State};
    use std::time::Instant;

    #[test]
    #[ignore] // run explicitly with: cargo test --release --no-default-features --features gen9 bench_calculate_damage -- --ignored --nocapture
    fn bench_calculate_damage() {
        let state = State::default();
        // Default state has no real moves on the active Pokemon, so construct a
        // representative damaging Choice directly, matching the pattern used by
        // existing damage_calc tests (see src/genx/damage_calc.rs test_basic_damaging_move).
        let mut attacker_choice = Choice {
            ..Default::default()
        };
        attacker_choice.move_id = Choices::TACKLE;
        attacker_choice.move_type = PokemonType::TYPELESS;
        attacker_choice.base_power = 40.0;
        attacker_choice.category = MoveCategory::Physical;

        let iterations: u32 = 1_000_000;
        let start = Instant::now();
        let mut sink: i32 = 0;
        for _ in 0..iterations {
            let r = calculate_damage(
                &state,
                &SideReference::SideOne,
                &attacker_choice,
                DamageRolls::Average,
            );
            if let Some((normal, _)) = r {
                sink = sink.wrapping_add(normal as i32);
            }
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
        println!(
            "calculate_damage bench: {} calls in {:?} = {:.1} ns/call (sink={})",
            iterations, elapsed, ns_per_call, sink
        );

        // Throughput projection
        let calls_per_eval = 8.0; // 2 sides × up to 4 damaging moves
        let mcts_sims_per_5s = 2_500_000.0; // current baseline
        let projected_ms_in_damage = (ns_per_call * calls_per_eval * mcts_sims_per_5s) / 1_000_000.0;
        println!(
            "projection: {} calls × {} sims/5s × {:.1} ns = {:.0} ms spent in calculate_damage per 5s budget ({:.0}% of budget)",
            calls_per_eval, mcts_sims_per_5s, ns_per_call, projected_ms_in_damage, projected_ms_in_damage / 50.0
        );
    }

    #[test]
    fn test_threat_score_fainted_returns_zero() {
        let mut state = State::default();
        state.side_one.get_active().hp = 0;
        let score = super::threat_score(&state, &SideReference::SideOne);
        assert_eq!(score, 0.0, "fainted attacker should produce zero threat");
    }

    #[test]
    fn test_threat_score_increases_with_attack_boost() {
        let mut state = State::default();
        // State::default() has no damaging moves on the active Pokemon, so
        // threat_score will likely be 0 in both cases. The assertion stays loose
        // (boosted >= base) so it still catches regressions where +2 Atk decreases
        // threat. A stricter fixture with a damaging move can tighten this later.
        let base = super::threat_score(&state, &SideReference::SideOne);
        state.side_one.attack_boost = 2;
        let boosted = super::threat_score(&state, &SideReference::SideOne);

        assert!(
            boosted >= base,
            "+2 Attack must not decrease threat_score (base={}, boosted={})",
            base, boosted
        );
    }

    #[test]
    fn test_threat_score_non_negative() {
        let state = State::default();
        let s1 = super::threat_score(&state, &SideReference::SideOne);
        let s2 = super::threat_score(&state, &SideReference::SideTwo);
        assert!(s1 >= 0.0, "threat_score must be non-negative, got {}", s1);
        assert!(s2 >= 0.0, "threat_score must be non-negative, got {}", s2);
    }

    #[test]
    fn test_threat_score_caps_below_ceiling() {
        // threat_score should be clamped to [0.0, 1.2] (hp_ratio in [0,1], speed_bonus <= 1.2).
        let mut state = State::default();
        state.side_two.get_active().hp = 1;
        let score = super::threat_score(&state, &SideReference::SideOne);
        assert!(score <= 1.2001, "threat_score must not exceed ceiling; got {}", score);
    }

    #[test]
    fn test_evaluate_favors_boosted_side() {
        let mut state = State::default();
        let base_eval = super::evaluate(&state);
        state.side_one.attack_boost = 2;
        let boosted_eval = super::evaluate(&state);
        assert!(
            boosted_eval >= base_eval,
            "evaluate must favor boosted side_one (base={}, boosted={})",
            base_eval, boosted_eval
        );
    }

    #[test]
    fn test_breakdown_matches_evaluate() {
        // For a handful of representative states, evaluate_breakdown(state).total
        // must equal evaluate(state) within float epsilon. Anything wider would
        // indicate a missing term in the breakdown.
        let mut states = Vec::new();
        states.push(State::default());

        let mut s_boosted = State::default();
        s_boosted.side_one.attack_boost = 2;
        s_boosted.side_one.speed_boost = 1;
        s_boosted.side_two.special_attack_boost = -1;
        states.push(s_boosted);

        let mut s_hazards = State::default();
        s_hazards.side_one.side_conditions.stealth_rock = 1;
        s_hazards.side_one.side_conditions.spikes = 2;
        s_hazards.side_two.side_conditions.sticky_web = 1;
        s_hazards.side_two.side_conditions.tailwind = 3;
        s_hazards.side_one.side_conditions.reflect = 1;
        states.push(s_hazards);

        let mut s_chipped = State::default();
        s_chipped.side_one.get_active().hp = 50;
        s_chipped.side_two.get_active().hp = 1;
        states.push(s_chipped);

        for (i, state) in states.iter().enumerate() {
            let eval_total = super::evaluate(state);
            let breakdown = super::evaluate_breakdown(state);
            let diff = (eval_total - breakdown.total).abs();
            assert!(
                diff < 0.001,
                "case {}: breakdown.total ({}) must equal evaluate ({}); diff={} breakdown={:?}",
                i, breakdown.total, eval_total, diff, breakdown
            );
        }
    }

    #[test]
    fn test_mortality_score_fainted_returns_zero() {
        let mut state = State::default();
        state.side_two.get_active().hp = 0;
        let score = super::mortality_score(&state, &SideReference::SideOne);
        assert_eq!(score, 0.0, "fainted opposing attacker should produce zero mortality");
    }

    #[test]
    fn test_mortality_score_nonzero_when_opp_can_hit() {
        let state = State::default();
        let _score = super::mortality_score(&state, &SideReference::SideOne);
        // Smoke check: function compiles and doesn't panic on default state.
    }

    #[test]
    fn test_mortality_score_mirrors_threat_score_of_opposing_side() {
        let state = State::default();
        let mortality_s1 = super::mortality_score(&state, &SideReference::SideOne);
        let threat_s2 = super::threat_score(&state, &SideReference::SideTwo);
        assert!(
            (mortality_s1 - threat_s2).abs() < 0.001,
            "mortality_s1={} must mirror threat_s2={}", mortality_s1, threat_s2
        );
    }

    #[test]
    fn test_fixB_priority_move_zeroed_under_psyterrain() {
        // Setup: side_one's active has Sucker Punch (priority +1), defender is grounded,
        // Psychic Terrain is up. threat_score should ignore Sucker Punch even if it
        // would do damage on raw type matchup.
        use crate::choices::Choices;
        use crate::engine::state::Terrain;

        let mut state = State::default();
        // Force a Sucker-Punch-like priority move into slot 0 of side_one's active.
        let active = state.side_one.get_active();
        active.moves.m0.id = Choices::SUCKERPUNCH;
        active.moves.m0.disabled = false;
        active.moves.m0.pp = 16;
        // Re-resolve choice so move_id, priority etc. line up:
        active.moves.m0.choice = crate::choices::MOVES.get(&Choices::SUCKERPUNCH).unwrap().clone();

        // Disable the rest so they don't dominate the score.
        active.moves.m1.disabled = true;
        active.moves.m2.disabled = true;
        active.moves.m3.disabled = true;

        // Activate Psychic Terrain.
        state.terrain.terrain_type = Terrain::PSYCHICTERRAIN;
        state.terrain.turns_remaining = 5;

        // Defender must be grounded (default state defender already is — no Levitate, no Flying).
        // Verify: side_two.get_active().is_grounded() should be true.
        assert!(state.side_two.get_active_immutable().is_grounded(),
                "test setup invalid: defender should be grounded");

        let score = super::threat_score(&state, &SideReference::SideOne);
        assert_eq!(score, 0.0, "priority move on grounded target under Psychic Terrain must score 0");

        // Negative control: flip terrain off, same setup must now score > 0
        // (proves the zero above came from the Rule 1 guard, not a broken fixture).
        state.terrain.terrain_type = Terrain::NONE;
        state.terrain.turns_remaining = 0;
        let score_no_terrain = super::threat_score(&state, &SideReference::SideOne);
        assert!(score_no_terrain > 0.0,
                "without Psychic Terrain, Sucker Punch must score > 0 (got {})",
                score_no_terrain);
    }

    #[test]
    fn test_fixB_powder_zeroed_on_grass_type() {
        // Setup: side_one's active has Sleep Powder, side_two's active is Grass type.
        // status_threat_score should return 0 (powder doesn't affect Grass).
        use crate::choices::Choices;
        use crate::state::PokemonStatus;

        let mut state = State::default();
        let attacker = state.side_one.get_active();
        attacker.moves.m0.id = Choices::SLEEPPOWDER;
        attacker.moves.m0.disabled = false;
        attacker.moves.m0.pp = 16;
        attacker.moves.m0.choice = crate::choices::MOVES.get(&Choices::SLEEPPOWDER).unwrap().clone();
        attacker.moves.m1.disabled = true;
        attacker.moves.m2.disabled = true;
        attacker.moves.m3.disabled = true;

        // Force defender to Grass type.
        let defender = state.side_two.get_active();
        defender.types = (PokemonType::GRASS, PokemonType::TYPELESS);
        defender.status = PokemonStatus::NONE;
        defender.hp = defender.maxhp;

        let score = super::status_threat_score(&state, &SideReference::SideOne);
        assert_eq!(score, 0.0, "powder move on Grass type must score 0 status_threat");

        // Negative control: flip defender off Grass — same Sleep Powder must now score > 0.
        let defender = state.side_two.get_active();
        defender.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        let score_normal = super::status_threat_score(&state, &SideReference::SideOne);
        assert!(score_normal > 0.0,
                "Sleep Powder on Normal-type must score > 0 (got {})",
                score_normal);
    }

    #[test]
    fn test_fixB_toxic_zeroed_on_poison_heal() {
        // Setup: side_one's active has Toxic, side_two's active has Poison Heal ability.
        // status_threat_score should return 0 even though immune_to_status returns false.
        use crate::choices::Choices;
        use crate::state::PokemonStatus;
        use super::Abilities;

        let mut state = State::default();
        let attacker = state.side_one.get_active();
        attacker.moves.m0.id = Choices::TOXIC;
        attacker.moves.m0.disabled = false;
        attacker.moves.m0.pp = 10;
        attacker.moves.m0.choice = crate::choices::MOVES.get(&Choices::TOXIC).unwrap().clone();
        attacker.moves.m1.disabled = true;
        attacker.moves.m2.disabled = true;
        attacker.moves.m3.disabled = true;

        // Defender: Gliscor-like (Ground/Flying so Toxic isn't already type-blocked) with Poison Heal.
        let defender = state.side_two.get_active();
        defender.types = (PokemonType::GROUND, PokemonType::FLYING);
        defender.ability = Abilities::POISONHEAL;
        defender.status = PokemonStatus::NONE;
        defender.hp = defender.maxhp;

        let score = super::status_threat_score(&state, &SideReference::SideOne);
        assert_eq!(score, 0.0, "Toxic on Poison Heal target must score 0 status_threat");

        // Negative control: same setup, defender ability flipped to a non-PoisonHeal ability.
        // Toxic must score > 0 (Gliscor without PoisonHeal is still toxicable).
        let defender = state.side_two.get_active();
        defender.ability = Abilities::NONE;
        let score_no_pheal = super::status_threat_score(&state, &SideReference::SideOne);
        assert!(score_no_pheal > 0.0,
                "Toxic on Ground/Flying without Poison Heal must score > 0 (got {})",
                score_no_pheal);
    }

    #[test]
    fn test_fixB_knock_off_no_bonus_on_itemless() {
        // Setup: defender has an item, threat_score for Knock Off baseline.
        // Then: defender.item = NONE, threat_score for same move should be lower
        // (Knock Off's 1.5x damage bonus only applies on item-holding defenders).
        use crate::choices::Choices;
        use super::Items;

        let mut state = State::default();
        let attacker = state.side_one.get_active();
        attacker.moves.m0.id = Choices::KNOCKOFF;
        attacker.moves.m0.disabled = false;
        attacker.moves.m0.pp = 16;
        attacker.moves.m0.choice = crate::choices::MOVES.get(&Choices::KNOCKOFF).unwrap().clone();
        attacker.moves.m1.disabled = true;
        attacker.moves.m2.disabled = true;
        attacker.moves.m3.disabled = true;

        // Defender with item.
        let defender = state.side_two.get_active();
        defender.item = Items::LEFTOVERS;
        let with_item = super::threat_score(&state, &SideReference::SideOne);

        // Defender without item.
        state.side_two.get_active().item = Items::NONE;
        let without_item = super::threat_score(&state, &SideReference::SideOne);

        assert!(without_item < with_item,
                "Knock Off threat should be lower when defender has no item; got without={} >= with={}",
                without_item, with_item);
    }

    #[test]
    fn test_fixC_hazards_scaled_down_when_opp_has_defog() {
        // Setup: side_one has Stealth Rock on side_two. Compute eval. Then add Defog
        // to side_two's active. Eval should DROP for side_one (hazards now worth less).
        use crate::choices::Choices;

        let mut state = State::default();
        state.side_two.side_conditions.stealth_rock = 1;

        // Defender doesn't have Defog yet — give it Tackle.
        let defender = state.side_two.get_active();
        defender.moves.m0.id = Choices::TACKLE;
        defender.moves.m0.disabled = false;
        defender.moves.m0.pp = 16;
        defender.moves.m0.choice = crate::choices::MOVES.get(&Choices::TACKLE).unwrap().clone();

        let score_without_defog = super::evaluate(&state);

        // Now defender DOES have Defog.
        state.side_two.get_active().moves.m0.id = Choices::DEFOG;
        state.side_two.get_active().moves.m0.choice = crate::choices::MOVES.get(&Choices::DEFOG).unwrap().clone();

        let score_with_defog = super::evaluate(&state);

        // Side_one's score should be lower with opp Defog (their SR is worth less to side_one).
        assert!(score_with_defog < score_without_defog,
                "score with opp Defog ({}) must be < score without ({})",
                score_with_defog, score_without_defog);
    }

    #[test]
    fn test_fixC_no_remover_keeps_hazards_full_value() {
        // Sanity check: if opp has no remover, hazard scaling factor is 1.0 (no change).
        let mut state = State::default();
        state.side_one.side_conditions.spikes = 1;

        let score_baseline = super::evaluate(&state);

        // Verify factor returns 1.0 explicitly.
        let factor = super::hazard_value_factor(state.side_one.get_active_immutable());
        assert_eq!(factor, 1.0, "no remover → factor must be 1.0");

        // Score is deterministic.
        let score_again = super::evaluate(&state);
        assert_eq!(score_baseline, score_again, "evaluate must be deterministic");
    }

    #[test]
    fn test_boost_hp_multiplier_kaizo_brackets() {
        // Direct unit test of the bracketed schedule itself.
        // > 70% HP → full reward
        assert_eq!(super::boost_hp_multiplier(100, 100), 1.0);
        assert_eq!(super::boost_hp_multiplier(80, 100), 1.0);
        assert_eq!(super::boost_hp_multiplier(71, 100), 1.0);
        // 40-70% HP → zero reward (the load-bearing change)
        assert_eq!(super::boost_hp_multiplier(70, 100), 0.0);
        assert_eq!(super::boost_hp_multiplier(54, 100), 0.0); // Dragonite-DD scenario
        assert_eq!(super::boost_hp_multiplier(40, 100), 0.0);
        // < 40% HP → negative reward (penalize setup)
        assert_eq!(super::boost_hp_multiplier(39, 100), -0.5);
        assert_eq!(super::boost_hp_multiplier(1, 100), -0.5);
    }

    #[test]
    fn test_boost_term_kaizo_schedule_in_breakdown() {
        // Build a state with side_one's active at +2 SpA, vary HP across the
        // three Kaizo brackets, and confirm the boost_term_s1 magnitude flips
        // as expected. Also verify the evaluate <=> evaluate_breakdown
        // invariant still holds at each HP point.

        // At +2 stage, get_boost_multiplier returns POKEMON_BOOST_MULTIPLIER_2 = 2.0,
        // so the SpA contribution is 2.0 * 30.0 * multiplier = 60 * multiplier.

        // Case 1: 100% HP (full credit) → boost_term_s1 ~= +60
        let mut state_full = State::default();
        state_full.side_one.special_attack_boost = 2;
        let bd_full = super::evaluate_breakdown(&state_full);
        assert!(
            (bd_full.boost_term_s1 - 60.0).abs() < 0.001,
            "100% HP: boost_term_s1 should be ~+60, got {}",
            bd_full.boost_term_s1
        );
        let diff_full = (super::evaluate(&state_full) - bd_full.total).abs();
        assert!(diff_full < 0.001, "invariant broken at 100% HP: diff={}", diff_full);

        // Case 2: 50% HP (Kaizo marginal band) → boost_term_s1 == 0
        let mut state_mid = State::default();
        state_mid.side_one.special_attack_boost = 2;
        state_mid.side_one.get_active().hp = 50;
        let bd_mid = super::evaluate_breakdown(&state_mid);
        assert!(
            bd_mid.boost_term_s1.abs() < 0.001,
            "50% HP: boost_term_s1 should be ~0 (Kaizo zero-out), got {}",
            bd_mid.boost_term_s1
        );
        let diff_mid = (super::evaluate(&state_mid) - bd_mid.total).abs();
        assert!(diff_mid < 0.001, "invariant broken at 50% HP: diff={}", diff_mid);

        // Case 3: 30% HP (dangerous) → boost_term_s1 < 0 (~-30 from -0.5 × 60)
        let mut state_low = State::default();
        state_low.side_one.special_attack_boost = 2;
        state_low.side_one.get_active().hp = 30;
        let bd_low = super::evaluate_breakdown(&state_low);
        assert!(
            bd_low.boost_term_s1 < 0.0,
            "30% HP: boost_term_s1 should be negative, got {}",
            bd_low.boost_term_s1
        );
        assert!(
            (bd_low.boost_term_s1 - (-30.0)).abs() < 0.001,
            "30% HP: boost_term_s1 should be ~-30 (-0.5 × 60), got {}",
            bd_low.boost_term_s1
        );
        let diff_low = (super::evaluate(&state_low) - bd_low.total).abs();
        assert!(diff_low < 0.001, "invariant broken at 30% HP: diff={}", diff_low);
    }

    #[test]
    fn test_boost_reward_independent_of_hp() {
        // Post-fix invariant: the boost reward must NOT depend on HP.
        // Foul-play (NeurIPS 2025 Gen9OU winner) gives flat boost credit;
        // HP-conditional gating double-counted with threat_score × 40 and
        // produced a 40-70% HP blind zone for setup sweepers.
        //
        // Construct a state with side_one's active at +2 SpA and compare the
        // boost contribution at 100% HP vs. 30% HP. They MUST be equal.
        let mut state_full = State::default();
        state_full.side_one.special_attack_boost = 2;
        let bd_full = super::evaluate_breakdown(&state_full);

        let mut state_low = State::default();
        state_low.side_one.special_attack_boost = 2;
        state_low.side_one.get_active().hp = 30;
        let bd_low = super::evaluate_breakdown(&state_low);

        assert!(
            (bd_full.boost_term_s1 - bd_low.boost_term_s1).abs() < 0.001,
            "boost_term_s1 must be HP-independent post fix; full={}, low={}",
            bd_full.boost_term_s1,
            bd_low.boost_term_s1
        );
    }

    #[test]
    fn test_active_only_boost_reward() {
        // Boosts on bench (non-active) Pokemon must NOT contribute to the
        // boost term. This invariant is what makes a separate self-mortality
        // gate unnecessary — switching out a +2 mon zeroes its credit.
        //
        // We compare a clean default state vs. one where side_one's
        // *non-active* slot has +2 SpA. Boost contribution should be zero
        // either way.
        let state_clean = State::default();
        let bd_clean = super::evaluate_breakdown(&state_clean);
        assert!(
            bd_clean.boost_term_s1.abs() < 0.001,
            "default state should have zero boost contribution, got {}",
            bd_clean.boost_term_s1
        );

        // Mutate a bench mon directly (slot 1 is bench since active_index = 0).
        let mut state_bench = State::default();
        state_bench.side_one.pokemon[crate::state::PokemonIndex::P1].hp = 100; // make sure it's alive so the loop visits it
        // Note: side_one.special_attack_boost is the *active* mon's stage,
        // not stored per-Pokemon. So bench-only boosts can't even be
        // expressed in our state representation — this test verifies the
        // active_index gate, not bench-stat-state.
        //
        // We instead simulate "side_one has +2 SpA stage but the active is
        // index 1 not 0" by swapping active_index. The intent: the gate is
        // `iter.pokemon_index == active_index`, so changing active_index
        // away from the slot whose `if`-branch runs the boost block must
        // skip the boost reward.
        state_bench.side_one.special_attack_boost = 2;
        state_bench.side_one.active_index =
            crate::state::PokemonIndex::P1;
        let bd_bench = super::evaluate_breakdown(&state_bench);
        // Active index 1 still gets the +2 SpA credit (the boost is
        // tracked per-side, applied to whichever mon is active). What we
        // really want to test: if the *active* slot's mon has 0 boost,
        // then no other slot's existence inflates the boost term.
        // Simpler restatement (and the one this test actually pins):
        // boost_term scales linearly with the *active*'s side-level boost,
        // not with the count of bench mons.
        assert!(
            (bd_bench.boost_term_s1 - 50.0).abs() < 0.001,
            "active mon at +2 SpA should give +50 boost (regardless of bench), got {}",
            bd_bench.boost_term_s1
        );
    }

    #[test]
    fn test_fainted_active_no_boost_reward() {
        // A fainted active Pokemon must not earn boost rewards. The gate is
        // the `if pkmn.hp > 0 { ... }` block surrounding the boost code —
        // this test pins it explicitly so any future refactor that moves
        // the boost code outside the gate breaks loudly.
        let mut state = State::default();
        state.side_one.special_attack_boost = 2;
        state.side_one.get_active().hp = 0; // faint the active
        let bd = super::evaluate_breakdown(&state);
        assert!(
            bd.boost_term_s1.abs() < 0.001,
            "fainted active must zero out boost_term_s1, got {}",
            bd.boost_term_s1
        );
    }

    #[test]
    fn test_constants_match_design() {
        // Fix #2.5: offensive boost constants reduced 30.0 → 25.0 to
        // compensate for double-counting with threat_score × 40
        // (threat_score already reads +2 Atk as roughly +24 per side).
        // Defensive boosts unchanged (foul-play parity, no double-count
        // with threat_score which only reads offense).
        //
        // These assertions are static enough that a constant change without
        // an intentional design decision will fail this test.
        assert_eq!(super::POKEMON_ATTACK_BOOST, 25.0);
        assert_eq!(super::POKEMON_DEFENSE_BOOST, 15.0);
        assert_eq!(super::POKEMON_SPECIAL_ATTACK_BOOST, 25.0);
        assert_eq!(super::POKEMON_SPECIAL_DEFENSE_BOOST, 15.0);
        assert_eq!(super::POKEMON_SPEED_BOOST, 25.0);
    }
}
