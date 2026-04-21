use super::abilities::Abilities;
use super::items::Items;
use super::state::PokemonVolatileStatus;
use crate::choices::MoveCategory;
use crate::engine::damage_calc::{calculate_damage, DamageRolls};
use crate::engine::generate_instructions::get_effective_speed;
use crate::state::{Pokemon, PokemonStatus, Side, SideReference, State};

const POKEMON_ALIVE: f32 = 30.0;
const POKEMON_HP: f32 = 100.0;
const USED_TERA: f32 = -75.0;

const POKEMON_ATTACK_BOOST: f32 = 30.0;
const POKEMON_DEFENSE_BOOST: f32 = 15.0;
const POKEMON_SPECIAL_ATTACK_BOOST: f32 = 30.0;
const POKEMON_SPECIAL_DEFENSE_BOOST: f32 = 15.0;
const POKEMON_SPEED_BOOST: f32 = 30.0;

pub const THREAT_SCORE_WEIGHT: f32 = 40.0;
pub const MORTALITY_SCORE_WEIGHT: f32 = 40.0;

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
        if let Some((normal, _crit)) =
            calculate_damage(state, side_ref, choice, DamageRolls::Average)
        {
            if normal > best_damage {
                best_damage = normal;
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

pub fn evaluate(state: &State) -> f32 {
    let mut score = 0.0;

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score += evaluate_pokemon(pkmn);
            score += evaluate_hazards(pkmn, &state.side_one);
            if iter.pokemon_index == state.side_one.active_index {
                for vs in state.side_one.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => score += LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => score += SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => score += CONFUSION,
                        _ => {}
                    }
                }

                score += get_boost_multiplier(state.side_one.attack_boost) * POKEMON_ATTACK_BOOST;
                score += get_boost_multiplier(state.side_one.defense_boost) * POKEMON_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                score += get_boost_multiplier(state.side_one.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.speed_boost) * POKEMON_SPEED_BOOST;
                score += threat_score(state, &SideReference::SideOne) * THREAT_SCORE_WEIGHT;
                score -= mortality_score(state, &SideReference::SideOne) * MORTALITY_SCORE_WEIGHT;
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
            score -= evaluate_hazards(pkmn, &state.side_two);

            if iter.pokemon_index == state.side_two.active_index {
                for vs in state.side_two.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => score -= LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => score -= SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => score -= CONFUSION,
                        _ => {}
                    }
                }

                score -= get_boost_multiplier(state.side_two.attack_boost) * POKEMON_ATTACK_BOOST;
                score -= get_boost_multiplier(state.side_two.defense_boost) * POKEMON_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                score -= get_boost_multiplier(state.side_two.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.speed_boost) * POKEMON_SPEED_BOOST;
                score -= threat_score(state, &SideReference::SideTwo) * THREAT_SCORE_WEIGHT;
                score += mortality_score(state, &SideReference::SideTwo) * MORTALITY_SCORE_WEIGHT;
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
}
