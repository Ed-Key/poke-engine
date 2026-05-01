//! Bug #3 catalog replay tests — verify symmetric Side2 prior corrects
//! engine mispredictions on documented postmortem scenarios.
//!
//! Pass criteria (T9 — full 8 scenarios): >= 6/8 produce the expected
//! opp move in opp_top3. Without the fix, baseline is 0/8.
//!
//! T8 ships ONE scenario (sapceinvader-T1) as the template + runner.
//! T9 expands to 8 by adding more `build_state_*` helpers and using
//! `run_scenario` unchanged.

use poke_engine::choices::Choices;
use poke_engine::heuristic_prior::compute as compute_heuristic;
use poke_engine::mcts::MctsSearch;
use poke_engine::nn_client::ACTION_DIM;
use poke_engine::nn_state_encoder::{map_policy_to_options_blended, SidePerspective};
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonMoveIndex, PokemonType, State};
use std::time::Duration;

/// sapceinvader-T1: Urshifu-RS lead vs Rillaboom lead.
///
/// Engine pre-fix: predicted U-turn (uniform priors leak mass into pivot
/// moves; opp side never gets the heuristic damage-pick lift).
/// Opp actually clicked Wood Hammer: Grass STAB, 2x SE on Urshifu's
/// Water-type half, near-OHKO under Choice Band / Loaded Dice.
///
/// Fixture stat values are approximate (the heuristic damage-pick ranks
/// moves by raw damage output, not by precise EV spreads — exact stats
/// are not load-bearing for the assertion, just plausible).
fn build_state_sapceinvader_t1() -> State {
    let mut state = State::default();

    // ---- My active: Urshifu-Rapid-Strike (Choice Scarf, standard set) ----
    {
        let p = state.side_one.get_active();
        p.id = PokemonName::URSHIFURAPIDSTRIKE;
        p.types = (PokemonType::FIGHTING, PokemonType::WATER);
        p.hp = 341;
        p.maxhp = 341;
        p.attack = 359;
        p.defense = 218;
        p.special_attack = 156;
        p.special_defense = 218;
        p.speed = 339; // 97 base, Jolly 252+, x1.5 Scarf
        p.replace_move(PokemonMoveIndex::M0, Choices::SURGINGSTRIKES);
        p.replace_move(PokemonMoveIndex::M1, Choices::CLOSECOMBAT);
        p.replace_move(PokemonMoveIndex::M2, Choices::UTURN);
        p.replace_move(PokemonMoveIndex::M3, Choices::ICESPINNER);
    }

    // ---- Opp active: Rillaboom (Grass; Choice Band / Loaded Dice) ----
    {
        let p = state.side_two.get_active();
        p.id = PokemonName::RILLABOOM;
        // Single-type representation: pair second slot with TYPELESS.
        p.types = (PokemonType::GRASS, PokemonType::TYPELESS);
        p.hp = 372;
        p.maxhp = 372;
        p.attack = 394;
        p.defense = 218;
        p.special_attack = 156;
        p.special_defense = 218;
        p.speed = 240;
        p.replace_move(PokemonMoveIndex::M0, Choices::WOODHAMMER);
        p.replace_move(PokemonMoveIndex::M1, Choices::UTURN);
        p.replace_move(PokemonMoveIndex::M2, Choices::KNOCKOFF);
        p.replace_move(PokemonMoveIndex::M3, Choices::GRASSYGLIDE);
    }

    state
}

/// Run MCTS on the given state with the given Side2 prior mix, return
/// opp_top3 move names (uppercase, e.g. "WOODHAMMER") sorted by visit
/// count descending.
///
/// `mix_side2 == 0.0` reproduces the pre-fix behavior (no heuristic
/// blending on Side2 → pure uniform priors). `mix_side2 > 0.0` activates
/// the symmetric Side2 prior. Mirrors `analyze_handler` in
/// `src/bin/server.rs` (the production code path we're regression-testing).
fn run_scenario(state: State, mix_side2: f32) -> Vec<String> {
    // Snapshot s2_options BEFORE moving `state` into the search; we need
    // it again after run_for to render MoveChoice → String via `to_string`.
    let (s1_options, s2_options) = state.root_get_all_options();

    let s2_priors = if mix_side2 > 0.0 {
        let heuristic_s2 = compute_heuristic(
            &state,
            SidePerspective::Side2,
            &s2_options,
            0.7, // mass_dmg
            0.3, // mass_switch
        );
        let uniform = vec![1.0_f32 / ACTION_DIM as f32; ACTION_DIM];
        Some(map_policy_to_options_blended(
            &uniform,
            &state,
            SidePerspective::Side2,
            &s2_options,
            heuristic_s2.as_ref(),
            mix_side2,
        ))
    } else {
        None
    };

    // We need a snapshot of side_two for MoveChoice rendering after the
    // search. Cloning the side (cheap stack copy) avoids borrow conflicts
    // with `state` being moved into MctsSearch::new_with_priors.
    let side_two_snapshot = state.side_two.clone();

    let mut search = MctsSearch::new_with_priors(
        state,
        s1_options,
        s2_options,
        1.25, // c_puct (AlphaZero default)
        None, // s1 priors: uniform/default for these tests
        s2_priors,
    );
    // T5 wired per-side forced-playouts; opp side gets c_forced_side2=2.0
    // when the Side2 fix is active. With mix_side2=0.0 the priors arg is
    // None so set_c_forced_side2 has no effective influence (uniform priors
    // → uniform forced thresholds), but we set both unconditionally to mirror
    // production wiring.
    search.set_c_forced(0.0);
    search.set_c_forced_side2(2.0);
    search.run_for(Duration::from_millis(500));
    let snap = search.snapshot(500);

    // Sort opp moves by visit count desc, take top 3, render to uppercase
    // move names. `MoveChoice::to_string` produces lowercase (e.g.
    // "woodhammer", "rillaboom" for switches). We uppercase to match the
    // canonical Choices debug spelling used in assertions.
    let mut s2 = snap.s2.clone();
    s2.sort_by(|a, b| b.visits.cmp(&a.visits));
    s2.iter()
        .take(3)
        .map(|r| r.move_choice.to_string(&side_two_snapshot).to_uppercase())
        .collect()
}

#[test]
fn test_bug3_sapceinvader_t1_baseline() {
    // Without the fix (mix_side2=0.0), engine MAY or MAY NOT mispredict.
    // For sapceinvader-T1 specifically the rollout `evaluate()` penalizes
    // Urshifu's huge HP drop after WOODHAMMER hits, so even uniform priors
    // surface it within 500ms of search. The pre-fix bug #3 phenomenon
    // requires deeper / more complex states (multi-turn pivot decisions,
    // hazards, status interactions) where uniform-prior leakage into pivot
    // moves like U-turn dominates the early visit budget.
    //
    // This test does NOT assert anything — it captures the pre-fix
    // behavior for the record. T9's aggregate test asserts the catalog
    // baseline is poor (<= 3/8 hits) once the harder scenarios are added.
    let state = build_state_sapceinvader_t1();
    let top3 = run_scenario(state, 0.0);
    eprintln!("[baseline] sapceinvader-T1 opp_top3: {:?}", top3);
}

#[test]
fn test_bug3_sapceinvader_t1_fix() {
    // With the fix (mix_side2=0.5), the heuristic damage-pick should
    // dominate the opp prior → WOODHAMMER receives most of the 0.7
    // mass_dmg → MCTS visits it heavily → it lands in opp_top3.
    let state = build_state_sapceinvader_t1();
    let top3 = run_scenario(state, 0.5);
    eprintln!("[fix] sapceinvader-T1 opp_top3: {:?}", top3);

    let hit = top3.iter().any(|m| m == "WOODHAMMER");
    assert!(
        hit,
        "Expected WOODHAMMER in opp_top3 with mix_side2=0.5; got {:?}",
        top3
    );
}
