//! Plan I, Fix #1.6: death-aware switch priority in `compute()`.
//!
//! Red-phase failing tests. The current `compute()` always allocates
//! `mass_dmg` (default 0.6) to the damage move and `mass_switch` (default
//! 0.3) to the switch. When the active Pokemon is in a "definitely dies
//! this turn" state — opp predicted damage > my HP AND opp moves first
//! AND I have no priority move that fires — we want compute() to shift
//! mass aggressively toward the switch slot, so the heuristic prior puts
//! Q-value pressure on switching out.
//!
//! Real-world impact (battle 2598375717 vs waldenjames, T4):
//! Dragonite 304/324 HP vs Pelipper 60% HP. Ice-Beam SE 4× OHKOs
//! Adamant Dragonite. Pelipper Modest 252 Spe outspeeds. Engine top1
//! = HEATRAN switch (Q=0.53) vs stay-in attack (Q=0.5x). User clicked
//! attack and lost. Heavier prior on switch → decisive Q gap → user
//! switches.
//!
//! See docs/superpowers/plans/2026-04-30-death-aware-switch-priority.md.

use poke_engine::choices::Choices;
use poke_engine::heuristic_prior::compute;
use poke_engine::nn_state_encoder::SidePerspective;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonType, State};

/// Test 1 — definitely dying: shift mass to switch.
///
/// Dragonite 60/324 HP vs Pelipper at full HP. Pelipper outspeeds
/// (252 Spe Modest > Adamant Dragonite). Pelipper's Ice Beam max-roll
/// hits 4× SE on Dragon/Flying and OHKOs from this HP. Dragonite has
/// only non-priority moves (Earthquake, Ice Spinner). Bench has live
/// Heatran (Fire/Steel) which resists Ice and is faster than Pelipper.
///
/// Note: Fix #1.5 already filters out non-firing moves in
/// `damage_calc_top_move`, so when definitely dying with no priority,
/// `dmg_pick == None`. compute() falls into its uniform-fill branch:
/// places 0.3 on the switch slot and spreads the remaining 0.7
/// uniformly across all OTHER legal slots — INCLUDING the dying move
/// slots. That's the leak.
///
/// Fix #1.6 must detect "definitely dying with viable switch" and
/// concentrate mass on the switch (>0.5), starving the dying move
/// slots (each <0.15 — they're still legal options but should not
/// receive the inflated uniform share).
#[test]
fn heuristic_shifts_mass_to_switch_when_definitely_dying() {
    let mut state = State::default();

    // Mariga's Dragonite — low HP, outspeeded, no priority moves.
    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 60;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().defense = 230;
    state.side_one.get_active().special_defense = 230;
    state.side_one.get_active().speed = 240;
    state.side_one.active_index = PokemonIndex::P0;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::ICESPINNER);

    // Bench: Heatran at full HP. Fire/Steel — 0.5× to Ice from Pelipper.
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_one.pokemon[PokemonIndex::P1].level = 100;
    state.side_one.pokemon[PokemonIndex::P1].hp = 385;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P1].defense = 248;
    state.side_one.pokemon[PokemonIndex::P1].special_defense = 384;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHPOWER);
    for idx in [
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[idx].hp = 0;
    }

    // Opp Pelipper (Water/Flying) — Modest 252 Spe outspeeds Adamant
    // Dragonite, Ice Beam SE 4× OHKOs.
    state.side_two.get_active().id = PokemonName::PELIPPER;
    state.side_two.get_active().types = (PokemonType::WATER, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 280;
    state.side_two.get_active().maxhp = 280;
    state.side_two.get_active().special_attack = 280;
    state.side_two.get_active().speed = 280; // > Dragonite's 240
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICEBEAM);

    let (s1_options, _) = state.root_get_all_options();
    let result = compute(&state, SidePerspective::Side1, &s1_options, 0.6, 0.3)
        .expect("heuristic should succeed (switch pick resolves)");

    // Switch pick is Heatran. dmg_pick may be None under Fix #1.5
    // (no moves fire); that's expected.
    assert_eq!(
        result.matchup_switch_pick,
        Some(PokemonName::HEATRAN),
        "switch pick should be Heatran"
    );

    // Total mass still sums to 1.0.
    let total: f32 = result.probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-4, "mass must sum to 1.0; got {}", total);

    // Switch mass occupies one slot in 4..9. Dying-move leak appears in
    // 0..4 (the dying move slots are still in `options` and currently
    // receive uniform fill mass). The fix must concentrate on switch.
    let switch_mass: f32 = result.probs[4..9].iter().cloned().fold(0.0, f32::max);
    let dying_move_mass: f32 = result.probs[0..4].iter().cloned().fold(0.0, f32::max);

    assert!(
        switch_mass > 0.5,
        "definitely-dying state must put >0.5 mass on switch; got switch={}, dying_move={}, probs={:?}",
        switch_mass,
        dying_move_mass,
        result.probs
    );
    assert!(
        dying_move_mass < 0.15,
        "definitely-dying state must put <0.15 mass on each dying move slot; got switch={}, dying_move={}, probs={:?}",
        switch_mass,
        dying_move_mass,
        result.probs
    );
    assert!(
        switch_mass > dying_move_mass,
        "switch mass must dominate dying-move mass; got switch={}, dying_move={}",
        switch_mass,
        dying_move_mass
    );
}

/// Test 2 — I survive opp's predicted attack: keep normal 0.6/0.3 split.
///
/// Same Dragonite vs Pelipper fixture, but Dragonite is at FULL HP (324)
/// — survives Pelipper's Ice Beam (just barely; Multiscale would help in
/// reality but we don't model it here). With i_survive==true, the
/// "definitely dying" branch must NOT trigger; the heuristic returns the
/// standard 0.6/0.3 split.
#[test]
fn heuristic_keeps_normal_split_when_i_survive() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 324;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().defense = 280; // bulkier
    state.side_one.get_active().special_defense = 280; // bulkier
    state.side_one.get_active().speed = 240;
    state.side_one.active_index = PokemonIndex::P0;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::ICESPINNER);

    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_one.pokemon[PokemonIndex::P1].level = 100;
    state.side_one.pokemon[PokemonIndex::P1].hp = 385;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P1].defense = 248;
    state.side_one.pokemon[PokemonIndex::P1].special_defense = 384;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHPOWER);
    for idx in [
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[idx].hp = 0;
    }

    state.side_two.get_active().id = PokemonName::PELIPPER;
    state.side_two.get_active().types = (PokemonType::WATER, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 280;
    state.side_two.get_active().maxhp = 280;
    state.side_two.get_active().special_attack = 200; // weaker than test 1
    state.side_two.get_active().speed = 280;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICEBEAM);

    let (s1_options, _) = state.root_get_all_options();
    let result = compute(&state, SidePerspective::Side1, &s1_options, 0.6, 0.3)
        .expect("heuristic should succeed");

    let dmg_mass: f32 = result.probs[0..4].iter().cloned().fold(0.0, f32::max);
    let switch_mass: f32 = result.probs[4..9].iter().cloned().fold(0.0, f32::max);

    // Normal split preserved.
    assert!(
        (dmg_mass - 0.6).abs() < 1e-3,
        "surviving state must keep 0.6 mass on dmg; got dmg={}",
        dmg_mass
    );
    assert!(
        (switch_mass - 0.3).abs() < 1e-3,
        "surviving state must keep 0.3 mass on switch; got switch={}",
        switch_mass
    );
}

/// Test 3 — I'd die but have priority: keep normal split.
///
/// Dragonite 60/324 HP vs Pelipper that outspeeds and OHKOs. BUT
/// Dragonite has Extreme Speed (priority +2). With my priority > opp
/// priority (Pelipper's Ice Beam is priority 0), Extreme Speed fires
/// first regardless of speed/HP. So "definitely dying" is false — I get
/// at least one hit in. compute() must keep the normal 0.6/0.3 split.
#[test]
fn heuristic_keeps_normal_split_when_i_have_priority() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 60;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().defense = 230;
    state.side_one.get_active().special_defense = 230;
    state.side_one.get_active().speed = 240;
    state.side_one.active_index = PokemonIndex::P0;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EXTREMESPEED);

    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_one.pokemon[PokemonIndex::P1].level = 100;
    state.side_one.pokemon[PokemonIndex::P1].hp = 385;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P1].defense = 248;
    state.side_one.pokemon[PokemonIndex::P1].special_defense = 384;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHPOWER);
    for idx in [
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[idx].hp = 0;
    }

    state.side_two.get_active().id = PokemonName::PELIPPER;
    state.side_two.get_active().types = (PokemonType::WATER, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 280;
    state.side_two.get_active().maxhp = 280;
    state.side_two.get_active().special_attack = 280;
    state.side_two.get_active().speed = 280;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICEBEAM);

    let (s1_options, _) = state.root_get_all_options();
    let result = compute(&state, SidePerspective::Side1, &s1_options, 0.6, 0.3)
        .expect("heuristic should succeed");

    let dmg_mass: f32 = result.probs[0..4].iter().cloned().fold(0.0, f32::max);
    let switch_mass: f32 = result.probs[4..9].iter().cloned().fold(0.0, f32::max);

    // Priority lets us fire — normal split.
    assert!(
        (dmg_mass - 0.6).abs() < 1e-3,
        "priority-mover state must keep 0.6 mass on dmg; got dmg={}",
        dmg_mass
    );
    assert!(
        (switch_mass - 0.3).abs() < 1e-3,
        "priority-mover state must keep 0.3 mass on switch; got switch={}",
        switch_mass
    );
}

/// Test 4 — I'd die AND no viable switch: fall through to None.
///
/// Dragonite 60/324 HP vs Pelipper outspeeding-OHKO. No priority move.
/// Bench is fully fainted — nothing to switch to. After Fix #1.5,
/// `damage_calc_top_move` already returns None because no moves fire
/// (we're slower and dying). Combined with no switch available, BOTH
/// heuristics skip → compute() returns None. The dying-penalty branch
/// must not crash and must not synthesize a non-None result; the caller
/// falls back to the raw NN policy.
#[test]
fn heuristic_falls_through_when_no_viable_switch() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 60;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().defense = 230;
    state.side_one.get_active().special_defense = 230;
    state.side_one.get_active().speed = 240;
    state.side_one.active_index = PokemonIndex::P0;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::ICESPINNER);

    // Faint ALL bench — no switch available.
    for idx in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[idx].hp = 0;
    }

    state.side_two.get_active().id = PokemonName::PELIPPER;
    state.side_two.get_active().types = (PokemonType::WATER, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 280;
    state.side_two.get_active().maxhp = 280;
    state.side_two.get_active().special_attack = 280;
    state.side_two.get_active().speed = 280;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICEBEAM);

    let (s1_options, _) = state.root_get_all_options();
    let result = compute(&state, SidePerspective::Side1, &s1_options, 0.6, 0.3);

    // dmg_pick is None (Fix #1.5 priority filter rejects all moves) and
    // switch_pick is None (no live bench). Both heuristics skip — caller
    // falls back to raw NN priors. This must remain true post-Fix #1.6;
    // the dying-penalty branch must not synthesize a non-None result.
    assert!(
        result.is_none(),
        "compute() must return None when both heuristics skip; got {:?}",
        result
    );
}
