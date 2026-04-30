//! Plan I, Fix #1.5: priority-aware damage_calc_top_move.
//!
//! Red-phase failing tests. The current `damage_calc_top_move` picks
//! the highest raw-damage move without checking whether the move
//! actually fires before the opp KOs us. These tests pin the desired
//! priority/speed/survival behavior surfaced by battle 2598333175 vs
//! Darkssz, T9 (Dragonite mirror, opp at +1 spe via Dragon Dance).
//!
//! See docs/superpowers/plans/2026-04-30-priority-aware-damage.md.

use poke_engine::choices::Choices;
use poke_engine::heuristic_prior::damage_calc_top_move;
use poke_engine::nn_state_encoder::SidePerspective;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonMoveIndex, PokemonType, State};

/// Test 1 — priority move chosen when slower attacker would die first.
///
/// Mariga's Dragonite at low HP (104/324) vs opp Dragonite that just
/// Dragon Danced (+1 atk +1 spe). Mariga's Dragonite is slower (no
/// Dance), opp's Dragon Claw OHKOs at 32% HP. Ice Spinner (4× SE on
/// Dragon/Flying) does max raw damage but never fires; Extreme Speed
/// (priority +2) connects first.
///
/// Current behavior: returns ICESPINNER (highest max-roll damage).
/// Desired behavior: returns EXTREMESPEED (only move that fires).
#[test]
fn priority_move_chosen_when_slower_attacker_would_die() {
    let mut state = State::default();

    // Mariga's Dragonite — low HP, slower than opp's +1 spe Dragonite.
    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 104;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().defense = 230;
    state.side_one.get_active().special_defense = 230;
    state.side_one.get_active().speed = 240; // base Dragonite ~ 80 base, ~240 stat
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICESPINNER);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EXTREMESPEED);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M3, Choices::DRAGONCLAW);

    // Opp Dragonite at +1 atk +1 spe (post-Dragon Dance). Set its
    // boost-applied speed via side_two.speed_boost = 1.
    state.side_two.get_active().id = PokemonName::DRAGONITE;
    state.side_two.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 324;
    state.side_two.get_active().maxhp = 324;
    state.side_two.get_active().attack = 350;
    state.side_two.get_active().defense = 230;
    state.side_two.get_active().special_defense = 230;
    state.side_two.get_active().speed = 240;
    state.side_two.attack_boost = 1;
    state.side_two.speed_boost = 1;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::DRAGONCLAW);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EXTREMESPEED);

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(Choices::EXTREMESPEED),
        "slower low-HP attacker dying to opp's predicted move must pick priority move that actually fires"
    );
}

/// Test 2 — highest damage chosen when attacker outspeeds (regression).
///
/// Same Dragonite mirror, but Mariga's Dragonite is at +1 spe (it
/// Danced) and opp is unboosted. Mariga outspeeds; Ice Spinner fires
/// before opp moves regardless of HP. Expect ICESPINNER.
#[test]
fn priority_aware_picks_highest_damage_when_attacker_outspeeds() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 100;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().speed = 240;
    state.side_one.speed_boost = 1; // Mariga used Dragon Dance
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICESPINNER);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EXTREMESPEED);

    state.side_two.get_active().id = PokemonName::DRAGONITE;
    state.side_two.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 324;
    state.side_two.get_active().maxhp = 324;
    state.side_two.get_active().attack = 350;
    state.side_two.get_active().defense = 230;
    state.side_two.get_active().speed = 240;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::DRAGONCLAW);

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(Choices::ICESPINNER),
        "attacker outspeeds opp → Ice Spinner fires first, max raw damage wins"
    );
}

/// Test 3 — highest damage chosen when slower attacker survives opp.
///
/// Bulky Dragonite at full HP, opp Dragonite at +1 spe but no boosts on
/// attack — Mariga is slower but survives Dragon Claw. Ice Spinner
/// fires AFTER opp's move (same turn, slower) but it DOES fire.
/// Expect ICESPINNER.
#[test]
fn priority_aware_picks_highest_damage_when_slower_but_survives() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.get_active().level = 100;
    state.side_one.get_active().hp = 324;
    state.side_one.get_active().maxhp = 324;
    state.side_one.get_active().attack = 350;
    state.side_one.get_active().defense = 280;
    state.side_one.get_active().special_defense = 280;
    state.side_one.get_active().speed = 240;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::ICESPINNER);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EXTREMESPEED);

    // Opp at +1 spe (outspeeds) but base attack stays — Dragon Claw
    // chips ~25-30%, never close to OHKOing Mariga at full HP.
    state.side_two.get_active().id = PokemonName::DRAGONITE;
    state.side_two.get_active().types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().hp = 324;
    state.side_two.get_active().maxhp = 324;
    state.side_two.get_active().attack = 350;
    state.side_two.get_active().defense = 230;
    state.side_two.get_active().speed = 240;
    state.side_two.speed_boost = 1;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::DRAGONCLAW);

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(Choices::ICESPINNER),
        "slower attacker that survives opp's hit still gets to fire its top-damage move"
    );
}

/// Test 4 — returns None when all damaging moves are 0 effective damage.
///
/// Mariga has only Earthquake; opp is Zapdos (Electric/Flying) — Ground
/// is 0× into Flying. No priority moves available. Should return None
/// because the only damaging move does 0 damage (immunity), regardless
/// of priority logic.
#[test]
fn priority_aware_returns_none_when_only_damaging_move_is_immune() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::GARCHOMP;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::GROUND);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::SWORDSDANCE);

    // Zapdos — Electric/Flying, Ground does 0× into Flying.
    state.side_two.get_active().id = PokemonName::ZAPDOS;
    state.side_two.get_active().types = (PokemonType::ELECTRIC, PokemonType::FLYING);

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(
        pick, None,
        "no damaging move connects (Ground 0× into Flying) → None"
    );
}

/// Test 5 — regression guard. Garchomp vs Heatran, EQ super-effective.
/// Garchomp outspeeds Heatran by default speed (102 vs 77 base, ~239
/// vs ~166 stat). EQ fires, both moves are non-priority, attacker
/// outspeeds → existing behavior preserved.
#[test]
fn priority_aware_preserves_garchomp_vs_heatran_eq_pick() {
    let mut state = State::default();
    state.side_one.get_active().id = PokemonName::GARCHOMP;
    state.side_one.get_active().types = (PokemonType::DRAGON, PokemonType::GROUND);
    state.side_one.get_active().attack = 359;
    state.side_one.get_active().speed = 239;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::DRAGONCLAW);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::STONEEDGE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M3, Choices::SWORDSDANCE);

    state.side_two.get_active().id = PokemonName::HEATRAN;
    state.side_two.get_active().types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_two.get_active().hp = 385;
    state.side_two.get_active().maxhp = 385;
    state.side_two.get_active().defense = 248;
    state.side_two.get_active().special_defense = 384;
    state.side_two.get_active().speed = 166;

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(Choices::EARTHQUAKE),
        "Garchomp outspeeds Heatran, EQ is 2× SE on Fire/Steel → must still pick EQ"
    );
}
