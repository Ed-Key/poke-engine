//! TDD red-phase tests for Aegislash Stance Change.
//!
//! Showdown behavior:
//! - Aegislash starts in Shield form (60/50/150/50/150/60).
//! - Using any DAMAGING move triggers a forme change to AEGISLASHBLADE
//!   (60/150/50/150/50/60) BEFORE damage is calculated, so Blade-form
//!   atk/spa is used in the damage calc.
//! - Using King's Shield (a Status move) reverts AEGISLASHBLADE back to
//!   AEGISLASH (Shield form).
//! - No other status move triggers a swap.
//!
//! These tests build on the harness style in tests/test_battle_mechanics.rs
//! (set_moves_on_pkmn_and_call_generate_instructions). They are expected
//! to FAIL on db57fa6 because src/genx/abilities.rs has no STANCECHANGE
//! handler and src/genx/base_stats.rs has no AEGISLASH/AEGISLASHBLADE
//! base-stat entries.

#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::choices::{Choices, MoveCategory};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::instruction::{FormeChangeInstruction, Instruction, StateInstructions};
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonType, SideReference, State};

fn set_moves_and_run(
    state: &mut State,
    side_one_move: Choices,
    side_two_move: Choices,
) -> Vec<StateInstructions> {
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, side_one_move);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, side_two_move);

    generate_instructions_from_move_pair(
        state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::Move(PokemonMoveIndex::M0),
        false,
    )
}

/// Locate the FormeChange instruction (if any) in any branch.
fn first_forme_change(branches: &[StateInstructions]) -> Option<&FormeChangeInstruction> {
    for branch in branches {
        for ins in &branch.instruction_list {
            if let Instruction::FormeChange(fc) = ins {
                return Some(fc);
            }
        }
    }
    None
}

/// Sum first-branch damage dealt to a side.
fn first_branch_damage_to(branches: &[StateInstructions], side: SideReference) -> i16 {
    let mut total = 0;
    if let Some(branch) = branches.first() {
        for ins in &branch.instruction_list {
            if let Instruction::Damage(d) = ins {
                if d.side_ref == side {
                    total += d.damage_amount;
                }
            }
        }
    }
    total
}

// --------------------------------------------------------------------------
// Test 1 — damaging move (Shadow Ball) flips Shield → Blade.
// --------------------------------------------------------------------------
#[test]
fn stance_change_swaps_to_blade_on_damaging_move() {
    let mut state = State::default();
    let attacker = state.side_one.get_active();
    attacker.id = PokemonName::AEGISLASH;
    attacker.ability = Abilities::STANCECHANGE;

    let branches = set_moves_and_run(&mut state, Choices::SHADOWBALL, Choices::SPLASH);

    // Whichever branch is taken (crit or not), the FormeChange must fire,
    // and it must convert AEGISLASH → AEGISLASHBLADE.
    let fc = first_forme_change(&branches).expect(
        "expected a FormeChange instruction when AEGISLASH uses a damaging move under Stance Change",
    );
    assert_eq!(fc.side_ref, SideReference::SideOne);
    assert_eq!(
        fc.name_change,
        PokemonName::AEGISLASHBLADE as i16 - PokemonName::AEGISLASH as i16,
        "name_change delta should swap AEGISLASH -> AEGISLASHBLADE",
    );
}

// --------------------------------------------------------------------------
// Test 2 — King's Shield flips Blade → Shield.
// --------------------------------------------------------------------------
#[test]
fn stance_change_swaps_back_to_shield_on_kings_shield() {
    let mut state = State::default();
    let attacker = state.side_one.get_active();
    attacker.id = PokemonName::AEGISLASHBLADE;
    attacker.ability = Abilities::STANCECHANGE;

    let branches = set_moves_and_run(&mut state, Choices::KINGSSHIELD, Choices::SPLASH);

    let fc = first_forme_change(&branches)
        .expect("expected a FormeChange instruction when AEGISLASHBLADE uses King's Shield");
    assert_eq!(fc.side_ref, SideReference::SideOne);
    assert_eq!(
        fc.name_change,
        PokemonName::AEGISLASH as i16 - PokemonName::AEGISLASHBLADE as i16,
        "name_change delta should swap AEGISLASHBLADE -> AEGISLASH",
    );
}

// --------------------------------------------------------------------------
// Test 3 — Damage calc uses Blade-form Special Attack.
//
// Compare:
//   A) Aegislash with Stance Change uses Shadow Ball   (Special)
//   B) Aegislash WITHOUT Stance Change uses Shadow Ball (Special)
//
// Case A should swap to Blade (spa 50 -> 150) before damage is rolled, so
// the damage dealt should be strictly greater than case B.
// --------------------------------------------------------------------------
#[test]
fn stance_change_uses_blade_spa_for_damage_calc() {
    // Helper: build a state with Aegislash on side_one and a non-Normal,
    // non-Dark target on side_two so Ghost-type Shadow Ball lands.
    fn build(stance_change: bool) -> State {
        let mut state = State::default();
        let pkmn = state.side_one.get_active();
        pkmn.id = PokemonName::AEGISLASH;
        pkmn.ability = if stance_change {
            Abilities::STANCECHANGE
        } else {
            Abilities::NONE
        };
        // Make the target a generic Psychic-type so Shadow Ball is
        // super-effective and we get a clear damage signal.
        let target = state.side_two.get_active();
        target.types = (PokemonType::PSYCHIC, PokemonType::TYPELESS);
        target.hp = 500;
        target.maxhp = 500;
        state
    }

    // --- Case A: Stance Change active ---
    let mut state_a = build(true);
    let branches_a = set_moves_and_run(&mut state_a, Choices::SHADOWBALL, Choices::SPLASH);
    let damage_a = first_branch_damage_to(&branches_a, SideReference::SideTwo);

    // --- Case B: control, no Stance Change ---
    let mut state_b = build(false);
    let branches_b = set_moves_and_run(&mut state_b, Choices::SHADOWBALL, Choices::SPLASH);
    let damage_b = first_branch_damage_to(&branches_b, SideReference::SideTwo);

    assert!(
        damage_a > damage_b,
        "Stance Change Aegislash should deal MORE damage than non-StanceChange (Blade spa > Shield spa). \
         got case_a={} case_b={}",
        damage_a,
        damage_b,
    );
}

// --------------------------------------------------------------------------
// Test 4 — Status moves OTHER than King's Shield do NOT trigger a swap.
// --------------------------------------------------------------------------
#[test]
fn stance_change_does_not_fire_on_non_kings_shield_status_move() {
    let mut state = State::default();
    let attacker = state.side_one.get_active();
    attacker.id = PokemonName::AEGISLASH;
    attacker.ability = Abilities::STANCECHANGE;

    let branches = set_moves_and_run(&mut state, Choices::SUBSTITUTE, Choices::SPLASH);

    // Sanity: Substitute is a Status-category move.
    assert_eq!(
        poke_engine::choices::MOVES
            .get(&Choices::SUBSTITUTE)
            .expect("SUBSTITUTE should be in MOVES")
            .category,
        MoveCategory::Status,
    );

    assert!(
        first_forme_change(&branches).is_none(),
        "no FormeChange should fire when AEGISLASH uses a non-damaging, non-King's-Shield move. \
         got: {:?}",
        branches,
    );
}

// --------------------------------------------------------------------------
// Test 5 — Switching in Aegislash-Blade reverts it to Aegislash (Shield).
//
// Showdown's actual behavior: Aegislash always returns to Shield form on
// switch-in. Without this revert, an Aegislash that swapped to Blade and
// switched out would stay Blade in the engine's simulation.
// --------------------------------------------------------------------------
#[test]
fn stance_change_reverts_to_shield_on_switch_in() {
    let mut state = State::default();
    // Bench Pokemon at P1 is Aegislash-Blade with Stance Change.
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::AEGISLASHBLADE;
    state.side_one.pokemon[PokemonIndex::P1].ability = Abilities::STANCECHANGE;

    let branches = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
        false,
    );

    let fc = first_forme_change(&branches)
        .expect("expected a FormeChange instruction when AEGISLASHBLADE switches in");
    assert_eq!(fc.side_ref, SideReference::SideOne);
    assert_eq!(
        fc.name_change,
        PokemonName::AEGISLASH as i16 - PokemonName::AEGISLASHBLADE as i16,
        "name_change delta should swap AEGISLASHBLADE -> AEGISLASH on switch-in",
    );

    // After applying the instructions of the first branch, the active Pokemon
    // should be Aegislash (Shield form).
    let first_branch = branches.first().expect("expected at least one branch");
    state.apply_instructions(&first_branch.instruction_list);
    assert_eq!(
        state.side_one.get_active().id,
        PokemonName::AEGISLASH,
        "Aegislash-Blade should revert to Aegislash (Shield) on switch-in",
    );
}
