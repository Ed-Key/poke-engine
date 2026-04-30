// Test for the Fake Out legality bug discovered in live battle 2597942559
// (Mega Medicham, T17–T19). Showdown's `Battle.runMove` enforces:
//
//     if (move.flags['failwhenusedasfirstturn'] && pokemon.activeTurns > 0) {
//         this.attrLastMove('[still]');
//         this.add('-fail', pokemon);
//         return false;
//     }
//
// (See Showdown sim/battle.ts ~ "useMoveInner".)
//
// In poke-engine, `Side::add_available_moves` (genx/state.rs:334) only filters
// Bloodmoon / Gigaton-Hammer repeats and Encore-locked moves. Fake Out is left
// as a legal option and `choice_effects.rs:213` only neuters its damage/flinch
// at execution time. Result: MCTS happily allocates millions of visits to a
// move that Showdown would auto-fail, producing wildly wrong opp predictions
// any turn that the active Pokemon was already on the field last turn.
//
// These tests reproduce that bug and will pass once `add_available_moves`
// gates Fake Out behind `last_used_move == LastUsedMove::Switch(_)` (the same
// signal `choice_effects.rs:213` already uses for damage neutering).

#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::choices::Choices;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{LastUsedMove, PokemonIndex, PokemonMoveIndex, State};

/// Bug repro: Medicham used Fake Out on turn N. On turn N+1 the engine still
/// lists Fake Out among side_two's options, even though `last_used_move` is
/// `Move(M0)` — proving the active wasn't just-switched-in.
#[test]
fn test_fakeout_illegal_when_last_used_move_is_a_move() {
    let mut state = State::default();
    state.use_last_used_move = true;

    // Side two active = Medicham-style: Fake Out at slot M0, Tackle at M1.
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::FAKEOUT);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::TACKLE);

    // It used Fake Out last turn — exactly the live-battle Medicham state.
    state.side_two.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);

    let (_s1_opts, s2_opts) = state.root_get_all_options();

    assert!(
        !s2_opts.contains(&MoveChoice::Move(PokemonMoveIndex::M0)),
        "Fake Out (M0) MUST NOT be a legal option after a non-switch \
         last_used_move — Showdown auto-fails this. Options were: {:?}",
        s2_opts,
    );
    // Sanity: a non-Fake-Out move is still legal.
    assert!(
        s2_opts.contains(&MoveChoice::Move(PokemonMoveIndex::M1)),
        "Tackle (M1) should still be legal. Options: {:?}",
        s2_opts,
    );
}

/// Stronger repro mirroring `last_used_move = "switch:0"` in proxy land —
/// this is what an actual just-switched-in Pokemon looks like, and it's the
/// ONE case where Fake Out should be legal. Guards against an over-eager fix
/// that would also remove Fake Out on the legitimate first turn.
#[test]
fn test_fakeout_legal_when_just_switched_in() {
    let mut state = State::default();
    state.use_last_used_move = true;

    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::FAKEOUT);

    // First turn out — last action was a switch.
    state.side_two.last_used_move = LastUsedMove::Switch(PokemonIndex::P0);

    let (_s1_opts, s2_opts) = state.root_get_all_options();

    assert!(
        s2_opts.contains(&MoveChoice::Move(PokemonMoveIndex::M0)),
        "Fake Out should still be legal on the turn we just switched in. \
         Options: {:?}",
        s2_opts,
    );
}

/// Defensive: `LastUsedMove::None` (e.g. battle start, or proxy never sent
/// the field) should also keep Fake Out legal. The fix should ONLY filter
/// when last_used_move is a concrete previous Move(...).
#[test]
fn test_fakeout_legal_when_last_used_move_is_none() {
    let mut state = State::default();
    state.use_last_used_move = true;

    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::FAKEOUT);
    // Default — no prior action recorded.
    state.side_two.last_used_move = LastUsedMove::None;

    let (_s1_opts, s2_opts) = state.root_get_all_options();

    assert!(
        s2_opts.contains(&MoveChoice::Move(PokemonMoveIndex::M0)),
        "With LastUsedMove::None the engine has no evidence of prior turns; \
         Fake Out should remain legal so we don't false-positive battle starts \
         or proxy gaps. Options: {:?}",
        s2_opts,
    );
}

/// Same bug, side one. Just to prove the missing gate is symmetric — a fix
/// that only patches one side would still fail this.
#[test]
fn test_fakeout_illegal_side_one_after_prior_move() {
    let mut state = State::default();
    state.use_last_used_move = true;

    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::FAKEOUT);
    state.side_one.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M0);

    let (s1_opts, _s2_opts) = state.root_get_all_options();

    assert!(
        !s1_opts.contains(&MoveChoice::Move(PokemonMoveIndex::M0)),
        "Fake Out must be filtered on side_one too. Options: {:?}",
        s1_opts,
    );
}
