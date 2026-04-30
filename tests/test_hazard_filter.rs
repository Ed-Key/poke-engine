//! Hazard-already-up filter (Bug #2 from analysis/open-bugs/2026-04-30-engine-bug-catalog.md).
//!
//! When a hazard is already at its maximum stack on the opponent's side, the
//! corresponding hazard-setter move is a no-op. MCTS at root should NOT see
//! it as a usable option, because the heuristic eval is symmetric across
//! "click the no-op move" and "do nothing" — the search explores the no-op
//! branch and visits collapse onto a turn-wasting move (gelks T14 evidence:
//! Tyranitar with SR up clicked SR again at 47% confidence, opp Bulk Up'd).

use poke_engine::choices::Choices;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonMoveIndex, State};

fn fresh_state_with_s1_move(move_id: Choices) -> State {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, move_id);
    state
}

fn s1_has_first_move(state: &State) -> bool {
    let (s1_opts, _) = state.root_get_all_options();
    s1_opts
        .iter()
        .any(|o| matches!(o, MoveChoice::Move(PokemonMoveIndex::M0)))
}

#[test]
fn stealth_rock_filtered_when_already_up_on_opp_side() {
    let mut state = fresh_state_with_s1_move(Choices::STEALTHROCK);
    state.side_two.side_conditions.stealth_rock = 1;
    assert!(
        !s1_has_first_move(&state),
        "Stealth Rock should be filtered when SR already up on side_two"
    );
}

#[test]
fn stealth_rock_present_when_not_up() {
    let state = fresh_state_with_s1_move(Choices::STEALTHROCK);
    assert_eq!(state.side_two.side_conditions.stealth_rock, 0);
    assert!(
        s1_has_first_move(&state),
        "Stealth Rock should be a usable option when SR is not yet up (regression guard)"
    );
}

#[test]
fn spikes_filtered_at_max_stack() {
    let mut state = fresh_state_with_s1_move(Choices::SPIKES);
    state.side_two.side_conditions.spikes = 3;
    assert!(
        !s1_has_first_move(&state),
        "Spikes should be filtered when already at 3 layers on side_two"
    );
}

#[test]
fn spikes_allowed_below_max_stack() {
    let mut state = fresh_state_with_s1_move(Choices::SPIKES);
    state.side_two.side_conditions.spikes = 2;
    assert!(
        s1_has_first_move(&state),
        "Spikes at 2 layers should still be usable (one more layer adds value)"
    );
}

#[test]
fn toxic_spikes_filtered_at_max_stack() {
    let mut state = fresh_state_with_s1_move(Choices::TOXICSPIKES);
    state.side_two.side_conditions.toxic_spikes = 2;
    assert!(
        !s1_has_first_move(&state),
        "Toxic Spikes should be filtered when already at 2 layers on side_two"
    );
}

#[test]
fn sticky_web_filtered_when_already_up() {
    let mut state = fresh_state_with_s1_move(Choices::STICKYWEB);
    state.side_two.side_conditions.sticky_web = 1;
    assert!(
        !s1_has_first_move(&state),
        "Sticky Web should be filtered when already up on side_two"
    );
}

#[test]
fn side_two_filter_uses_side_one_conditions() {
    // Mirror direction: when side_two is the attacker, the filter must look
    // at side_one's conditions, not its own.
    let mut state = State::default();
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::STEALTHROCK);
    state.side_one.side_conditions.stealth_rock = 1;
    let (_, s2_opts) = state.root_get_all_options();
    let s2_has_sr = s2_opts
        .iter()
        .any(|o| matches!(o, MoveChoice::Move(PokemonMoveIndex::M0)));
    assert!(
        !s2_has_sr,
        "side_two's SR should be filtered when side_one already has SR up"
    );
}
