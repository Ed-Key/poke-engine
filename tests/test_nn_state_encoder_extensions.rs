//! Plan I Fix #4: NN encoder serialization gaps for `last_used_move` and
//! `volatile_statuses`.
//!
//! Without these fields the Kakuna NN policy can't see when an opponent is
//! locked into Outrage / Sleep Talk, or when CONFUSION / SUBSTITUTE / etc.
//! are active. The simulator (Layer 5) handles them correctly; this fix
//! patches the **prior** layer (Layer 1) so any future retrained Kakuna
//! receives a faithful state. (The current sidecar model has no weights
//! for these fields — see plan doc caveat.)
//!
//! Schema (camelCase, mirrors `BattleRequest` in `translate.rs`):
//!   - `sideOne.lastUsedMove`     → `"move:<idx>"` | `"switch:<idx>"` | `"move:none"`
//!   - `sideTwo.lastUsedMove`     → same.
//!   - `sideOne.pokemon[active].volatileStatuses` → `["CONFUSION", "LEECHSEED", ...]`
//!     (only policy-relevant statuses; uninteresting ones are dropped).
//!
//! These tests fail until `encode_side` and `encode_pokemon` in
//! `src/nn_state_encoder.rs` are extended.

#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::engine::state::PokemonVolatileStatus;
use poke_engine::nn_state_encoder::{encode, SidePerspective};
use poke_engine::state::{LastUsedMove, PokemonIndex, PokemonMoveIndex, State};

// ---------------------------------------------------------------------------
// last_used_move
// ---------------------------------------------------------------------------

#[test]
fn encode_serializes_last_used_move_for_each_side() {
    // side_one used its M1 last turn; side_two just switched in to slot P2.
    let mut state = State::default();
    state.side_one.last_used_move = LastUsedMove::Move(PokemonMoveIndex::M1);
    state.side_two.last_used_move = LastUsedMove::Switch(PokemonIndex::P2);

    let encoded = encode(&state, SidePerspective::Side1);

    // Format mirrors `LastUsedMove::serialize` (state.rs:55-62) and the
    // input convention documented at translate.rs:56-64.
    assert_eq!(
        encoded["sideOne"]["lastUsedMove"].as_str(),
        Some("move:1"),
        "side_one lastUsedMove not serialized correctly: {}",
        encoded["sideOne"]["lastUsedMove"],
    );
    assert_eq!(
        encoded["sideTwo"]["lastUsedMove"].as_str(),
        Some("switch:2"),
        "side_two lastUsedMove not serialized correctly: {}",
        encoded["sideTwo"]["lastUsedMove"],
    );
}

#[test]
fn encode_serializes_battle_start_last_used_move() {
    // Default-constructed state has LastUsedMove::None on both sides; this
    // is the "turn 1, no move yet" case the sidecar must see as `move:none`.
    let state = State::default();
    let encoded = encode(&state, SidePerspective::Side1);

    assert_eq!(
        encoded["sideOne"]["lastUsedMove"].as_str(),
        Some("move:none"),
    );
    assert_eq!(
        encoded["sideTwo"]["lastUsedMove"].as_str(),
        Some("move:none"),
    );
}

// ---------------------------------------------------------------------------
// volatile_statuses
// ---------------------------------------------------------------------------

#[test]
fn encode_serializes_volatile_statuses_array() {
    // Active mon on side_one is confused + leech-seeded.
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);

    let encoded = encode(&state, SidePerspective::Side1);
    let active_idx = encoded["sideOne"]["activeIndex"].as_u64().unwrap() as usize;
    let vs = encoded["sideOne"]["pokemon"][active_idx]["volatileStatuses"]
        .as_array()
        .expect("volatileStatuses array on active pokemon");

    let strs: Vec<String> = vs
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        strs.contains(&"CONFUSION".to_string()),
        "expected CONFUSION in volatileStatuses, got {:?}",
        strs,
    );
    assert!(
        strs.contains(&"LEECHSEED".to_string()),
        "expected LEECHSEED in volatileStatuses, got {:?}",
        strs,
    );
    assert_eq!(strs.len(), 2, "expected exactly 2 entries, got {:?}", strs);
}

#[test]
fn encode_omits_uninteresting_volatile_statuses() {
    // Mix policy-relevant (CONFUSION, SUBSTITUTE, LOCKEDMOVE) with several
    // we don't want polluting the policy input (FLINCH — fleeting, ROOST
    // — single-turn typing flag, FOCUSENERGY — minor crit flag).
    let mut state = State::default();
    let interesting = [
        PokemonVolatileStatus::CONFUSION,
        PokemonVolatileStatus::SUBSTITUTE,
        PokemonVolatileStatus::LOCKEDMOVE,
    ];
    let uninteresting = [
        PokemonVolatileStatus::FLINCH,
        PokemonVolatileStatus::ROOST,
        PokemonVolatileStatus::FOCUSENERGY,
    ];
    for v in interesting.iter().chain(uninteresting.iter()) {
        state.side_one.volatile_statuses.insert(*v);
    }

    let encoded = encode(&state, SidePerspective::Side1);
    let active_idx = encoded["sideOne"]["activeIndex"].as_u64().unwrap() as usize;
    let vs = encoded["sideOne"]["pokemon"][active_idx]["volatileStatuses"]
        .as_array()
        .expect("volatileStatuses array on active pokemon");

    let strs: Vec<String> = vs
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    for k in ["CONFUSION", "SUBSTITUTE", "LOCKEDMOVE"] {
        assert!(
            strs.contains(&k.to_string()),
            "expected {} retained in encoded volatileStatuses, got {:?}",
            k,
            strs,
        );
    }
    for k in ["FLINCH", "ROOST", "FOCUSENERGY"] {
        assert!(
            !strs.contains(&k.to_string()),
            "expected {} dropped from encoded volatileStatuses (uninteresting), got {:?}",
            k,
            strs,
        );
    }
}

#[test]
fn encode_volatile_statuses_empty_when_clean() {
    // Default state has no volatiles — encoded array must exist and be empty
    // (rather than omitted) so the sidecar schema is stable across turns.
    let state = State::default();
    let encoded = encode(&state, SidePerspective::Side1);
    let active_idx = encoded["sideOne"]["activeIndex"].as_u64().unwrap() as usize;
    let vs = encoded["sideOne"]["pokemon"][active_idx]["volatileStatuses"]
        .as_array()
        .expect("volatileStatuses array always present (empty when clean)");
    assert_eq!(vs.len(), 0, "expected empty volatileStatuses, got {:?}", vs);
}

#[test]
fn encode_volatile_statuses_only_on_active_mon() {
    // Reserves should not carry volatile_statuses (the engine model only
    // tracks side-level volatiles for the active mon). Encoded reserves
    // get an empty array.
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::CONFUSION);

    let encoded = encode(&state, SidePerspective::Side1);
    let active_idx = encoded["sideOne"]["activeIndex"].as_u64().unwrap() as usize;
    for slot in 0..6usize {
        let vs = encoded["sideOne"]["pokemon"][slot]["volatileStatuses"]
            .as_array()
            .unwrap_or_else(|| {
                panic!(
                    "volatileStatuses missing on slot {} (should be empty array)",
                    slot
                )
            });
        if slot == active_idx {
            assert_eq!(vs.len(), 1);
        } else {
            assert_eq!(
                vs.len(),
                0,
                "reserve slot {} should have empty volatileStatuses, got {:?}",
                slot,
                vs
            );
        }
    }
}
