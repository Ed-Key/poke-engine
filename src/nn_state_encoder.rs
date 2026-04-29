//! Plan E state encoder: rust `State` -> `BattleRequest` JSON for the metamon
//! sidecar.
//!
//! The Phase 2 sidecar consumes the same `BattleRequest` shape that
//! `translate.rs` ALREADY parses inward (see `BattleRequest`/`SideInput`/
//! `PokemonInput`/`MoveInput` at `src/translate.rs:14-92`). This file is the
//! reverse direction: take a Rust `State` and emit equivalent JSON.
//!
//! TODO: this module is a Day 2 placeholder filled in later in Plan E Phase
//! 4-5. The Day 1 commit only wires up the file so the lib compiles.

// (Day 2 implementation lands in a follow-up commit.)
