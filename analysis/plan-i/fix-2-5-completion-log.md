# Fix #2.5: Eval Boost Recalibration — Completion Log

**Date:** 2026-04-30
**Branch:** feat/plan-i-prior-dampening
**Plan:** cobblemon-copilot/docs/superpowers/plans/2026-04-30-fix-2-5-eval-boost-recalibration.md

## What changed
- Deleted `boost_hp_multiplier` (HP-bracketed schedule) from `src/genx/evaluate.rs`.
- Removed the multiplier from 4 call sites (s1/s2 in both `evaluate` and `evaluate_breakdown`).
- Reduced `POKEMON_ATTACK_BOOST`, `POKEMON_SPECIAL_ATTACK_BOOST`, `POKEMON_SPEED_BOOST` from 30.0 → 25.0.
- Defensive boosts unchanged (`POKEMON_DEFENSE_BOOST`, `POKEMON_SPECIAL_DEFENSE_BOOST` remain 15.0).
- Deleted `test_boost_hp_multiplier_kaizo_brackets` and `test_boost_term_kaizo_schedule_in_breakdown` (pinned the deleted bracket schedule).
- Added `test_boost_reward_independent_of_hp`, `test_active_only_boost_reward`, `test_fainted_active_no_boost_reward`, `test_constants_match_design`.

## Why
The bracket schedule (full credit > 70% HP, zero 40-70%, negative < 40%) double-counted with `threat_score × THREAT_SCORE_WEIGHT (40)`, which already encodes a +2 boost as roughly +24 per side. Result: a 40-70% HP blind zone where setup sweepers registered as zero on both terms, producing two recent losses (celadoncityboogies Greninja Battle Bond at ~65% HP, Gregguru Ogerpon-Wellspring +2 Atk at ~70% HP). Foul-play (NeurIPS 2025 Gen9OU winner) uses flat boost credit; we now match.

## Verification
- `cargo test --release --features=gen9,terastallization` passes 1040 tests (0 failed, 3 ignored).
- Self-mortality is naturally handled by the active+alive gate on the enclosing `if iter.pokemon_index == active_index { if pkmn.hp > 0 { ... } }` block.

## Commits
- 6264eb6 test(fix-2.5): failing test for HP-independent boost reward
- d21492d feat(fix-2.5): remove HP-bracketed boost multiplier from evaluate
- 0053b20 test(fix-2.5): pin active-only + fainted-active boost gates
- 30b36b9 feat(fix-2.5): reduce offensive boost constants 30.0 → 25.0
- e4886b2 chore(fix-2.5): delete obsolete kaizo bracket tests and helper

## Next
Tier-4 replay verification (deferred): replay Gregguru T13 and celadoncityboogies T11 against the new eval to confirm the previously-recommended setup move now scores below the safer alternative. Requires harness setup; tracked separately.
