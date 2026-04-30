# Plan: Aegislash Stance Change handler

## Goal

Wire `Abilities::STANCECHANGE` so Aegislash flips to AEGISLASHBLADE before
its damaging moves are rolled, and back to AEGISLASH when it uses King's
Shield — matching Showdown's `[from] ability: Stance Change` protocol
events and fixing the threat-underprediction observed in battle 2598193640.

## Working environment

- Worktree: `/Users/edkiboma/Projects/poke-engine/.worktrees/plan-i`
- Branch: `feat/plan-i-prior-dampening`
- Pre-fix HEAD (Fake Out gate): `db57fa6`
- Failing-tests commit (TDD red phase): `6584592`
- Build/test: `cargo test --release --features=gen9,terastallization`
- Baseline: 1012 tests passing pre-this-work; +4 new tests (1 passing, 3 red).
- Off-limits ports: 7267, 7270, 7271, 7273.

## Files to modify

1. **`src/genx/base_stats.rs`** — add `PokemonName::AEGISLASH` and
   `PokemonName::AEGISLASHBLADE` to the `base_stats()` match.
   `recalculate_stats` will panic without them. (~2 lines.)

2. **`src/genx/abilities.rs`** — add the `STANCECHANGE` arm inside the
   `match active_pkmn.ability` block in `pub fn ability_before_move`
   (currently at line 583). Mirror the IceFace/Cramorant pattern:
   push `Instruction::FormeChange`, mutate `active_pkmn.id`, then call
   `active_pkmn.recalculate_stats(side_ref, instructions)`. (~25 lines.)

3. **`tests/test_aegislash_stance_change.rs`** — already added as commit
   `6584592` (red phase). No further changes needed — implementation must
   make the 3 failing tests pass without breaking the 1 passing test.

No changes required to `src/genx/generate_instructions.rs` — the existing
`before_move` already calls `ability_before_move` at the right point in
the pipeline (line 1588, before damage calc and modify_attack_being_used).

## Existing patterns we mirror

- **IceFace** (`abilities.rs:1378–1430`, `1403–1416`): swaps EISCUE ↔
  EISCUENOICE on switch-in / weather change. Pushes `FormeChange`, sets
  `active_pkmn.id`, calls `recalculate_stats`.
- **Disguise** (`abilities.rs:542–579`): defending side, hooked in
  `ability_before_move` against incoming damaging moves. Same instruction
  shape (`FormeChange` + `recalculate_stats`).
- **Gulp Missile / Cramorant** (`abilities.rs:584–601`): attacking side
  forme change inside `ability_before_move`'s `match active_pkmn.ability`
  block — exact same hook and shape we need.

## Hook semantics

- **Damaging-move case** — fires when `active_pkmn.id == AEGISLASH` AND
  `choice.category != MoveCategory::Status`. King's Shield is Status,
  Substitute is Status, so neither will accidentally trigger this arm.
  Hook point: `ability_before_move` runs in `before_move` (line 1588 of
  `generate_instructions.rs`) BEFORE `ability_modify_attack_being_used`
  (line 1594), which is where atk/spa are read for damage calc — so the
  swap correctly affects this move.
- **King's Shield case** — fires when `active_pkmn.id == AEGISLASHBLADE`
  AND `choice.move_id == Choices::KINGSSHIELD`. (King's Shield is the only
  move that flips Blade back to Shield.) Same hook (`ability_before_move`),
  fires regardless of whether Protect-style success is rolled, matching
  Showdown — Stance Change reverts on USE, not on success.

## TDD steps

1. **Red** (DONE, commit `6584592`): 4 tests in
   `tests/test_aegislash_stance_change.rs`, 3 fail with informative
   panics:
   - `stance_change_swaps_to_blade_on_damaging_move`
   - `stance_change_swaps_back_to_shield_on_kings_shield`
   - `stance_change_uses_blade_spa_for_damage_calc`
   - `stance_change_does_not_fire_on_non_kings_shield_status_move` (passes
     vacuously)

2. **Green** (TO DO):
   - Add Aegislash entries to `src/genx/base_stats.rs`:
     - `AEGISLASH` → `(60, 50, 150, 50, 150, 60)`
     - `AEGISLASHBLADE` → `(60, 150, 50, 150, 50, 60)`
   - In `src/genx/abilities.rs::ability_before_move`, inside the
     `match active_pkmn.ability` block (active side, near line 583), add:
     ```rust
     Abilities::STANCECHANGE => {
         let new_forme = if active_pkmn.id == PokemonName::AEGISLASH
             && choice.category != MoveCategory::Status
         {
             Some(PokemonName::AEGISLASHBLADE)
         } else if active_pkmn.id == PokemonName::AEGISLASHBLADE
             && choice.move_id == Choices::KINGSSHIELD
         {
             Some(PokemonName::AEGISLASH)
         } else {
             None
         };
         if let Some(new_forme) = new_forme {
             instructions.instruction_list.push(Instruction::FormeChange(
                 FormeChangeInstruction {
                     side_ref: *side_ref,
                     name_change: new_forme as i16 - active_pkmn.id as i16,
                 },
             ));
             active_pkmn.id = new_forme;
             active_pkmn.recalculate_stats(side_ref, instructions);
         }
     }
     ```
     `Choices` and `FormeChangeInstruction` are already imported at the
     top of `abilities.rs` (used by Cramorant / IceFace / Disguise).

3. **Verify**:
   - `cargo test --release --features=gen9,terastallization --test test_aegislash_stance_change`
     — all 4 must pass.
   - `cargo test --release --features=gen9,terastallization --no-fail-fast`
     — full suite must show baseline 1012 + 4 new passing (1016 total),
     0 failures.
   - Spot-check: pre-existing `test_iceface_*` and `test_minior_*`
     forme-change tests still pass (we don't touch their code paths,
     but they exercise the same `recalculate_stats` plumbing).

## Risk

1. **Order-of-instructions bug**: if implementation puts `FormeChange`
   AFTER damage calc instead of before, damage will use Shield-form
   spa/atk and `stance_change_uses_blade_spa_for_damage_calc` catches it
   (case_a > case_b assertion). Mitigated by hooking
   `ability_before_move` (runs before `ability_modify_attack_being_used`).

2. **Forgotten `base_stats()` entry**: the `_ => panic!` arm in
   `src/genx/base_stats.rs:78` will crash any forme change. The first
   passing test will smoke-test this immediately.

3. **`base_ability` divergence**: Showdown treats Stance Change as
   permanent (Skill Swap / Worry Seed fail). Existing handlers don't
   guard against `base_ability` here either, and our scope only covers
   the form swap. If a future test team Skill-Swaps onto Aegislash, the
   engine may misbehave — out of scope, document as a known gap.

4. **King's Shield protect logic**: King's Shield is also a Protect-like
   move that adds `KINGSSHIELD` volatile. Our handler runs in
   `ability_before_move` BEFORE the Protect/volatile-status logic, so
   the volatile-status side-effect path is unchanged. The form swap
   should be the only new instruction.

## Estimated LOC

- `src/genx/base_stats.rs`: +2 lines.
- `src/genx/abilities.rs`: ~25 lines (one new match arm with two
  sub-conditions + standard `FormeChange` + `recalculate_stats` shape).
- Tests: already shipped at `6584592` (+202 lines, one-time).
- **Total implementation diff: ~27 lines.** Matches the audit's "~30
  lines" estimate.

## Out of scope (deferred)

- Aegislash team-preview detection (lead always shows Shield in Showdown
  protocol — already implicit since Aegislash IS the lead form).
- Skill Swap / Worry Seed / Gastro Acid interactions (no test team uses
  these against Aegislash today).
- Mold Breaker bypass: per Bulbapedia, Stance Change is NOT bypassed by
  Mold Breaker. The existing `mold_breaker_ignores` filter on
  `abilities.rs:355` does not list `STANCECHANGE`, so no change needed.
- Mega-evolution interaction: AEGISLASH has no mega.
