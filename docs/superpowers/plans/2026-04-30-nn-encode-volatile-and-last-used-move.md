# NN Encoder: serialize `last_used_move` + active `volatile_statuses`

**Plan I — Fix #4 (engine open-bug catalog 2026-04-30, item 4).**
**Branch:** `feat/plan-i-prior-dampening` (worktree at `.worktrees/plan-i`).

## Goal

Extend `src/nn_state_encoder.rs` so the JSON sent to the Kakuna NN sidecar
includes (a) each side's `last_used_move` and (b) the active mon's
policy-relevant `volatile_statuses`. Today both are silently dropped.

## CAVEAT (read first)

**This fix is plumbing-only — it does NOT improve Kakuna's predictions today.**

The current sidecar model (`metamon-spike/sidecar/kakuna_loader.py`) was
trained on a fixed input schema that does NOT include `lastUsedMove` or
`volatileStatuses`. Adding these fields will:
- be silently ignored by the sidecar's translator
  (`PolicyRequest.state: dict[str, Any]` — no field validation)
- NOT cause the model to output better priors for choice-locked or
  CONFUSION/SUBSTITUTE/LOCKEDMOVE states

Why ship now anyway:
1. **Future-proofing** — when we retrain or fine-tune Kakuna with the
   richer schema, the data will be present in every served state.
2. **Schema alignment** — the encoded JSON should faithfully reflect the
   simulator's `State`. Missing fields create silent drift between what
   Layer 5 simulates and what Layer 1 was supposedly trained on.

The user should **not** expect immediate sim-WR improvement from this fix.

## Files to modify

- `src/nn_state_encoder.rs` (only):
  - `encode_side(side: &Side) -> Value` — add `lastUsedMove` field.
  - `encode_pokemon(p: &Pokemon, boosts: Value) -> Value` — add
    `volatileStatuses` field. Caller must thread the active mon's
    side-level volatiles into this slot only; reserves get `[]`.
  - Add a small `encode_volatile_statuses(set) -> Value` helper that
    filters down to the policy-relevant set and emits an array of
    `Debug`-formatted enum names (uppercase, e.g. `"CONFUSION"`).

**DO NOT touch:** `map_policy_to_options*` (alphabetical-permutation
logic, owned by Plan E and unrelated to this fix).

## Schema

Top-level encoded JSON gains two fields (camelCase, mirroring the
existing convention and `BattleRequest` input format in `translate.rs`):

```json
{
  "sideOne": {
    "...existing...": "...",
    "lastUsedMove": "move:1",       // or "switch:2", or "move:none"
    "pokemon": [
      {
        "...existing...": "...",
        "volatileStatuses": ["CONFUSION", "SUBSTITUTE"]
      },
      { "...reserve...": "...", "volatileStatuses": [] },
      ...
    ]
  },
  "sideTwo": { ... },
  ...
}
```

### `lastUsedMove` format
Mirror `LastUsedMove::serialize` (state.rs:55–62), which is already what
`translate.rs:851` reads back:
- `LastUsedMove::Move(M0..M3)` → `"move:0"` / `"move:1"` / `"move:2"` / `"move:3"`
- `LastUsedMove::Switch(P0..P5)` → `"switch:0"` … `"switch:5"`
- `LastUsedMove::None` → `"move:none"`

### `volatileStatuses` format
Array of strings. Filter the engine's `HashSet<PokemonVolatileStatus>` down
to the policy-relevant set:
- `CONFUSION`, `LEECHSEED`, `SUBSTITUTE`
- `LOCKEDMOVE`, `ENCORE`, `TAUNT`, `YAWN`

Drop everything else (FLINCH, ROOST, PROTECT, FOCUSENERGY, the
PROTOSYNTHESIS/QUARKDRIVE family, etc.) — these are too fleeting or too
many-valued to enrich the policy meaningfully and would just dilute the
input. Output is `format!("{:?}", v)` so naming matches the enum
(`"CONFUSION"`, etc., uppercase).

The active-mon array is always present (empty when clean) so the schema
doesn't go missing-vs-present across turns. Reserves always get `[]`.

## TDD test list

Failing tests committed in `tests/test_nn_state_encoder_extensions.rs`
(6 tests, all currently red):

1. `encode_serializes_last_used_move_for_each_side` — set
   `side_one.last_used_move = LastUsedMove::Move(M1)` and
   `side_two.last_used_move = LastUsedMove::Switch(P2)`; assert
   `sideOne.lastUsedMove == "move:1"` and `sideTwo.lastUsedMove == "switch:2"`.
2. `encode_serializes_battle_start_last_used_move` —
   `LastUsedMove::None` on both sides → `"move:none"`.
3. `encode_serializes_volatile_statuses_array` — insert CONFUSION +
   LEECHSEED into `side_one.volatile_statuses`; encoded active mon's
   `volatileStatuses` contains exactly those two.
4. `encode_omits_uninteresting_volatile_statuses` — insert
   {CONFUSION, SUBSTITUTE, LOCKEDMOVE, FLINCH, ROOST, FOCUSENERGY};
   only the first three survive in the encoded array.
5. `encode_volatile_statuses_empty_when_clean` — default state →
   `volatileStatuses == []` (present, empty).
6. `encode_volatile_statuses_only_on_active_mon` — reserves all get
   `[]`, only the active slot reflects the inserted volatile.

Test commit SHA: see Step 5 below.

## Estimated LOC

- `encode_volatile_statuses` helper: ~10
- Threading volatiles through `encode_pokemon` (one new arg or
  side-aware caller): ~10
- Adding `lastUsedMove` to `encode_side`: ~5
- Updating `encode_side` to pass volatiles only to active slot: ~10

**Target: ~50 LOC.** Stretch ceiling: 60 LOC.

## Risks (top 3)

1. **(MITIGATED) Sidecar rejects extra fields.** Verified safe:
   `PolicyRequest.state: dict[str, Any]` (`nn_sidecar.py:67`) accepts any
   keys, and `state_translator.py` only reads named keys. No risk of
   500-errors on the live :7267/:7270/:7271/:7273 sidecars.

2. **Wrong filter set hides a useful signal.** We're hand-picking which
   volatiles matter (CONFUSION / LEECHSEED / SUBSTITUTE / LOCKEDMOVE /
   ENCORE / TAUNT / YAWN). If a future Kakuna fine-tune wants e.g.
   PERISH3/2/1 or PROTECT or ATTRACT, we'll have to widen the filter and
   regenerate training data. Mitigation: filter list lives in one
   helper; adding more is a one-line edit and a bumped test.

3. **Reserves don't carry per-mon volatiles, but the schema implies they
   could.** The engine model only tracks side-level volatiles for the
   active mon; `Pokemon` itself has no volatile_statuses field. Test #6
   explicitly locks down `[]` for reserves so a future refactor can't
   silently send reserve data that doesn't exist. Risk is purely
   self-consistency, not correctness.

## Out of scope / non-goals

- Retraining Kakuna with the new fields (orthogonal effort).
- Adding `volatile_status_durations` to the encoded state — useful, but
  out of scope for this 50-LOC fix; file separately if it's needed.
- Touching `map_policy_to_options*` — Fix #1 (heuristic prior) territory.
- Adding `last_used_move` to per-Pokemon entries — only side-level is
  needed; the engine doesn't track per-mon last-move history.
