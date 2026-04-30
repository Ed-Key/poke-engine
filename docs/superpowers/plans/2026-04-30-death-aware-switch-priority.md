# Plan I — Fix #1.6: Death-aware switch priority in compute()

**Date:** 2026-04-30
**Worktree:** `/Users/edkiboma/Projects/poke-engine/.worktrees/plan-i`
**Branch:** `feat/plan-i-prior-dampening`
**Target file:** `src/heuristic_prior.rs` (only)
**Red-phase commit:** `e3d4f91` — `test(plan-i): failing tests for death-aware switch priority`

## Goal (one sentence)

When the active Pokemon is in a "definitely dies this turn" state (predicted opp damage ≥ my HP, opp moves first, no priority move that fires), `compute()` must shift heuristic prior mass aggressively from the dying-move slots onto the switch slot so MCTS sees a decisive Q-gap on switch (battle 2598375717 vs waldenjames T4: Adamant Dragonite 304/324 vs Pelipper Ice Beam OHKO; engine returned Q=0.53 on Heatran switch vs ~0.5x on stay-in attack — too close — Mariga clicked attack and died).

## Bug mechanism observed in red-phase output

Test 1 with HEAD `e3d4f91`:
```
probs = [0.0875, 0.0875, 0.0875, 0.0875,  // dying move slots 0..3 (each get uniform leak)
         0.3,                              // switch slot (Heatran)
         0.0, 0.0, 0.0, 0.0,
         0.0875, 0.0875, 0.0875, 0.0875]   // tera variants of dying moves
```
Fix #1.5 already filters dying moves out of `damage_calc_top_move` (returns `None`). But `compute()`'s uniform-fill branch then redistributes the un-placed `1 - mass_switch = 0.7` across all OTHER legal slots, INCLUDING the dying move slots, giving each dying move ~0.0875. The switch only gets 0.3. Net: NN-blended prior is too close on stay-in vs switch → MCTS Q's stay close → user clicks the wrong action.

## Approach

In `compute()`, after resolving `dmg_pick` and `switch_pick`, detect the "definitely dying" state and dynamically swap the mass parameters before the existing placement logic runs.

```rust
let dmg_pick = damage_calc_top_move(state, perspective);
let switch_pick = matchup_switch_pick(state, perspective);

if dmg_pick.is_none() && switch_pick.is_none() {
    return None;
}

// Fix #1.6: detect "definitely dying" state and override masses to
// concentrate on the switch slot. Only fires when a viable switch
// exists — if there's nothing to switch to, fall through.
let (mass_dmg, mass_switch) = if switch_pick.is_some()
    && is_definitely_dying(state, perspective)
{
    // Dump 90% on switch, leave 10% for the rest (uniform fill pulls
    // from this remainder; dying-move slots will each get << 0.05).
    (0.0_f32, 0.9_f32)
} else {
    (mass_dmg, mass_switch)
};

// ... existing slot-resolution + placement code unchanged ...
```

Where `is_definitely_dying` is a new private helper:

```rust
fn is_definitely_dying(state: &State, perspective: SidePerspective) -> bool {
    let (my_side_ref, opp_side_ref, active_idx) = match perspective {
        SidePerspective::Side1 => (
            SideReference::SideOne, SideReference::SideTwo,
            state.side_one.active_index,
        ),
        SidePerspective::Side2 => (
            SideReference::SideTwo, SideReference::SideOne,
            state.side_two.active_index,
        ),
    };
    let active = match perspective {
        SidePerspective::Side1 => state.side_one.get_active_immutable(),
        SidePerspective::Side2 => state.side_two.get_active_immutable(),
    };

    let (predicted_opp_dmg, opp_priority) =
        predicted_opp_max_damage_and_priority(state, active_idx, perspective);
    let i_die = (active.hp - predicted_opp_dmg) <= 0;
    if !i_die {
        return false;
    }

    // My top fire-able priority among legal damaging moves.
    let mut my_top_priority: i8 = i8::MIN;
    let mut have_damaging = false;
    for mv in active.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 { continue; }
        let c = &mv.choice;
        if c.category == MoveCategory::Status || c.category == MoveCategory::Switch {
            continue;
        }
        have_damaging = true;
        if c.priority > my_top_priority { my_top_priority = c.priority; }
    }
    if !have_damaging {
        // Status-only active — `damage_calc_top_move` already returns
        // None; let normal flow handle it. Don't trigger dying-penalty.
        return false;
    }

    let my_speed = get_effective_speed(state, &my_side_ref);
    let opp_speed = get_effective_speed(state, &opp_side_ref);

    // I "definitely move after" iff opp's predicted move strictly
    // out-prioritizes mine, OR priorities tie and opp's speed >= mine.
    let i_definitely_move_after = (my_top_priority < opp_priority)
        || (my_top_priority == opp_priority && my_speed <= opp_speed);

    i_die && i_definitely_move_after
}
```

### Why `mass_dmg = 0.0` instead of "shift by mass_dmg × 0.7"

The brief proposed `(mass_dmg * 0.3, mass_switch + mass_dmg * 0.7)` = `(0.18, 0.72)`. Empirically with the red-phase fixture: switch_pick is the only thing fired by the dmg-side (because Fix #1.5 returns `None` for dmg_pick when dying), and we want the dying move slots to get the LEAST possible mass. Cleanest rule: zero out the dmg allocation entirely when dying — the existing uniform-fill then spreads only `1 - 0.9 = 0.1` across all non-switch slots (including dying moves), giving each dying move ≪ 0.025. Switch dominates by ~36×.

This keeps the change minimal: a 2-line override of the input parameters; all downstream slot-resolution and placement logic stays identical.

## Files to modify

- `src/heuristic_prior.rs` — add `is_definitely_dying` helper, add the mass-override block at the top of `compute()`. **No other files.**

No imports change (`SideReference`, `get_effective_speed`, `MoveCategory` are all already in scope from Fix #1.5).

## TDD test list

File: `tests/test_death_aware_switch_priority.rs` (committed in red phase, `e3d4f91`).

Red-phase verification under HEAD `e3d4f91`:

| # | Test | Status | Why |
|---|------|--------|-----|
| 1 | `heuristic_shifts_mass_to_switch_when_definitely_dying` | **FAIL** | switch=0.3, dying_move=0.0875 — bug-trigger |
| 2 | `heuristic_keeps_normal_split_when_i_survive` | ok | regression guard |
| 3 | `heuristic_keeps_normal_split_when_i_have_priority` | ok | regression guard |
| 4 | `heuristic_falls_through_when_no_viable_switch` | ok | regression guard |

Test 1 asserts (post-fix):
- `result.matchup_switch_pick == Some(HEATRAN)`
- `switch_mass > 0.5` (target: 0.9)
- per-slot `dying_move_mass < 0.15` (target: ~0.012 = 0.1 / 8 unfilled slots)
- `switch_mass > dying_move_mass`
- `probs.iter().sum() ≈ 1.0`

Test 2 asserts: `dmg_mass ≈ 0.6, switch_mass ≈ 0.3` (full HP Dragonite vs weakened Pelipper, i_survive==true → no override).

Test 3 asserts: same as Test 2 — Dragonite 60/324 with Extreme Speed (priority +2) vs Pelipper (priority 0). My top priority > opp priority → `i_definitely_move_after == false` → no override.

Test 4 asserts: `compute() == None` — bench fully fainted + dying active means both `dmg_pick` (Fix #1.5 filter) and `switch_pick` (no candidates) return None. Override branch is gated on `switch_pick.is_some()`, so it doesn't fire here.

Existing tests in `tests/test_heuristic_prior.rs`, `tests/test_priority_aware_damage.rs`, `tests/test_survivability_switch.rs` must continue to pass — confirmed under HEAD: all green.

## Estimated LOC

- `is_definitely_dying` helper body: ~30 LOC.
- 4-line mass override in `compute()`: ~4 LOC.
- Doc comment on `is_definitely_dying`: ~5 LOC.
- **Total: ~35-40 LOC** in `src/heuristic_prior.rs`.

## Top 3 risks

1. **Speed ties go to opp (strict `<=` for opp).** `i_definitely_move_after = ... my_speed <= opp_speed`. At exact speed ties (50/50 in the actual game), we conservatively assume opp moves first. This matches Fix #1.5's strict-`>` symmetry and is the safer side for the heuristic ("if you might die, recommend switch"). Documented inline; explicit out-of-scope item below.
2. **Opp's actual move ≠ predicted top-damage move.** `predicted_opp_max_damage_and_priority` returns the priority of opp's max-damage move. If opp picks a status move (Toxic, Will-O-Wisp, Recover) instead, we wrongly trigger the dying-penalty. Mitigation: same approximation Fix #1 already lives with for `matchup_switch_pick`. Real opp-move prediction is a future plan. The downside is asymmetric: false-positive triggers a switch recommendation, which costs tempo but rarely loses the game vs a status move.
3. **Ability priority bumps not modeled.** Prankster (status +1), Gale Wings (Flying +1 if full HP), Triage (heal +3) on either side would change effective priority. Choice's `.priority` field is the static base; we don't compute effective priority. For our fix's purpose this is conservative on my side (we underestimate my priority → may over-trigger dying-penalty for Prankster users) and aggressive on opp side (we underestimate opp priority → may under-trigger). Acceptable; symmetric to Fix #1.5's open issue.

## Out of scope

- **Weather effects on damage calc** — separate parallel investigation.
- **Opp's actual move prediction** — uses top-damage move as proxy.
- **Ability priority bumps** — Prankster / Gale Wings / Triage.
- **Trick Room / paralyze full-skip / sleep / freeze** — `get_effective_speed` covers paralysis speed-cut but not full-skip RNG.
- **Multi-turn lookahead** — "I survive this turn, die next turn even after my move" is not modeled.
- **Tuning the override constants** — `(0.0, 0.9)` is the proposed initial choice. If A/B testing later shows over-correction, sweep `(mass_dmg, mass_switch) ∈ {(0.0, 0.9), (0.1, 0.8), (0.0, 0.95)}`.

## Ambiguities to resolve before implementation

1. **Override magnitude.** `(0.0, 0.9)` proposed (zero out dmg, dump 0.9 on switch). Alternative: `(mass_dmg * 0.3, mass_switch + mass_dmg * 0.7) = (0.18, 0.72)` per the brief's pseudocode. Recommend `(0.0, 0.9)` — the brief's variant still leaks 0.18 onto a single dying move slot, which is more than uniform. The whole point is to starve dying moves.
2. **Priority equality + speed equality.** Spec says "my_top_priority == opp_priority AND my_speed <= opp_speed → I move after". This treats speed ties as opp-first. Recommend keeping this conservative rule (matches Fix #1.5).
3. **Status-only active.** When the active has only status moves, `damage_calc_top_move` returns None and `have_damaging` is false. We early-return false from `is_definitely_dying` rather than triggering. Rationale: status-only is a different bug class (locked-into-Recover-while-dying); the user might genuinely want to click Recover. Out of scope here.

## Red-phase commit

Tests committed:
- File: `tests/test_death_aware_switch_priority.rs`
- Commit SHA: `e3d4f91`
- Commit message: `test(plan-i): failing tests for death-aware switch priority`
- Result: 1/4 fails under HEAD `e3d4f91` (bug-trigger); 3/4 pass (regression guards).
