# Plan I — Fix #1.5: Priority-aware damage_calc_top_move

**Date:** 2026-04-30
**Worktree:** `/Users/edkiboma/Projects/poke-engine/.worktrees/plan-i`
**Branch:** `feat/plan-i-prior-dampening`
**Target file:** `src/heuristic_prior.rs` (only)

## Goal (one sentence)

Make `damage_calc_top_move` zero-out the effective damage of any move that does NOT actually fire (slower attacker dies to opp's predicted hit before its own move resolves), so the heuristic prefers priority/connecting moves like Extreme Speed when the alternative is a high-damage move that gets KO'd before it fires (battle 2598333175 vs Darkssz, T9: Dragonite mirror, opp at +1 spe → engine wrongly picked Ice Spinner over Extreme Speed).

## Approach

Pre-compute three turn-order scalars once, then per-move filter:

1. `predicted_opp_max_damage` — opp's max-roll damage to MY current active. Reuses `predicted_opp_max_damage_against` (added in Fix #1) with `candidate_idx = my_side.active_index`.
2. `opp_predicted_priority` — priority of opp's max-damage move. Computed by extending `predicted_opp_max_damage_against` to return `(damage, priority)` (or a sibling helper `predicted_opp_top_priority`). The helper extension is cleaner: one state-clone, one move-iteration, two return values.
3. `my_speed`, `opp_speed` — via `crate::engine::generate_instructions::get_effective_speed(state, &side_ref)`.

Per damaging move, decide `i_move_first` from priority then speed; if `i_move_first || i_survive`, the move fires — score it normally. Otherwise score it as 0 (effectively excluded; if all moves score 0, fall back as today via `candidates.is_empty()` check).

### Modified pseudocode

```rust
pub fn damage_calc_top_move(state: &State, perspective: SidePerspective) -> Option<Choices> {
    let (attacking_side, my_active, my_side_ref, opp_side_ref, my_active_idx) = match perspective {
        SidePerspective::Side1 => (
            SideReference::SideOne, state.side_one.get_active_immutable(),
            SideReference::SideOne, SideReference::SideTwo,
            state.side_one.active_index,
        ),
        SidePerspective::Side2 => (
            SideReference::SideTwo, state.side_two.get_active_immutable(),
            SideReference::SideTwo, SideReference::SideOne,
            state.side_two.active_index,
        ),
    };

    // Survivability: opp's max damage and priority of that move.
    let (predicted_opp_dmg, opp_priority) =
        predicted_opp_max_damage_and_priority(state, my_active_idx, perspective);
    let i_survive = (my_active.hp - predicted_opp_dmg) > 0;

    let my_speed = get_effective_speed(state, &my_side_ref);
    let opp_speed = get_effective_speed(state, &opp_side_ref);

    let mut candidates: Vec<(Choices, i16, f32)> = Vec::new();
    for mv in my_active.moves.into_iter() {
        if mv.disabled || mv.pp <= 0 { continue; }
        let choice = &mv.choice;
        if choice.category == MoveCategory::Status || choice.category == MoveCategory::Switch {
            continue;
        }
        if let Some((max_dmg, _crit)) = calculate_damage(state, &attacking_side, choice, DamageRolls::Max) {
            if max_dmg <= 0 { continue; }

            let i_move_first = if choice.priority > opp_priority {
                true
            } else if choice.priority < opp_priority {
                false
            } else {
                my_speed > opp_speed
            };
            if !(i_move_first || i_survive) { continue; }      // move never fires

            let acc = if choice.accuracy < 0.0 { 100.0 } else { choice.accuracy };
            let score = choice.base_power * (acc / 100.0);
            candidates.push((mv.id, max_dmg, score));
        }
    }
    // ... existing tie-break + return ...
}
```

### Helpers to add / extend (private to `heuristic_prior.rs`)

- **Extend** `predicted_opp_max_damage_against` → rename internally to a `predicted_opp_max_damage_and_priority(state, candidate_idx, perspective) -> (i16, i8)`, and keep a thin shim `predicted_opp_max_damage_against` that returns `.0` so Fix #1's `matchup_switch_pick` keeps working unchanged. Tracks the priority of whichever opp move is the current `max_dmg`.
- **Use** `get_effective_speed` from `crate::engine::generate_instructions` (already `pub(crate)`, importable from `heuristic_prior.rs` since it's in the same crate; see `src/genx/evaluate.rs:6` for precedent).

## Files to modify

- `src/heuristic_prior.rs` — modify `damage_calc_top_move` body, extend the predicted-opp helper to also surface priority. **No other files.**

## TDD test list

File: `tests/test_priority_aware_damage.rs` (committed in red phase).

1. `priority_move_chosen_when_slower_attacker_would_die` — Mariga's Dragonite 104/324 HP, opp Dragonite at +1 atk +1 spe (Dragon Danced). Mariga's moves: Ice Spinner / Extreme Speed / Earthquake / Dragon Claw. Asserts `Some(EXTREMESPEED)`. **Fails under HEAD** (returns ICESPINNER).
2. `priority_aware_picks_highest_damage_when_attacker_outspeeds` — Mariga Danced (+1 spe), opp unboosted. Asserts `Some(ICESPINNER)`. Regression guard.
3. `priority_aware_picks_highest_damage_when_slower_but_survives` — Mariga at full HP & bulky, opp at +1 spe but base attack. Mariga is slower yet survives Dragon Claw. Asserts `Some(ICESPINNER)`. Regression guard (clarifies test #3 in the brief: surviving means Ice Spinner DOES fire, just second).
4. `priority_aware_returns_none_when_only_damaging_move_is_immune` — Garchomp w/ Earthquake + Swords Dance vs Zapdos (Electric/Flying). Ground 0× into Flying. Asserts `None`.
5. `priority_aware_preserves_garchomp_vs_heatran_eq_pick` — full Garchomp vs Heatran fixture from the existing suite, with explicit speed (239 vs 166). Garchomp outspeeds, EQ fires, must still pick `EARTHQUAKE`. Regression guard.

Existing tests in `tests/test_heuristic_prior.rs` (`damage_calc_top_move_picks_super_effective_ko`, `damage_calc_top_move_tiebreaks_with_base_power_x_acc`, `damage_calc_top_move_returns_none_when_only_status_moves`) and all `tests/test_survivability_switch.rs` tests must continue to pass — they use full-HP fixtures or status-only fixtures where priority logic is a no-op.

Red-phase verification:
- 1 of 5 new tests fails under current HEAD `f9d3bb8` (the bug-trigger test).
- 4 of 5 pass — these are the regression guards. They guarantee the green-phase fix doesn't break the cases where the heuristic is already correct (outspeed → top damage, slower-but-bulky → top damage, immunity → None, baseline Garchomp/Heatran).

## Estimated LOC

- Extend `predicted_opp_max_damage_and_priority` body: ~5 LOC (track priority alongside max_dmg).
- Shim/rename existing `predicted_opp_max_damage_against` caller in `matchup_switch_pick`: ~2 LOC.
- New imports (`SideReference` already in scope; add `crate::engine::generate_instructions::get_effective_speed`): ~1 LOC.
- Refactor `damage_calc_top_move` body (precompute scalars + per-move `i_move_first`/`can_i_fire` filter): ~25 LOC.
- **Total: ~33 LOC** in `src/heuristic_prior.rs`.

## Top 3 risks

1. **Speed-tie / paralyze / Trick Room ignored.** `get_effective_speed` already accounts for boosts, items (Choice Scarf), abilities (Chlorophyll, Quick Feet), weather, and paralysis — but the tiebreak in `i_move_first` uses strict `>`. Speed ties default to opp-moves-first, which is conservative (will reject high-damage moves at exact speed ties even though half the time they fire). Acceptable for a heuristic; documented in the body comment. Trick Room would invert the comparison — explicitly out of scope.
2. **Opp's "predicted top move" isn't necessarily what they pick.** We use opp's max-roll damage move via `predicted_opp_max_damage_and_priority`. If opp picks a different move (e.g., a status setup), our `predicted_opp_dmg` is overstated → we overcorrect toward priority moves. Mitigation: this is the same approximation Fix #1 already lives with for `matchup_switch_pick`; we're consistent. Real opp-move prediction is a future plan.
3. **Helper extension API churn.** Renaming `predicted_opp_max_damage_against` to return a tuple breaks the call site in `matchup_switch_pick`. Mitigation: keep the old name as a shim that returns `.0` of the new tuple; zero behavior change for Fix #1.

## Out of scope

- **Opp's actual move prediction** — we use `predicted_opp_max_damage_and_priority` (top damage + that move's priority) as a proxy.
- **Opp ability priority bumps** — Prankster (status moves +1 prio), Gale Wings (Flying moves +1 if full HP), Triage (healing +3) would change opp's effective priority. The Choice's `priority` field doesn't reflect these. Deferred; would require a `compute_effective_priority(choice, attacker)` helper.
- **My ability priority bumps** — same deferred for symmetry.
- **Multi-turn survival** — "I survive this turn but die next turn even after my move" is not modeled. We make a one-turn fire/no-fire decision.
- **Speed ties** — strict `>` means ties go to opp; not modeling the 50/50 random resolution.
- **Status conditions affecting fire order** (paralysis full-turn skip, sleep, freeze) — `get_effective_speed` covers paralysis speed-cut but not "this Pokemon won't move at all". Treat as out of scope for the heuristic.

## Ambiguities to resolve before implementation

1. **Helper naming** — extend existing `predicted_opp_max_damage_against` to a `_and_priority` variant + shim, or add a sibling `predicted_opp_top_priority` that re-walks the moves? Recommend the tuple-return + shim (one walk, one clone).
2. **Speed-tie behavior** — strict `>` (current proposal) or coin-flip? Recommend strict `>`; matches the conservative spirit of "would this move actually fire?"
3. **Trick Room** — assert `state.trick_room.is_active() == false` and skip priority-aware logic entirely if active? Recommend ignore for now (TR is rare in singles ladder, and the heuristic is just a prior — MCTS can correct).

## Red-phase commit

Tests committed in next step:
- File: `tests/test_priority_aware_damage.rs`
- Commit message: `test(plan-i): failing tests for priority-aware damage heuristic`
- 1/5 tests fails under HEAD `f9d3bb8` (the core bug-trigger); 4/5 are regression guards that will continue to pass post-fix.
