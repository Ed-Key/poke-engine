# Plan I — Fix #1: Survivability-aware switch heuristic

**Date:** 2026-04-30
**Worktree:** `/Users/edkiboma/Projects/poke-engine/.worktrees/plan-i`
**Branch:** `feat/plan-i-prior-dampening`
**Target file:** `src/heuristic_prior.rs` (only)

## Goal (one sentence)

Make `matchup_switch_pick` reject bench Pokemon that would die on entry (hazards) or to the opp's predicted top-damage move, then break type-matchup ties by remaining HP, so the heuristic stops recommending Urshifu-style "good matchup but already half-dead" switches.

## Approach: **Option C (hybrid)** — pre-filter then survivability-weighted score

**Justification.** Pure pre-filter (B) is brittle: damage rolls are uncertain, and a candidate that survives at the min roll but dies at the max should still be discouraged, not silently dropped. Pure penalty term (A) is hard to scale: the type-eff score has range roughly `[-12, +12]` (3 moves × 4× each side) and naive penalties either dominate trivially or get drowned out. Hybrid: hard-eliminate "definitely dead" candidates (hazards alone OHKO, or opp max-roll OHKOs), then for survivors compute `score = type_eff_diff − survivability_term` where the term is tiny (hp_fraction) and only acts as a tiebreak.

### Modified pseudocode

```rust
fn matchup_switch_pick(state, perspective) -> Option<PokemonName> {
    if my_side.force_trapped { return None; }
    let mut viable = Vec::new();

    for pkmn in bench_alive_non_active {
        let hp_after_hazards = pkmn.hp - hazard_damage_on_entry(pkmn, my_side);
        if hp_after_hazards <= 0 { continue; }                       // pre-filter (a)

        let opp_top_dmg = predicted_opp_max_damage_against(pkmn);    // pre-filter (b)
        if hp_after_hazards - opp_top_dmg <= 0 { continue; }

        let score = matchup_score_against(opp, pkmn)
                  - 0.5 * (hp_after_hazards as f32 / pkmn.maxhp as f32);
        viable.push((pkmn.id, score));
    }

    viable.into_iter().min_by(score).map(|(name, _)| name)
}
```

### Helpers to add (private to `heuristic_prior.rs`)

1. `hazard_damage_on_entry(pkmn: &Pokemon, my_side: &Side) -> i16`
   - Mirrors the logic in `genx/generate_instructions.rs:441-475`: respects `HEAVYDUTYBOOTS`, `MAGICGUARD`, `is_grounded()`, applies SR `maxhp/8 × Rock_eff`, spikes `maxhp × layers / 8`.
2. `predicted_opp_max_damage_against(state, perspective, candidate_idx) -> i16`
   - Clones state, sets `my_side.active_index = candidate_idx`, walks opp's legal damaging moves with `calculate_damage(_, _, _, DamageRolls::Max)`, returns max. Reuses the existing helper machinery from `damage_calc_top_move`.

## Files to modify

- `src/heuristic_prior.rs` — modify `matchup_switch_pick`, add two private helpers.

(No other files. `nn_state_encoder.rs` is owned by Fix #4's planner; `generate_instructions.rs` is read-only this session.)

## TDD test list

File: `tests/test_survivability_switch.rs` (committed in red phase).

1. `switch_picks_higher_hp_among_equal_matchups`
   FIRE/STEEL Heatran (19/385 HP) vs FIRE/STEEL Dialga (385/385 HP), opp Volcarona Bug Buzz. Tie on type-eff sum; Heatran dies to Bug Buzz (hp too low), Dialga survives. Assert `Some(DIALGA)`.

2. `switch_avoids_low_hp_into_spikes`
   3 spikes layers; Toxapex (50/300, grounded) has best matchup vs Volcarona but eats 75 spikes damage and dies; Dragonite (350/350, Flying) survives. Assert `Some(DRAGONITE)`.

3. `switch_avoids_dying_to_predicted_opp_move`
   Lando-T at 30/319 HP has best type matchup vs opp Heatran (Ground 4× vs Heatran), but Heatran's Magma Storm OHKOs at 30 HP. Toxapex full HP survives. Assert `Some(TOXAPEX)`.

4. `switch_returns_none_when_no_viable_candidate`
   3 spikes + every bench Toxapex at 30/300 HP (all grounded, all die to 75 spikes dmg). Assert `None`.

5. `switch_still_picks_best_matchup_when_all_viable` (regression guard)
   Heatran + Toxapex both full HP, no hazards, opp Volcarona Bug Buzz. Heatran has better matchup. Assert `Some(HEATRAN)` — current behavior preserved when survivability is not a factor.

Existing tests in `tests/test_heuristic_prior.rs` (`matchup_switch_picks_best_resist_profile`, `matchup_switch_returns_none_when_force_trapped`, `matchup_switch_returns_none_when_last_alive`, `compute_returns_correct_mass_distribution`) must continue to pass — they use full-HP fixtures where survivability is a no-op.

Red-phase verification:
- 4 of 5 new tests fail under current `e2694ed` HEAD.
- 1 test (`switch_still_picks_best_matchup_when_all_viable`) passes — that is the regression guard, by design.

## Estimated LOC

- `hazard_damage_on_entry` helper: ~20 LOC
- `predicted_opp_max_damage_against` helper: ~20 LOC
- Refactor of `matchup_switch_pick` body (filter + scoring): ~15 LOC
- **Total: ~55 LOC** in `src/heuristic_prior.rs`.

## Top 3 risks

1. **State clone cost in hot path.** `matchup_switch_pick` runs at every NN-prior call. Cloning `State` for each bench candidate (up to 5) in `predicted_opp_max_damage_against` could be measurable. Mitigation: clone the side once, mutate `active_index` in place, restore on drop via a guard. Or skip prediction for candidates already eliminated by hazards. If profiling shows >1ms regression, fall back to a cheap approximation: `opp_top_move.base_power × type_eff(opp_move_type, candidate) × 0.5` (skips stats, accuracy, items).

2. **HP tiebreak weight (`0.5 × hp_fraction`) is hand-tuned.** Type-eff sums range roughly `[-12, +12]`, so `0.5` ensures HP only breaks ties within ~0.5 of equal — but it could swing real matchups. Mitigation: confirm the existing `matchup_switch_picks_best_resist_profile` (gap of −3.75 vs −1.5 = 2.25) is unaffected, and add a compute-test asserting the gap on the regression guard fixture is preserved.

3. **`predicted_opp_max_damage_against` may interact unexpectedly with first-turn-only flags or volatile statuses.** The cloned state preserves `last_used_move`, `protect_used`, etc., which may make moves look unavailable that are actually fine (or vice versa). Mitigation: use `DamageRolls::Max` and only iterate `category != Status/Switch && pp > 0 && !disabled` — the same filter already used in `damage_calc_top_move`. Don't try to simulate any pre-move state changes.

## Out of scope

- **Predicting opp's actual move.** We use opp's current top-damage move via `DamageRolls::Max` as a proxy. Real prediction would require a recursive heuristic (their best response to our switch); that's a Plan I+1 feature.
- **Status (toxic counter, sleep turns) in survivability.** The bug report mentions status; for now only HP + hazards + predicted hit are gated. Adding toxic-counter awareness is ~5 extra LOC and can ride on this commit if simple, otherwise deferred.
- **Multi-turn survival.** "Will the switch-in survive 2 turns" is out of scope; we score one-turn entry only.
- **Switch-in immunities/abilities (Levitate, Volt Absorb, etc.).** Already handled implicitly via `type_effectiveness_modifier` in the matchup score; not re-evaluated here. `Magic Guard` and `Heavy-Duty Boots` ARE checked in the hazard helper (matching the engine).

## Ambiguities to resolve before implementation

1. **Toxic counter** — include in survivability or defer? Recommend defer; mention in commit message.
2. **`hp_fraction` weight constant** — 0.5 is the proposed default. Open question whether to make this configurable via `mass_dmg`/`mass_switch`-style param. Recommend hardcode for now; revisit if tournament data shows it's mis-tuned.
3. **Should `predicted_opp_max_damage_against` consider opp's choice-locked move when `choice_band` etc. are detectable?** Out of scope per the brief (use `damage_calc_top_move`-style top damage).

## Red-phase commit

Tests committed in next step:
- File: `tests/test_survivability_switch.rs`
- Commit message: `test(plan-i): failing tests for survivability-aware switch`
- 4/5 tests fail under HEAD `e2694ed`; 1 passes as regression guard.
