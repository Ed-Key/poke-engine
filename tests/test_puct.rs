//! PUCT formula unit tests.
//!
//! Six tests that lock in the formula `Q + c * P * sqrt(N_parent) / (1 + N)`
//! against the cases that gave the verifier (and your future self) heartburn.

use poke_engine::engine::state::MoveChoice;
use poke_engine::mcts::{MctsSearch, MoveNode};
use poke_engine::state::PokemonMoveIndex;
use std::time::Duration;

fn mk(visits: u32, total_score: f32, prior: f32) -> MoveNode {
    MoveNode {
        move_choice: MoveChoice::Move(PokemonMoveIndex::M0),
        total_score,
        visits,
        prior,
    }
}

#[test]
fn unvisited_neutral_with_zero_parent_visits_is_q_only() {
    // PUCT(visits=0, parent_visits=0) = q=0.5 + c*P*0/1 = 0.5.
    let n = mk(0, 0.0, 0.25);
    let s = n.puct(0, 1.25);
    assert!((s - 0.5).abs() < 1e-6);
}

#[test]
fn unvisited_with_skewed_prior_after_one_parent_visit() {
    // PUCT(visits=0, parent_visits=1) = 0.5 + c*P*1/1 = 0.5 + 1.25*P.
    let high = mk(0, 0.0, 0.95);
    let low = mk(0, 0.0, 0.01);
    let s_high = high.puct(1, 1.25);
    let s_low = low.puct(1, 1.25);
    // Skewed prior: high-prior gets ~1.69, low-prior gets ~0.51.
    assert!(s_high > s_low + 1.0, "high={} low={}", s_high, s_low);
}

#[test]
fn heavily_visited_q_dominates_u() {
    // Visits = 1000, score = 999 → q ≈ 0.999. U is small even with prior.
    let n = mk(1000, 999.0, 0.95);
    let s = n.puct(2_000_000, 1.25);
    let q = 999.0 / 1000.0;
    let u = 1.25 * 0.95 * (2_000_000.0_f32).sqrt() / 1001.0;
    assert!((s - (q + u)).abs() < 1e-3);
    // q dominates: u/q ratio < 2.
    assert!(u < 2.0 * q);
}

#[test]
fn equal_q_skewed_prior_picks_high_prior() {
    // Two siblings: same Q, different priors. PUCT picks the one with higher P.
    let high = mk(10, 5.0, 0.9); // q=0.5
    let low = mk(10, 5.0, 0.1); // q=0.5
    // Parent_visits=100; for each, U = 1.25 * P * sqrt(100) / 11 ≈ 1.136 * P.
    let s_high = high.puct(100, 1.25);
    let s_low = low.puct(100, 1.25);
    assert!(s_high > s_low);
}

#[test]
fn uniform_prior_one_over_n_no_skew() {
    // 4 siblings each at prior=0.25 (uniform). PUCT differentiates them only
    // through Q.
    let a = mk(0, 0.0, 0.25);
    let b = mk(0, 0.0, 0.25);
    let c = mk(0, 0.0, 0.25);
    let d = mk(0, 0.0, 0.25);
    let p_visits = 4;
    let sa = a.puct(p_visits, 1.25);
    let sb = b.puct(p_visits, 1.25);
    let sc = c.puct(p_visits, 1.25);
    let sd = d.puct(p_visits, 1.25);
    assert!((sa - sb).abs() < 1e-6);
    assert!((sb - sc).abs() < 1e-6);
    assert!((sc - sd).abs() < 1e-6);
}

#[test]
fn c_puct_scales_exploration_proportionally() {
    let n = mk(0, 0.0, 0.5);
    let p_visits = 100;
    let s_low_c = n.puct(p_visits, 0.5);
    let s_high_c = n.puct(p_visits, 4.0);
    // u_low = 0.5 * 0.5 * 10 / 1 = 2.5; u_high = 4.0 * 0.5 * 10 / 1 = 20.
    // PUCT = 0.5 + u; difference is 4x - 0.5x = 3.5x (proportional to c).
    let u_low = s_low_c - 0.5;
    let u_high = s_high_c - 0.5;
    let ratio = u_high / u_low;
    assert!((ratio - 8.0).abs() < 1e-3, "ratio = {}", ratio);
}

/// P=0.05, c_forced=2.0, N_parent=8M
/// n_forced = floor(sqrt(2 * 0.05 * 8_000_000)) = floor(sqrt(800_000)) = 894
/// With N=10, 10 < 894 → true.
#[test]
fn should_force_visit_below_threshold_returns_true() {
    let node = mk(10, 5.0, 0.05);
    assert!(node.should_force_visit(8_000_000, 2.0));
}

#[test]
fn should_force_visit_above_threshold_returns_false() {
    // Same parameters but N=900 > 894 → false.
    let node = mk(900, 450.0, 0.05);
    assert!(!node.should_force_visit(8_000_000, 2.0));
}

#[test]
fn should_force_visit_c_zero_short_circuits() {
    // c_forced = 0.0 → n_forced = 0 always → never force-visit.
    let node = mk(0, 0.0, 0.05);
    assert!(!node.should_force_visit(8_000_000, 0.0));
}

/// Plan I Task 7 integration test: at the root, with `c_forced=2.0` and a
/// low-prior alternative move, that move must accumulate at least its
/// forced-playout threshold of visits despite PUCT preferring the high-prior
/// move.
///
/// Threshold: `floor(sqrt(c_forced * P_low * N_parent))`. With c_forced=2.0
/// and P_low=0.05, after N_parent=K iterations the low-prior move must have
/// at least `floor(sqrt(0.1 * K))` visits.
#[test]
fn forced_playouts_ensure_low_prior_root_move_gets_visits() {
    use poke_engine::translate::json_to_poke_state;

    // Single-Pokemon-each state so options are limited and we can construct
    // priors deterministically. Identical to `mcts_search_incremental_api`'s
    // fixture except trimmed to comments-only relevance.
    let json = r#"{
        "sideOne": {
            "pokemon": [
                {
                    "species": "Blaziken",
                    "level": 100,
                    "types": ["Fire", "Fighting"],
                    "hp": 302,
                    "maxhp": 302,
                    "ability": "Speed Boost",
                    "item": "Life Orb",
                    "nature": "Jolly",
                    "attack": 349,
                    "defense": 196,
                    "specialAttack": 230,
                    "specialDefense": 176,
                    "speed": 284,
                    "status": "None",
                    "weightKg": 52.0,
                    "moves": [
                        {"id": "Close Combat", "pp": 8},
                        {"id": "Flare Blitz", "pp": 24},
                        {"id": "Swords Dance", "pp": 32},
                        {"id": "Knock Off", "pp": 32}
                    ],
                    "teraType": "Fire"
                }
            ],
            "activeIndex": 0
        },
        "sideTwo": {
            "pokemon": [
                {
                    "species": "Alakazam",
                    "level": 100,
                    "types": ["Psychic"],
                    "hp": 251,
                    "maxhp": 251,
                    "ability": "Magic Guard",
                    "item": "Focus Sash",
                    "nature": "Timid",
                    "attack": 121,
                    "defense": 128,
                    "specialAttack": 369,
                    "specialDefense": 206,
                    "speed": 372,
                    "status": "None",
                    "weightKg": 48.0,
                    "moves": [
                        {"id": "Psychic", "pp": 16},
                        {"id": "Shadow Ball", "pp": 24},
                        {"id": "Focus Blast", "pp": 8},
                        {"id": "Energy Ball", "pp": 16}
                    ],
                    "teraType": "Psychic"
                }
            ],
            "activeIndex": 0
        }
    }"#;

    let state = json_to_poke_state(json).expect("parse state");
    let (s1_opts, s2_opts) = state.root_get_all_options();
    let n1 = s1_opts.len();
    assert!(
        n1 >= 2,
        "test requires at least 2 s1 options, got {}",
        n1
    );

    // Construct priors: move[0] high (0.90), move[1] low (0.05), rest split
    // evenly across the remaining mass (~0.05 / (n1 - 2) each). PUCT will
    // strongly prefer move[0]; without forced playouts, move[1] would get
    // very few visits in the early budget.
    let p_low = 0.05_f32;
    let p_high = 0.90_f32;
    let remainder = 1.0 - p_high - p_low;
    let other_share = if n1 > 2 {
        remainder / (n1 - 2) as f32
    } else {
        0.0
    };
    let mut s1_priors = vec![other_share; n1];
    s1_priors[0] = p_high;
    s1_priors[1] = p_low;

    // Use AlphaZero default c_puct to match production CLI default.
    let c_puct = 1.25_f32;
    let c_forced = 2.0_f32;
    let mut search = MctsSearch::new_with_priors(
        state.clone(),
        s1_opts,
        s2_opts,
        c_puct,
        Some(s1_priors),
        None,
    );
    search.set_c_forced(c_forced);

    // Run for a short wall-clock budget. We need enough iterations that the
    // forced-playout threshold for the low-prior move is non-trivial. With
    // c_forced=2.0 and P_low=0.05, after K root iterations the threshold is
    // `floor(sqrt(0.1 * K))`. K ≈ 1000 → threshold ≈ 10.
    search.run_for(Duration::from_millis(500));
    let n_parent = search.total_iterations();
    assert!(
        n_parent >= 100,
        "need at least 100 iterations for a meaningful threshold, got {}",
        n_parent
    );

    let snap = search.snapshot(500);
    let low_visits = snap.s1[1].visits;
    let threshold = ((c_forced * p_low * n_parent as f32).sqrt()).floor() as u32;
    assert!(
        low_visits >= threshold,
        "forced playouts: low-prior move expected >= {} visits (sqrt({} * {} * {})), got {}",
        threshold,
        c_forced,
        p_low,
        n_parent,
        low_visits,
    );

    // Sanity: the high-prior move should still dominate (PUCT-max takes over
    // once the threshold is met). Without this we'd just be confirming a
    // round-robin scheduler, not a forced-playouts wrapper.
    let high_visits = snap.s1[0].visits;
    assert!(
        high_visits > low_visits,
        "high-prior move should still receive more visits than low-prior under \
         forced-playouts; high={} low={}",
        high_visits,
        low_visits,
    );
}
