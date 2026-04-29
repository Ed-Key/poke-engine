//! Plan E end-to-end integration test: Iron Crown T5 smoking gun.
//!
//! Reconstructs the Iron Crown T5 state with Garchomp on `sideOne` (so
//! the engine's "side one optimizes for player" assumption holds).
//! Runs MCTS twice — heuristic vs NN-eval — and asserts that NN-eval
//! recommends Earthquake (regular or Tera variant), reproducing Phase 1's
//! 96.2% policy mass on EQ now via the FULL MCTS pipeline.
//!
//! Marked `#[ignore]` because it requires the sidecar to be running. To run:
//!
//!     # Terminal 1
//!     source ~/Projects/metamon-spike/metamon/.venv-py310/bin/activate
//!     python -m sidecar.nn_sidecar &
//!
//!     # Terminal 2
//!     cd ~/Projects/poke-engine
//!     cargo test --release --features terastallization \
//!         --test test_nn_mcts_integration -- --ignored --nocapture
//!
//! The heuristic-mode run is ALSO ignored (same gating) because its only
//! purpose is producing a side-by-side comparison for the assertion
//! against the NN-mode result.

use std::sync::Arc;
use std::time::{Duration, Instant};

use poke_engine::engine::state::MoveChoice;
use poke_engine::eval_kind::EvalKind;
use poke_engine::mcts::{MctsResult, MctsSearch, DEFAULT_C_PUCT};
use poke_engine::nn_client::NnClient;
use poke_engine::translate::auto_detect_and_parse;

/// Iron Crown T5 with Garchomp on sideOne (swap of the postmortem fixture).
const FIXTURE: &str = include_str!("fixtures/iron_crown_t5_garchomp_side1.json");

/// Time budget per search. Per the verifier's note on the "Iron Crown T5
/// acceptance" criterion: with deep convergence (~2.5M sims/sec) the
/// heuristic Q dominates the prior. We keep the budget short enough that
/// the prior still has visible influence, but long enough for stability.
const SEARCH_MS: u64 = 1500;

/// Common sidecar URL for the integration tests. Default sidecar binds 7273.
fn sidecar_url() -> String {
    std::env::var("POKE_ENGINE_NN_URL").unwrap_or_else(|_| "http://localhost:7273".to_string())
}

/// Run one MCTS search; return (best_move_name, full_snapshot, s1_move_names).
fn run_one(eval_kind: &EvalKind, c_puct: f32, label: &str) -> (String, MctsResult, Vec<String>) {
    let state = auto_detect_and_parse(FIXTURE).expect("parse Iron Crown T5 fixture");
    let (s1_options, s2_options) = state.root_get_all_options();
    assert!(!s1_options.is_empty(), "no legal moves for Garchomp");
    let s1_move_names: Vec<String> = s1_options
        .iter()
        .map(|mc| mc.to_string(&state.side_one))
        .collect();

    eprintln!("[{}] s1_options:", label);
    for (i, name) in s1_move_names.iter().enumerate() {
        eprintln!("    [{}] {} = {:?}", i, name, s1_options[i]);
    }

    let start = Instant::now();
    let mut search = MctsSearch::new_with_eval(
        state.clone(),
        s1_options.clone(),
        s2_options,
        eval_kind,
        c_puct,
    );
    search.run_for(Duration::from_millis(SEARCH_MS));
    let snap = search.snapshot(start.elapsed().as_millis() as u64);

    eprintln!(
        "[{}] {} iterations in {:.2}s",
        label,
        snap.iteration_count,
        start.elapsed().as_secs_f32()
    );
    let mut sorted: Vec<(usize, &str, u32, f32)> = snap
        .s1
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let avg = if r.visits > 0 {
                r.total_score / r.visits as f32
            } else {
                0.0
            };
            // Match move name back to s1_options order via MoveChoice equality.
            let name_idx = s1_options
                .iter()
                .position(|mc| mc == &r.move_choice)
                .unwrap_or(i);
            let name: &str = &s1_move_names[name_idx];
            (i, name, r.visits, avg)
        })
        .collect();
    sorted.sort_by(|a, b| b.2.cmp(&a.2));
    eprintln!("[{}] visit-distribution top-8:", label);
    for (rank, (_, name, visits, avg)) in sorted.iter().take(8).enumerate() {
        eprintln!("    {}: {:<25} visits={:<10} avg={:.4}", rank, name, visits, avg);
    }
    let (best_name, _) = sorted
        .iter()
        .map(|(_, n, _, _)| (n.to_string(), 0))
        .next()
        .expect("at least one move");
    (best_name, snap, s1_move_names)
}

#[test]
#[ignore]
fn iron_crown_t5_nn_picks_earthquake() {
    let state = auto_detect_and_parse(FIXTURE).expect("fixture parses");
    let (s1_opts, _) = state.root_get_all_options();
    let s1_names: Vec<String> = s1_opts
        .iter()
        .map(|mc| mc.to_string(&state.side_one))
        .collect();
    eprintln!("=== Iron Crown T5 ===");
    eprintln!("Garchomp's options ({}):", s1_opts.len());
    for n in &s1_names {
        eprintln!("    {}", n);
    }

    // 1) Heuristic baseline.
    let (best_heur, snap_heur, _) =
        run_one(&EvalKind::Heuristic, DEFAULT_C_PUCT, "heuristic");
    eprintln!("HEURISTIC best move: {}", best_heur);

    // 2) NN-prior mode. Build client now (test is gated on sidecar running).
    let nn_client = Arc::new(NnClient::new(sidecar_url(), Duration::from_secs(15)));
    nn_client
        .healthz()
        .expect("sidecar must be running for this test");
    let eval_kind = EvalKind::Nn(nn_client.clone());
    // Use AlphaZero's c_puct=1.25 in the NN-mode test (matches spec default).
    let (best_nn, snap_nn, _) = run_one(&eval_kind, 1.25, "nn");
    eprintln!("NN-PRIOR best move: {}", best_nn);

    // Assertion 1: NN mode picks an EQ variant (regular or Tera).
    let nn_lc = best_nn.to_lowercase();
    assert!(
        nn_lc.contains("earthquake"),
        "NN-mode must pick an EQ variant, got: {} (heur picked: {})",
        best_nn,
        best_heur
    );

    // Assertion 2: SR is the audit's incorrect baseline; check the NN visit
    // distribution actually skews away from it.
    let sr_visits: u32 = snap_nn
        .s1
        .iter()
        .filter(|r| matches!(
            r.move_choice,
            MoveChoice::Move(_) | MoveChoice::MoveTera(_)
        ))
        .filter(|r| {
            let name = r.move_choice.to_string(&auto_detect_and_parse(FIXTURE).unwrap().side_one);
            name.to_lowercase().contains("stealthrock")
        })
        .map(|r| r.visits)
        .sum();
    let eq_visits: u32 = snap_nn
        .s1
        .iter()
        .filter(|r| matches!(
            r.move_choice,
            MoveChoice::Move(_) | MoveChoice::MoveTera(_)
        ))
        .filter(|r| {
            let name = r.move_choice.to_string(&auto_detect_and_parse(FIXTURE).unwrap().side_one);
            name.to_lowercase().contains("earthquake")
        })
        .map(|r| r.visits)
        .sum();
    eprintln!(
        "NN visit shares: EQ={} SR={} ratio={:.2}x",
        eq_visits,
        sr_visits,
        eq_visits as f32 / sr_visits.max(1) as f32
    );
    assert!(
        eq_visits > sr_visits * 5,
        "EQ visits ({}) should dwarf SR visits ({}) under NN priors",
        eq_visits,
        sr_visits
    );

    // Assertion 3: ratio of NN-mode's iteration count to heuristic's should
    // be of similar order. (NN adds one HTTP call at root, ~19ms; SEARCH_MS
    // is 1500ms so the overhead is small.)
    let iter_ratio = snap_nn.iteration_count as f32 / snap_heur.iteration_count as f32;
    eprintln!(
        "iter counts: heur={}  nn={}  ratio={:.3}x",
        snap_heur.iteration_count, snap_nn.iteration_count, iter_ratio
    );
    assert!(
        iter_ratio > 0.5 && iter_ratio < 2.0,
        "NN search throughput should not collapse: heur={} nn={}",
        snap_heur.iteration_count,
        snap_nn.iteration_count
    );

    eprintln!("=== Iron Crown T5 PASSED ===");
}
