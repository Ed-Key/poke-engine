use axum::{
    body::Body,
    extract::{Json, State as AxumState},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
use poke_engine::engine::evaluate::{evaluate_breakdown, EvalBreakdown};
use poke_engine::eval_kind::EvalKind;
use poke_engine::mcts::{MctsResult, MctsSearch};
use poke_engine::nn_client::NnClient;
use poke_engine::translate::auto_detect_and_parse;
use serde::Serialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::cors::{Any, CorsLayer};

const DEFAULT_PORT: u16 = 7267;
const DEFAULT_TIME_LIMIT_MS: u64 = 5000;
const DEFAULT_UPDATE_INTERVAL_MS: u64 = 250;

/// CLI args for the poke-engine MCTS server.
///
/// All flags fall back to a matching environment variable so that the existing
/// `PORT=7267 cargo run --bin server` invocations keep working unchanged.
/// New flags introduced by Plan E Phase 4-5:
///   - `--nn-eval`           POKE_ENGINE_NN_EVAL   (bool)
///   - `--nn-url`            POKE_ENGINE_NN_URL    (default http://localhost:7273)
///   - `--nn-timeout-ms`     POKE_ENGINE_NN_TIMEOUT_MS (default 2000)
///   - `--c-puct`            POKE_ENGINE_C_PUCT    (default 1.25)
#[derive(Parser, Debug, Clone)]
#[command(name = "poke-engine-server")]
#[command(about = "poke-engine MCTS server (axum + tokio)")]
pub struct Cli {
    /// TCP port to bind. (Was previously read directly from $PORT.)
    #[arg(long, env = "PORT", default_value_t = DEFAULT_PORT)]
    pub port: u16,

    /// Enable Plan E NN-prior evaluation. Requires the metamon sidecar to be
    /// running at `--nn-url`. When unset, behavior is identical to the
    /// pre-Plan-E engine (heuristic-only).
    #[arg(long, env = "POKE_ENGINE_NN_EVAL", default_value_t = false)]
    pub nn_eval: bool,

    /// Base URL of the metamon sidecar. Only consulted when `--nn-eval` is set.
    #[arg(long, env = "POKE_ENGINE_NN_URL", default_value = "http://localhost:7273")]
    pub nn_url: String,

    /// Sidecar request timeout in milliseconds.
    #[arg(long, env = "POKE_ENGINE_NN_TIMEOUT_MS", default_value_t = 2000)]
    pub nn_timeout_ms: u64,

    /// PUCT exploration constant. AlphaZero default is 1.25.
    #[arg(long, env = "POKE_ENGINE_C_PUCT", default_value_t = 1.25)]
    pub c_puct: f32,

    /// Plan I: heuristic prior mix weight (`λ`). 0.0 = pure NN policy
    /// (default; current production behavior). 0.1 = recommended after A/B.
    #[arg(long, env = "POKE_ENGINE_HEURISTIC_MIX", default_value_t = 0.0)]
    pub heuristic_prior_mix: f32,

    /// Plan I: KataGo Forced Playouts constant. 0.0 = disabled (default).
    /// 2.0 = KataGo's published value.
    #[arg(long, env = "POKE_ENGINE_FORCED_C", default_value_t = 0.0)]
    pub forced_playouts_c: f32,

    /// Plan I Side2 extension: heuristic prior mix for opponent. 0.0 = uniform
    /// (Plan I behavior); 0.5 = balanced. Side2 has no NN so uniform is the
    /// blend baseline. Mirrors `--heuristic-prior-mix` (Side1).
    #[arg(long, env = "POKE_ENGINE_HEURISTIC_MIX_SIDE2", default_value_t = 0.0)]
    pub heuristic_prior_mix_side2: f32,

    /// Plan I Side2 extension: forced-playouts c-constant for opponent. 0.0
    /// = no-op. Mirrors `--forced-playouts-c` (Side1).
    #[arg(long, env = "POKE_ENGINE_FORCED_C_SIDE2", default_value_t = 0.0)]
    pub forced_playouts_c_side2: f32,

    /// Plan I: heuristic prior mass on damage-calc top move slot.
    #[arg(long, default_value_t = 0.6)]
    pub heuristic_prior_mass_dmg: f32,

    /// Plan I: heuristic prior mass on matchup-switch slot.
    #[arg(long, default_value_t = 0.3)]
    pub heuristic_prior_mass_switch: f32,

    /// engine-prior-tuning: cap each per-action NN prior at this value before
    /// renormalize. `1.0` (default) is a no-op (bit-identical pre-branch).
    /// `0.5` is a recommended starting point — clips runaway top-1 priors
    /// (currently >=0.95 on ~42% of records) and redistributes the excess
    /// mass uniformly across the rest of the action set.
    #[arg(long, env = "POKE_ENGINE_PRIOR_CAP", default_value_t = 1.0)]
    pub prior_cap: f32,

    /// engine-prior-tuning: Dirichlet noise concentration α mixed into the
    /// ROOT priors (matching AlphaZero). `0.0` (default) disables.
    /// AlphaZero-chess uses `0.3`; for Pokemon's smaller action set try
    /// `0.3`-`0.5`. Has no effect when `dirichlet_eps == 0.0`.
    #[arg(long, env = "POKE_ENGINE_DIRICHLET_ALPHA", default_value_t = 0.0)]
    pub dirichlet_alpha: f32,

    /// engine-prior-tuning: fraction of Dirichlet noise blended with root
    /// priors: `prior' = (1-eps)*prior + eps*dirichlet`. `0.0` (default)
    /// disables. AlphaZero default `0.25`.
    #[arg(long, env = "POKE_ENGINE_DIRICHLET_EPS", default_value_t = 0.0)]
    pub dirichlet_eps: f32,

    /// engine-prior-tuning: slope of the leaf-eval sigmoid. Default `0.0125`
    /// matches pre-branch behavior (saturates at ~±200). Reduce to ~`0.005`
    /// to give MCTS more dynamic range over evaluate()'s ~[-300, +300]
    /// output. WARNING: changes the leaf-value scale; tests with hard-coded
    /// expectations may need updating when this is non-default.
    #[arg(long, env = "POKE_ENGINE_EVAL_SLOPE", default_value_t = 0.0125)]
    pub eval_slope: f32,
}

/// Shared per-process state plumbed through axum handlers.
///
/// Cheap to clone (Arc-wrapped client). Created once at startup; passed by
/// value into the Router via `with_state`.
#[derive(Clone)]
pub struct AppState {
    pub eval_kind: EvalKind,
    pub c_puct: f32,
    /// Plan I: heuristic prior mix weight (`λ`). 0.0 → bit-identical to pre-Plan-I.
    pub heuristic_prior_mix: f32,
    /// Plan I: KataGo Forced Playouts constant. 0.0 → disabled.
    pub forced_playouts_c: f32,
    pub heuristic_prior_mix_side2: f32,
    pub forced_playouts_c_side2: f32,
    /// Plan I: heuristic prior mass on damage-calc top move slot.
    pub heuristic_prior_mass_dmg: f32,
    /// Plan I: heuristic prior mass on matchup-switch slot.
    pub heuristic_prior_mass_switch: f32,
}

// -- Request / Response types --


#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalyzeResponse {
    best_move: String,
    confidence: f32,
    simulations: u32,
    depth: u32,
    time_ms: u64,
    reasoning: Vec<ReasoningStep>,
    alternatives: Vec<Alternative>,
}

#[derive(Serialize)]
struct ReasoningStep {
    turn: usize,
    you: String,
    them: String,
}

#[derive(Serialize)]
struct Alternative {
    #[serde(rename = "move")]
    move_name: String,
    confidence: f32,
    note: String,
}

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    port: u16,
    engine: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// If the request body has a top-level "hypotheses" array, return the parsed
/// list of per-hypothesis JSON strings; otherwise return None.
///
/// We keep each hypothesis as a String (rather than serde_json::Value) so the
/// existing `auto_detect_and_parse(&str)` path stays unchanged.
fn extract_hypotheses(body: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let arr = v.get("hypotheses")?.as_array()?;
    if arr.is_empty() {
        return None;
    }
    Some(arr.iter().map(|h| h.to_string()).collect())
}

/// Spawn one PIMC worker thread. Each parses its own hypothesis JSON and runs
/// its own MctsSearch end-to-end. Returns a JoinHandle wrapping Result<MctsResult, String>.
///
/// Plan E: each worker calls the NN sidecar at the root for ITS hypothesis,
/// using the shared `eval_kind`. This is per the verifier's R-MISSING-1 note:
/// PIMC × NN serializes through the sidecar's GIL anyway, so K parallel
/// workers each issuing one /policy call is fine for K up to ~8 — the wall
/// clock is K × ~19ms inside the sidecar, parallelizable as far as the GIL
/// allows.
fn handles_spawn(
    hypothesis_json: String,
    budget_ms: u64,
    _seed: u64,
    eval_kind: EvalKind,
    c_puct: f32,
) -> std::thread::JoinHandle<Result<poke_engine::mcts::MctsResult, String>> {
    use std::time::{Duration, Instant};

    std::thread::spawn(move || {
        let state = auto_detect_and_parse(&hypothesis_json)
            .map_err(|e| format!("State parse error: {}", e))?;
        let (s1_options, s2_options) = state.root_get_all_options();
        if s1_options.is_empty() {
            return Err("No legal moves for side one".to_string());
        }
        let mut search = MctsSearch::new_with_eval(
            state.clone(),
            s1_options,
            s2_options,
            &eval_kind,
            c_puct,
        );
        let start = Instant::now();
        search.run_for(Duration::from_millis(budget_ms));
        Ok(search.snapshot(start.elapsed().as_millis() as u64))
    })
}

// Streaming NDJSON event emitted by /analyze/stream.
// Field names MUST stay in sync with the Python EngineClient consumer at
// /Users/edkiboma/Projects/showdown-copilot/src/showdown_copilot/engine_client.py
#[derive(Serialize)]
struct StreamUpdate {
    event: &'static str,
    #[serde(rename = "bestMove")]
    best_move: String,
    confidence: f32,
    sims: u32,
    depth: u32,
    pv: Vec<String>,
    alternatives: Vec<Alternative>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

// -- Handlers --

async fn status_handler() -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        port: DEFAULT_PORT,
        engine: "poke-engine-gen9".to_string(),
    })
}

async fn analyze_handler(
    AxumState(app): AxumState<AppState>,
    body: String,
) -> Result<Json<AnalyzeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Extract timeLimit if present in the JSON
    let parsed_top = serde_json::from_str::<serde_json::Value>(&body).ok();
    let time_limit_ms = parsed_top
        .as_ref()
        .and_then(|v| v.get("timeLimit")?.as_u64())
        .unwrap_or(DEFAULT_TIME_LIMIT_MS);
    // Engine-log correlation keys forwarded by the proxy (apply_belief).
    // Both are optional — direct-to-engine callers (e.g. Cobblemon mod)
    // omit them and the [ENGINE-INSTRUMENT] payload renders `null`.
    let battle_id: Option<String> = parsed_top
        .as_ref()
        .and_then(|v| v.get("battleId").and_then(|x| x.as_str()))
        .map(String::from);
    let turn: Option<u32> = parsed_top
        .as_ref()
        .and_then(|v| v.get("turn").and_then(|x| x.as_u64()))
        .map(|u| u as u32);
    // engine-seed-plumbing: optional deterministic RNG seed. When present,
    // the search becomes reproducible — same input + same seed yields a
    // bit-identical bestMove. Absent (the production default) preserves
    // pre-branch behavior using the thread-local RNG.
    let seed: Option<u64> = parsed_top
        .as_ref()
        .and_then(|v| v.get("seed").and_then(|x| x.as_u64()));

    let raw_json = body;
    let eval_kind = app.eval_kind.clone();
    let c_puct = app.c_puct;
    // Plan I: capture mix/mass/forced-playouts knobs into the closure.
    // All four are f32 (Copy), so this is a cheap by-value capture.
    let heuristic_prior_mix = app.heuristic_prior_mix;
    let forced_playouts_c = app.forced_playouts_c;
    let heuristic_prior_mass_dmg = app.heuristic_prior_mass_dmg;
    let heuristic_prior_mass_switch = app.heuristic_prior_mass_switch;
    // Plan I Side2 (Bug #3 fix): symmetric heuristic prior on the opponent
    // perspective. Default 0.0 keeps pre-fix behavior bit-identical.
    // `forced_playouts_c_side2` (T5) wires the per-side forced-playouts
    // c-constant on the opp dimension via `set_c_forced_side2`.
    let heuristic_prior_mix_side2 = app.heuristic_prior_mix_side2;
    let forced_playouts_c_side2 = app.forced_playouts_c_side2;

    // Translate to poke-engine State — catch panics from deserialization.
    // NN client calls happen inside this spawn_blocking thread, NEVER from
    // the surrounding async context (would stall the executor).
    let result = tokio::task::spawn_blocking(move || {
        // Translate JSON -> State (auto-detects format)
        let state = auto_detect_and_parse(&raw_json)
            .map_err(|e| format!("State parse error: {}", e))?;

        // Get legal options (root includes tera/mega)
        let (s1_options, s2_options) = state.root_get_all_options();

        if s1_options.is_empty() {
            return Err("No legal moves for side one".to_string());
        }

        // Snapshot side_one for move name resolution
        let s1_move_names: Vec<String> = s1_options
            .iter()
            .map(|mc| mc.to_string(&state.side_one))
            .collect();
        // Plan I (Task 9 telemetry): also snapshot s2 move names + the
        // pre-search eval breakdown for the [ENGINE-INSTRUMENT] line. State
        // is moved into MctsSearch below, so capture these now. We also keep
        // a clone of state and s1_options for later prior-mapping.
        let s2_move_names: Vec<String> = s2_options
            .iter()
            .map(|mc| mc.to_string(&state.side_two))
            .collect();
        let pre_search_breakdown = evaluate_breakdown(&state);
        let state_for_telemetry = state.clone();
        let s1_options_for_telemetry: Vec<poke_engine::engine::state::MoveChoice> =
            s1_options.clone();

        // Plan I: when blending is requested AND we're in NN mode, fetch the
        // raw NN policy ourselves, blend with the heuristic prior, and pass
        // the result through `new_with_priors`. Otherwise fall through to the
        // original `new_with_eval` path — bit-identical to pre-Plan-I when
        // both mix and forced-playouts are at their 0.0 defaults.
        let start = Instant::now();
        let use_blended = heuristic_prior_mix > 0.0 && eval_kind.uses_nn();
        // Plan I telemetry captures (Task 9): populated only on the blended
        // path; default-off requests keep them empty / None.
        let mut telemetry_raw_nn_probs: Vec<f32> = Vec::new();
        let mut telemetry_heuristic: Option<poke_engine::heuristic_prior::HeuristicPrior> = None;
        let mut telemetry_s1_priors_blended: Vec<f32> = Vec::new();
        // Plan I Side2 (T6): mirror Side1 telemetry locals on opp side. Stay
        // empty / None on the default-off path (--heuristic-prior-mix-side2=0).
        let mut telemetry_s2_priors_blended: Vec<f32> = Vec::new();
        let mut telemetry_heuristic_s2: Option<poke_engine::heuristic_prior::HeuristicPrior> = None;
        let mut search = if use_blended {
            // SAFETY of unwrap: `uses_nn()` guarantees the EvalKind::Nn arm.
            let client = match &eval_kind {
                poke_engine::eval_kind::EvalKind::Nn(c) => c.clone(),
                _ => unreachable!("guarded by uses_nn() above"),
            };
            let heuristic = poke_engine::heuristic_prior::compute(
                &state,
                poke_engine::nn_state_encoder::SidePerspective::Side1,
                &s1_options,
                heuristic_prior_mass_dmg,
                heuristic_prior_mass_switch,
            );
            let s1_priors_blended = {
                let json = poke_engine::nn_state_encoder::encode(
                    &state,
                    poke_engine::nn_state_encoder::SidePerspective::Side1,
                );
                match client.policy(&json, poke_engine::nn_client::Perspective::P1) {
                    Ok(resp) => {
                        // Plan I telemetry: capture the raw NN policy BEFORE
                        // we hand `resp.probs` to map_policy_to_options_blended.
                        telemetry_raw_nn_probs = resp.probs.clone();
                        let blended = poke_engine::nn_state_encoder::map_policy_to_options_blended(
                            &resp.probs,
                            &state,
                            poke_engine::nn_state_encoder::SidePerspective::Side1,
                            &s1_options,
                            heuristic.as_ref(),
                            heuristic_prior_mix,
                        );
                        telemetry_s1_priors_blended = blended.clone();
                        Some(blended)
                    }
                    Err(e) => {
                        log::warn!(
                            "NN client failed at root (Plan I blended path): {} — falling back to uniform priors",
                            e
                        );
                        None
                    }
                }
            };
            telemetry_heuristic = heuristic;
            // Plan I Side2 (Bug #3 fix): mirror the heuristic blend on opp side.
            // Side2 has no NN policy — blend heuristic with uniform baseline so
            // the opp model's prior reflects "what move would damage me / what
            // switch would survive" instead of pure uniform. Default mix=0.0
            // preserves pre-fix behavior (priors stay None).
            //
            // Scope limit: this only activates inside the `use_blended` arm
            // (i.e. heuristic_prior_mix > 0.0 AND eval_kind.uses_nn()). If the
            // operator passes --heuristic-prior-mix-side2 > 0.0 without also
            // setting --heuristic-prior-mix > 0.0 in NN mode, Side2 prior will
            // not activate. T4 mirrors this in analyze_stream_handler.
            let s2_priors_blended = if heuristic_prior_mix_side2 > 0.0 {
                let heuristic_s2 = poke_engine::heuristic_prior::compute(
                    &state,
                    poke_engine::nn_state_encoder::SidePerspective::Side2,
                    &s2_options,
                    heuristic_prior_mass_dmg,
                    heuristic_prior_mass_switch,
                );
                // Plan I Side2 (T6): capture opp heuristic for telemetry
                // before it's consumed by the blender below.
                telemetry_heuristic_s2 = heuristic_s2.clone();
                // Build uniform NN-stand-in: 1/ACTION_DIM in every slot. As
                // mix → 1.0 the blend is pure heuristic; as mix → 0.0 it is
                // pure uniform (matches pre-fix behavior).
                let uniform_probs = vec![
                    1.0_f32 / poke_engine::nn_client::ACTION_DIM as f32;
                    poke_engine::nn_client::ACTION_DIM
                ];
                let blended = poke_engine::nn_state_encoder::map_policy_to_options_blended(
                    &uniform_probs,
                    &state,
                    poke_engine::nn_state_encoder::SidePerspective::Side2,
                    &s2_options,
                    heuristic_s2.as_ref(),
                    heuristic_prior_mix_side2,
                );
                // Plan I Side2 (T6): capture blended opp priors for telemetry.
                telemetry_s2_priors_blended = blended.clone();
                Some(blended)
            } else {
                None
            };
            // engine-seed-plumbing: `seed` is plumbed through the seeded
            // constructor. None preserves pre-branch behavior bit-identically.
            MctsSearch::new_with_priors_seeded(
                state,
                s1_options,
                s2_options,
                c_puct,
                s1_priors_blended,
                s2_priors_blended,
                seed,
            )
        } else {
            // NOTE: when `use_blended == false` (default flags or non-NN eval),
            // we take the original Plan-pre-I path. `heuristic_prior_mix_side2`
            // is intentionally NOT consulted here — the opp prior remains
            // whatever `new_with_eval` constructs. T4/T5 may revisit this.
            // engine-seed-plumbing: route through the seeded variant so the
            // optional seed flows into the same RNG used by `sample_node`.
            MctsSearch::new_with_eval_seeded(
                state,
                s1_options,
                s2_options,
                &eval_kind,
                c_puct,
                seed,
            )
        };
        // Plan I: forced-playouts root constant. 0.0 (default) is a no-op.
        search.set_c_forced(forced_playouts_c);
        // Plan I Side2 (Bug #3 fix): per-side forced-playouts c-constant on
        // the opp dimension. 0.0 (default) preserves Side1-only behavior.
        search.set_c_forced_side2(forced_playouts_c_side2);
        search.run_for(Duration::from_millis(time_limit_ms));
        let mcts_result = search.snapshot(start.elapsed().as_millis() as u64);
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Plan I (Task 9): emit ONE [ENGINE-INSTRUMENT] line per /analyze
        // request, populated with telemetry when the blended path ran.
        let telemetry = InstrumentTelemetry {
            raw_nn_probs: &telemetry_raw_nn_probs,
            heuristic: telemetry_heuristic.as_ref(),
            s1_priors_blended: &telemetry_s1_priors_blended,
            forced_playouts_triggered: search.forced_playouts_triggered,
            state: Some(&state_for_telemetry),
            s1_options: &s1_options_for_telemetry,
            s2_priors_blended: &telemetry_s2_priors_blended,
            heuristic_s2: telemetry_heuristic_s2.as_ref(),
            forced_playouts_triggered_s2: search.forced_playouts_triggered_side2(),
            battle_id: battle_id.as_deref(),
            turn,
        };
        emit_engine_instrument(
            &pre_search_breakdown,
            &mcts_result,
            &s1_move_names,
            &s2_move_names,
            "analyze",
            &telemetry,
        );

        // Process side-one results: pair with move names, sort by visits descending
        let mut scored: Vec<(String, f32, u32)> = mcts_result
            .s1
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let name = s1_move_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("{:?}", r.move_choice));
                let avg = if r.visits > 0 {
                    r.total_score / r.visits as f32
                } else {
                    0.0
                };
                (name, avg, r.visits)
            })
            .collect();

        // Sort by visits (most visited = best per MCTS)
        scored.sort_by(|a, b| b.2.cmp(&a.2));

        let best = &scored[0];
        let alternatives: Vec<Alternative> = scored[1..]
            .iter()
            .filter(|(_, _, visits)| *visits > 0)
            .map(|(name, conf, visits)| Alternative {
                move_name: name.to_uppercase(),
                confidence: *conf,
                note: format!("{} visits", visits),
            })
            .collect();

        let reasoning: Vec<ReasoningStep> = mcts_result
            .principal_variation
            .iter()
            .enumerate()
            .map(|(i, step)| ReasoningStep {
                turn: i + 1,
                you: step.s1_move.clone(),
                them: step.s2_move.clone(),
            })
            .collect();
        let pv_depth = reasoning.len() as u32;

        Ok(AnalyzeResponse {
            best_move: best.0.to_uppercase(),
            confidence: best.1,
            simulations: mcts_result.iteration_count,
            depth: pv_depth,
            time_ms: elapsed_ms,
            reasoning,
            alternatives,
        })
    })
    .await;

    match result {
        Ok(Ok(response)) => Ok(Json(response)),
        Ok(Err(msg)) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: msg }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("MCTS analysis failed: {}", e),
            }),
        )),
    }
}

// Build a StreamUpdate from a MctsResult snapshot + the list of s1 move
// names (indexed to match mcts_result.s1).
//
// Mirrors the post-processing from analyze_handler (sort by visits desc,
// split into best/alternatives, compute pv from principal_variation), but
// emits the StreamUpdate shape the Python EngineClient expects.
fn build_stream_update(
    event: &'static str,
    mcts_result: &MctsResult,
    s1_move_names: &[String],
) -> StreamUpdate {
    // Pair each s1 result with its move name and avg-score
    let mut scored: Vec<(String, f32, u32)> = mcts_result
        .s1
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let name = s1_move_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("{:?}", r.move_choice));
            let avg = if r.visits > 0 {
                r.total_score / r.visits as f32
            } else {
                0.0
            };
            (name, avg, r.visits)
        })
        .collect();

    // Sort by visits (most visited = best per MCTS)
    scored.sort_by(|a, b| b.2.cmp(&a.2));

    // Handle the degenerate case where scored is empty (shouldn't happen at
    // runtime — analyze_handler already guarantees s1_options non-empty —
    // but be defensive in streaming path).
    let (best_move, confidence) = if let Some(first) = scored.first() {
        (first.0.to_uppercase(), first.1)
    } else {
        (String::new(), 0.0)
    };

    let alternatives: Vec<Alternative> = if scored.len() > 1 {
        scored[1..]
            .iter()
            .filter(|(_, _, visits)| *visits > 0)
            .map(|(name, conf, visits)| Alternative {
                move_name: name.to_uppercase(),
                confidence: *conf,
                note: format!("{} visits", visits),
            })
            .collect()
    } else {
        Vec::new()
    };

    // Flatten principal_variation steps into a string array: each step
    // becomes "you=<MOVE> them=<MOVE>". Keep it human-readable; the Python
    // consumer renders these verbatim.
    let pv: Vec<String> = mcts_result
        .principal_variation
        .iter()
        .map(|step| format!("you={} them={}", step.s1_move, step.s2_move))
        .collect();
    let depth = pv.len() as u32;

    StreamUpdate {
        event,
        best_move,
        confidence,
        sims: mcts_result.iteration_count,
        depth,
        pv,
        alternatives,
        message: None,
    }
}

/// Plan I telemetry inputs for the `[ENGINE-INSTRUMENT]` log line. All fields
/// are optional / empty in the default-off path (no NN, no heuristic mix);
/// the consumer treats empty raw_nn_probs as "no NN this request" and emits
/// 0.0 entropy / top1 / null picks / [] blend table.
pub struct InstrumentTelemetry<'a> {
    /// Raw 13-slot NN policy from the sidecar (alphabetical convention).
    /// Empty when the request didn't go through the NN-blended path.
    pub raw_nn_probs: &'a [f32],
    /// Heuristic prior result, if computed for this request.
    pub heuristic: Option<&'a poke_engine::heuristic_prior::HeuristicPrior>,
    /// Blended priors in the same order as `s1_options` (and therefore the
    /// same order as `s1_move_names`). Empty when not blended.
    pub s1_priors_blended: &'a [f32],
    /// Forced-playouts trigger count from the search.
    pub forced_playouts_triggered: u32,
    /// The state at root — needed to recompute the per-option NN priors via
    /// `map_policy_to_options` for the prior_blend_per_top5 join.
    pub state: Option<&'a poke_engine::state::State>,
    /// The s1 options at root, in the same order as `s1_move_names`. Used
    /// alongside `state` to call `map_policy_to_options`.
    pub s1_options: &'a [poke_engine::engine::state::MoveChoice],
    /// Plan I Side2 extension: opp-side priors after heuristic+uniform blend.
    /// Empty when --heuristic-prior-mix-side2 == 0.0.
    pub s2_priors_blended: &'a [f32],
    /// Plan I Side2 extension: opp-side heuristic before blending.
    pub heuristic_s2: Option<&'a poke_engine::heuristic_prior::HeuristicPrior>,
    /// Plan I Side2 extension: count of forced visits triggered on opp side.
    pub forced_playouts_triggered_s2: u32,
    /// Correlation key: Showdown battle room id (e.g. "battle-gen9ou-2256378900").
    /// `None` when the request didn't go through the proxy belief overlay
    /// (legacy direct-to-engine callers, default-off path).
    pub battle_id: Option<&'a str>,
    /// Correlation key: Showdown turn number at the time of this request.
    /// Same default-off semantics as `battle_id`.
    pub turn: Option<u32>,
}

impl<'a> InstrumentTelemetry<'a> {
    /// Default-off telemetry: no NN, no heuristic, all empty / zero. Used in
    /// code paths that don't (yet) participate in the Plan I blended pipeline
    /// (PIMC aggregator, streaming single-search) so they still produce valid
    /// log lines.
    pub fn empty(forced_playouts_triggered: u32) -> Self {
        Self {
            raw_nn_probs: &[],
            heuristic: None,
            s1_priors_blended: &[],
            forced_playouts_triggered,
            state: None,
            s1_options: &[],
            s2_priors_blended: &[],
            heuristic_s2: None,
            forced_playouts_triggered_s2: 0,
            // Correlation keys default to None on this constructor — the
            // signature stays backward-compat (callers pass `forced` only)
            // and the engine-instrument payload renders `null` for both.
            battle_id: None,
            turn: None,
        }
    }
}

/// Shannon entropy of a probability distribution (natural log). Treats
/// p == 0 as 0 contribution (`0 * log 0 := 0`). Returns 0.0 on empty input.
/// We add 0.0 at the end to normalize any signed-zero artifacts that arise
/// from `-p * ln(p)` for p just above zero.
fn policy_entropy(probs: &[f32]) -> f32 {
    let h: f32 = probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    h + 0.0
}

/// Top-1 probability mass. Returns 0.0 on empty input.
fn policy_top1(probs: &[f32]) -> f32 {
    probs.iter().cloned().fold(0.0_f32, f32::max)
}

/// Emit a single-line `[ENGINE-INSTRUMENT]` JSON log with the eval breakdown
/// + top-5 s1 / top-3 s2 MCTS branches (visits, avg value, prior). Designed
/// to be greppable from `~/plan-e-engine-*.log` for post-hoc diagnostics of
/// "why did the engine pick X?" — never used for control flow, observation
/// only.
///
/// Call once per `/analyze/stream` request, AFTER the final MCTS snapshot,
/// so visit counts are populated. JSON is kept under ~1KB by truncating to
/// top-5/top-3 with rounded floats. If the result has 0 sims, we still emit
/// (with an empty branches array) — silence is worse than empty.
///
/// Plan I additions (Task 9): `policy_entropy`, `policy_top1_prob`,
/// `forced_playouts_triggered`, `heuristic_pick_dmg`, `heuristic_pick_switch`,
/// `prior_blend_per_top5`. All six are populated from `telemetry`; default-off
/// requests pass an `empty()` telemetry and get neutral values.
fn emit_engine_instrument(
    breakdown: &EvalBreakdown,
    mcts_result: &MctsResult,
    s1_move_names: &[String],
    s2_move_names: &[String],
    label: &str,
    telemetry: &InstrumentTelemetry<'_>,
) {
    fn round2(x: f32) -> f32 {
        // Round to 2 decimals to keep log lines compact and stable.
        (x * 100.0).round() / 100.0
    }
    fn round4(x: f32) -> f32 {
        (x * 10000.0).round() / 10000.0
    }

    // Plan I: precompute the per-option NN priors (in s1_options order) so we
    // can join them with my_top5 below. Empty when this request didn't query
    // the NN (default-off path).
    let raw_nn_priors_in_options_order: Vec<f32> = if !telemetry.raw_nn_probs.is_empty()
        && telemetry.state.is_some()
        && !telemetry.s1_options.is_empty()
    {
        poke_engine::nn_state_encoder::map_policy_to_options(
            telemetry.raw_nn_probs,
            telemetry.state.unwrap(),
            poke_engine::nn_state_encoder::SidePerspective::Side1,
            telemetry.s1_options,
        )
    } else {
        Vec::new()
    };

    // Sort s1 by visits desc, take top 5.
    let mut s1_idx: Vec<usize> = (0..mcts_result.s1.len()).collect();
    s1_idx.sort_by(|a, b| mcts_result.s1[*b].visits.cmp(&mcts_result.s1[*a].visits));
    let my_top5: Vec<serde_json::Value> = s1_idx
        .iter()
        .take(5)
        .map(|&i| {
            let r = &mcts_result.s1[i];
            let name = s1_move_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("{:?}", r.move_choice))
                .to_uppercase();
            let avg = if r.visits > 0 { r.total_score / r.visits as f32 } else { 0.0 };
            // Prior is not stored on MctsSideResult — we don't expose root
            // priors through snapshot(). Caller-visible workaround: prior
            // is implicit via visit share. Emit null and let the analyst
            // compute share from total iterations if needed.
            serde_json::json!({
                "move": name,
                "visits": r.visits,
                "value": round4(avg),
            })
        })
        .collect();

    // Plan I: per-top-5 raw NN prior + blended prior, joined by option index.
    // Both `raw_nn_priors_in_options_order` and `telemetry.s1_priors_blended`
    // are indexed in s1_options order, which is the same order s1_move_names
    // (and mcts_result.s1) uses. So `i` from s1_idx is the join key directly.
    let prior_blend_per_top5: Vec<serde_json::Value> = if raw_nn_priors_in_options_order.is_empty()
        || telemetry.s1_priors_blended.is_empty()
    {
        Vec::new()
    } else {
        s1_idx
            .iter()
            .take(5)
            .map(|&i| {
                let r = &mcts_result.s1[i];
                let name = s1_move_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("{:?}", r.move_choice))
                    .to_uppercase();
                let prior_nn = raw_nn_priors_in_options_order.get(i).copied().unwrap_or(0.0);
                let prior_blended = telemetry.s1_priors_blended.get(i).copied().unwrap_or(0.0);
                serde_json::json!({
                    "move": name,
                    "prior_nn": round4(prior_nn),
                    "prior_blended": round4(prior_blended),
                })
            })
            .collect()
    };

    let mut s2_idx: Vec<usize> = (0..mcts_result.s2.len()).collect();
    s2_idx.sort_by(|a, b| mcts_result.s2[*b].visits.cmp(&mcts_result.s2[*a].visits));
    let opp_top3: Vec<serde_json::Value> = s2_idx
        .iter()
        .take(3)
        .map(|&i| {
            let r = &mcts_result.s2[i];
            let name = s2_move_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("{:?}", r.move_choice))
                .to_uppercase();
            let avg = if r.visits > 0 { r.total_score / r.visits as f32 } else { 0.0 };
            serde_json::json!({
                "move": name,
                "visits": r.visits,
                "value": round4(avg),
            })
        })
        .collect();

    // Plan I: heuristic pick names. `Option<&HeuristicPrior>` -> Option<String>.
    // We use Debug for Choices / PokemonName; serde_json renders None as null.
    let heuristic_pick_dmg: Option<String> = telemetry
        .heuristic
        .and_then(|h| h.damage_calc_pick.as_ref().map(|c| format!("{:?}", c)));
    let heuristic_pick_switch: Option<String> = telemetry
        .heuristic
        .and_then(|h| h.matchup_switch_pick.as_ref().map(|p| format!("{:?}", p)));
    // Plan I Side2 (T6): mirror heuristic pick names on opp side. None when
    // --heuristic-prior-mix-side2 == 0.0 (default), serializes as null.
    let s2_heuristic_pick_dmg: Option<String> = telemetry
        .heuristic_s2
        .and_then(|h| h.damage_calc_pick.as_ref().map(|c| format!("{:?}", c)));
    let s2_heuristic_pick_switch: Option<String> = telemetry
        .heuristic_s2
        .and_then(|h| h.matchup_switch_pick.as_ref().map(|p| format!("{:?}", p)));

    let payload = serde_json::json!({
        // Correlation keys (engine-log → postmortem join). Render `null`
        // when the request didn't come through the proxy belief overlay
        // (legacy direct-to-engine callers, PIMC default-off path).
        "battle_id": telemetry.battle_id,
        "turn": telemetry.turn,
        "label": label,
        "sims": mcts_result.iteration_count,
        "my_top5": my_top5,
        "opp_top3": opp_top3,
        "eval_breakdown": {
            "total": round2(breakdown.total),
            "hp_term": round2(breakdown.hp_term),
            "hazards_term": round2(breakdown.hazards_term),
            "boost_term_s1": round2(breakdown.boost_term_s1),
            "boost_term_s2": round2(breakdown.boost_term_s2),
            "threat_score_s1": round2(breakdown.threat_score_s1),
            "threat_score_s2": round2(breakdown.threat_score_s2),
            "volatile_status_term": round2(breakdown.volatile_status_term),
            "side_conditions_term": round2(breakdown.side_conditions_term),
            "tera_term": round2(breakdown.tera_term),
            "status_threat_term": round2(breakdown.status_threat_term),
        },
        // Plan I telemetry fields. Default-off path emits 0.0 / null / [].
        // Entropy/top1 are computed over the legal-subset policy (what PUCT
        // actually consumes), not the raw 13-slot vector — slots for fainted
        // reserves and empty move slots would dilute the signal.
        "policy_entropy": round4(policy_entropy(&raw_nn_priors_in_options_order)),
        "policy_top1_prob": round4(policy_top1(&raw_nn_priors_in_options_order)),
        "forced_playouts_triggered": telemetry.forced_playouts_triggered,
        "heuristic_pick_dmg": heuristic_pick_dmg,
        "heuristic_pick_switch": heuristic_pick_switch,
        "prior_blend_per_top5": prior_blend_per_top5,
        // Plan I Side2 (T6): symmetric Side2 fields. Empty / null on the
        // default-off path (--heuristic-prior-mix-side2 == 0.0).
        "s2_priors_blended": telemetry.s2_priors_blended,
        "s2_heuristic_pick_dmg": s2_heuristic_pick_dmg,
        "s2_heuristic_pick_switch": s2_heuristic_pick_switch,
        "forced_playouts_triggered_s2": telemetry.forced_playouts_triggered_s2,
    });

    // serde_json::to_string is infallible for json! values built from primitives.
    let line = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    log::info!("[ENGINE-INSTRUMENT] {}", line);
}

fn error_stream_update(msg: String) -> StreamUpdate {
    StreamUpdate {
        event: "error",
        best_move: String::new(),
        confidence: 0.0,
        sims: 0,
        depth: 0,
        pv: Vec::new(),
        alternatives: Vec::new(),
        message: Some(msg),
    }
}

async fn analyze_stream_handler(
    AxumState(app): AxumState<AppState>,
    body: String,
) -> impl IntoResponse {
    // Parse out optional timing overrides from the request body. Defaults
    // are 5000ms for the time limit and 250ms for the update interval.
    // NOTE: this endpoint uses `timeLimitMs` / `updateIntervalMs` to match
    // the Python EngineClient. The legacy `/analyze` endpoint still uses
    // `timeLimit` (no Ms suffix) per the Cobblemon mod contract.
    let parsed = serde_json::from_str::<serde_json::Value>(&body).ok();
    let time_limit_ms = parsed
        .as_ref()
        .and_then(|v| v.get("timeLimitMs")?.as_u64())
        .unwrap_or(DEFAULT_TIME_LIMIT_MS);
    let update_interval_ms = parsed
        .as_ref()
        .and_then(|v| v.get("updateIntervalMs")?.as_u64())
        .unwrap_or(DEFAULT_UPDATE_INTERVAL_MS);
    // Engine-log correlation keys forwarded by the proxy (apply_belief).
    // Both are optional — direct-to-engine callers omit them and the
    // [ENGINE-INSTRUMENT] payload renders `null`.
    let battle_id: Option<String> = parsed
        .as_ref()
        .and_then(|v| v.get("battleId").and_then(|x| x.as_str()))
        .map(String::from);
    let turn: Option<u32> = parsed
        .as_ref()
        .and_then(|v| v.get("turn").and_then(|x| x.as_u64()))
        .map(|u| u as u32);
    // engine-seed-plumbing: optional deterministic RNG seed. Mirrors the
    // /analyze handler — when present, makes the streaming search
    // reproducible for replay-based A/B testing.
    let seed: Option<u64> = parsed
        .as_ref()
        .and_then(|v| v.get("seed").and_then(|x| x.as_u64()));

    let raw_json = body;
    let eval_kind = app.eval_kind.clone();
    let c_puct = app.c_puct;
    // Plan I (Task 8b): mirror analyze_handler — capture mix/mass/forced-playouts
    // knobs by-value (all f32, Copy) so the blocking thread below can apply the
    // blended-prior + forced-playouts wiring on this streaming path too.
    let heuristic_prior_mix = app.heuristic_prior_mix;
    let forced_playouts_c = app.forced_playouts_c;
    let heuristic_prior_mass_dmg = app.heuristic_prior_mass_dmg;
    let heuristic_prior_mass_switch = app.heuristic_prior_mass_switch;
    // Plan I Side2 (Bug #3 fix): mirror analyze_handler — symmetric heuristic
    // prior on the opponent perspective. Default 0.0 keeps pre-fix behavior
    // bit-identical. T5 wires the per-side forced-playouts c-constant via
    // `set_c_forced_side2`.
    let heuristic_prior_mix_side2 = app.heuristic_prior_mix_side2;
    let forced_playouts_c_side2 = app.forced_playouts_c_side2;

    // Channel between the blocking search thread and the async streaming
    // response body. Buffer 32 is enough for ~30s of updates at 1 Hz.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);

    // Dedicated OS thread. Node uses raw *mut Node parent pointers and is
    // !Send, so we cannot use tokio::task::spawn_blocking (which schedules
    // onto a Send-requiring pool). std::thread::spawn gives us a pinned
    // thread that owns the search for its entire lifetime.
    //
    // NN client calls (when --nn-eval is on) happen INSIDE this thread
    // through MctsSearch::new_with_eval. Never call the NN client from the
    // surrounding async context.
    std::thread::spawn(move || {
        // Helper: serialize a StreamUpdate as a single NDJSON line and push
        // to the channel. Returns false if the receiver has been dropped
        // (client disconnected).
        fn emit(
            tx: &tokio::sync::mpsc::Sender<String>,
            update: &StreamUpdate,
        ) -> bool {
            let mut line = match serde_json::to_string(update) {
                Ok(s) => s,
                Err(_) => return true, // skip malformed update, keep going
            };
            line.push('\n');
            tx.blocking_send(line).is_ok()
        }

        // Capture first hypothesis before consuming the vec, so we can resolve
        // s1 move names for the response after aggregation.
        let first_hyp_for_names = match extract_hypotheses(&raw_json) {
            Some(hs) if !hs.is_empty() => hs[0].clone(),
            _ => raw_json.clone(),  // unused in single-search path; harmless
        };

        // PIMC dispatch: if "hypotheses" is present, run K parallel searches,
        // aggregate, emit one final event. Otherwise fall through to single-search.
        if let Some(hypotheses) = extract_hypotheses(&raw_json) {
            let k = hypotheses.len();
            let per_hypothesis_budget = time_limit_ms / (k as u64).max(1);

            // Spawn K worker threads; each owns one hypothesis end-to-end.
            let mut handles = Vec::with_capacity(k);
            for (i, hyp_json) in hypotheses.into_iter().enumerate() {
                let h = handles_spawn(
                    hyp_json,
                    per_hypothesis_budget,
                    i as u64,
                    eval_kind.clone(),
                    c_puct,
                );
                handles.push(h);
            }
            // Join all. If any worker errored, abort PIMC and emit error.
            let mut results = Vec::with_capacity(k);
            for h in handles {
                match h.join() {
                    Ok(Ok(r)) => results.push(r),
                    Ok(Err(msg)) => {
                        let _ = emit(&tx, &error_stream_update(format!("PIMC worker error: {}", msg)));
                        return;
                    }
                    Err(_) => {
                        let _ = emit(&tx, &error_stream_update("PIMC worker panicked".to_string()));
                        return;
                    }
                }
            }
            if results.is_empty() {
                let _ = emit(&tx, &error_stream_update("PIMC produced no results".to_string()));
                return;
            }

            // Aggregate.
            let aggregated = poke_engine::mcts::aggregate_pimc(results, false, None);

            // Synthesize the s1 move-name list from the FIRST hypothesis's state.
            // (All K hypotheses share the same player-side team — only opp varies.)
            let (s1_names, s2_names, pimc_breakdown) =
                match poke_engine::translate::auto_detect_and_parse(&first_hyp_for_names) {
                    Ok(state) => {
                        let (s1_options, s2_options) = state.root_get_all_options();
                        let s1n: Vec<String> = s1_options.iter().map(|mc| mc.to_string(&state.side_one)).collect();
                        let s2n: Vec<String> = s2_options.iter().map(|mc| mc.to_string(&state.side_two)).collect();
                        let bd = evaluate_breakdown(&state);
                        (s1n, s2n, bd)
                    }
                    Err(_) => (Vec::new(), Vec::new(), EvalBreakdown::default()),
                };
            // [ENGINE-INSTRUMENT] for PIMC aggregated result. Eval breakdown
            // is from the FIRST hypothesis (same player-side state across K).
            // PIMC path doesn't (yet) participate in Plan I blending — pass
            // empty telemetry so the new fields are present-but-neutral.
            // Override the correlation keys so the PIMC log line still joins
            // back to the postmortem on (battle_id, turn).
            let mut pimc_telemetry = InstrumentTelemetry::empty(0);
            pimc_telemetry.battle_id = battle_id.as_deref();
            pimc_telemetry.turn = turn;
            emit_engine_instrument(
                &pimc_breakdown,
                &aggregated,
                &s1_names,
                &s2_names,
                "pimc",
                &pimc_telemetry,
            );
            let final_update = build_stream_update("final", &aggregated, &s1_names);
            let _ = emit(&tx, &final_update);
            return;
        }

        // Parse the state. If parsing fails, emit a single error event and
        // return.
        let state = match auto_detect_and_parse(&raw_json) {
            Ok(s) => s,
            Err(e) => {
                let _ = emit(&tx, &error_stream_update(format!("State parse error: {}", e)));
                return;
            }
        };

        let (s1_options, s2_options) = state.root_get_all_options();
        if s1_options.is_empty() {
            let _ = emit(
                &tx,
                &error_stream_update("No legal moves for side one".to_string()),
            );
            return;
        }

        // Snapshot side_one for move name resolution BEFORE moving state
        // into the search.
        let s1_move_names: Vec<String> = s1_options
            .iter()
            .map(|mc| mc.to_string(&state.side_one))
            .collect();
        // Also snapshot s2 names + eval breakdown for the [ENGINE-INSTRUMENT]
        // log line. State is moved into MctsSearch below, so capture these now.
        let s2_move_names: Vec<String> = s2_options
            .iter()
            .map(|mc| mc.to_string(&state.side_two))
            .collect();
        let pre_search_breakdown = evaluate_breakdown(&state);
        // Plan I (Task 8b telemetry): keep state + s1_options snapshots for
        // the populated [ENGINE-INSTRUMENT] line on the blended path. Mirrors
        // analyze_handler's `state_for_telemetry` / `s1_options_for_telemetry`.
        let state_for_telemetry = state.clone();
        let s1_options_for_telemetry: Vec<poke_engine::engine::state::MoveChoice> =
            s1_options.clone();

        // Plan I (Task 8b): when blending is requested AND we're in NN mode,
        // fetch the raw NN policy ourselves, blend with the heuristic prior,
        // and pass through `new_with_priors`. Otherwise fall through to the
        // original `new_with_eval` path. Mirrors analyze_handler — both
        // compute Side1 (NN-blended) and Side2 (uniform-blended when
        // --heuristic-prior-mix-side2 > 0.0) priors.
        let use_blended = heuristic_prior_mix > 0.0 && eval_kind.uses_nn();
        let mut telemetry_raw_nn_probs: Vec<f32> = Vec::new();
        let mut telemetry_heuristic: Option<poke_engine::heuristic_prior::HeuristicPrior> = None;
        let mut telemetry_s1_priors_blended: Vec<f32> = Vec::new();
        // Plan I Side2 (T6): mirror Side1 telemetry locals on opp side. Stay
        // empty / None on the default-off path (--heuristic-prior-mix-side2=0).
        let mut telemetry_s2_priors_blended: Vec<f32> = Vec::new();
        let mut telemetry_heuristic_s2: Option<poke_engine::heuristic_prior::HeuristicPrior> = None;
        // MctsSearch owns the State. Cloning is cheap relative to a
        // multi-second MCTS run.
        let mut search = if use_blended {
            // SAFETY of unwrap: `uses_nn()` guarantees the EvalKind::Nn arm.
            let client = match &eval_kind {
                poke_engine::eval_kind::EvalKind::Nn(c) => c.clone(),
                _ => unreachable!("guarded by uses_nn() above"),
            };
            let heuristic = poke_engine::heuristic_prior::compute(
                &state,
                poke_engine::nn_state_encoder::SidePerspective::Side1,
                &s1_options,
                heuristic_prior_mass_dmg,
                heuristic_prior_mass_switch,
            );
            let s1_priors_blended = {
                let json = poke_engine::nn_state_encoder::encode(
                    &state,
                    poke_engine::nn_state_encoder::SidePerspective::Side1,
                );
                match client.policy(&json, poke_engine::nn_client::Perspective::P1) {
                    Ok(resp) => {
                        telemetry_raw_nn_probs = resp.probs.clone();
                        let blended = poke_engine::nn_state_encoder::map_policy_to_options_blended(
                            &resp.probs,
                            &state,
                            poke_engine::nn_state_encoder::SidePerspective::Side1,
                            &s1_options,
                            heuristic.as_ref(),
                            heuristic_prior_mix,
                        );
                        telemetry_s1_priors_blended = blended.clone();
                        Some(blended)
                    }
                    Err(e) => {
                        log::warn!(
                            "NN client failed at root (Plan I blended path, stream): {} — falling back to uniform priors",
                            e
                        );
                        None
                    }
                }
            };
            telemetry_heuristic = heuristic;
            // Plan I Side2 (Bug #3 fix): mirror the heuristic blend on opp side.
            // Side2 has no NN policy — blend heuristic with uniform baseline so
            // the opp model's prior reflects "what move would damage me / what
            // switch would survive" instead of pure uniform. Default mix=0.0
            // preserves pre-fix behavior (priors stay None).
            //
            // Scope limit: this only activates inside the `use_blended` arm
            // (i.e. heuristic_prior_mix > 0.0 AND eval_kind.uses_nn()). If the
            // operator passes --heuristic-prior-mix-side2 > 0.0 without also
            // setting --heuristic-prior-mix > 0.0 in NN mode, Side2 prior will
            // not activate.
            let s2_priors_blended = if heuristic_prior_mix_side2 > 0.0 {
                let heuristic_s2 = poke_engine::heuristic_prior::compute(
                    &state,
                    poke_engine::nn_state_encoder::SidePerspective::Side2,
                    &s2_options,
                    heuristic_prior_mass_dmg,
                    heuristic_prior_mass_switch,
                );
                // Plan I Side2 (T6): capture opp heuristic for telemetry
                // before it's consumed by the blender below.
                telemetry_heuristic_s2 = heuristic_s2.clone();
                // Build uniform NN-stand-in: 1/ACTION_DIM in every slot. As
                // mix → 1.0 the blend is pure heuristic; as mix → 0.0 it is
                // pure uniform (matches pre-fix behavior).
                let uniform_probs = vec![
                    1.0_f32 / poke_engine::nn_client::ACTION_DIM as f32;
                    poke_engine::nn_client::ACTION_DIM
                ];
                let blended = poke_engine::nn_state_encoder::map_policy_to_options_blended(
                    &uniform_probs,
                    &state,
                    poke_engine::nn_state_encoder::SidePerspective::Side2,
                    &s2_options,
                    heuristic_s2.as_ref(),
                    heuristic_prior_mix_side2,
                );
                // Plan I Side2 (T6): capture blended opp priors for telemetry.
                telemetry_s2_priors_blended = blended.clone();
                Some(blended)
            } else {
                None
            };
            // engine-seed-plumbing: thread the seed through the seeded
            // constructor (None = pre-branch behavior).
            MctsSearch::new_with_priors_seeded(
                state.clone(),
                s1_options,
                s2_options,
                c_puct,
                s1_priors_blended,
                s2_priors_blended,
                seed,
            )
        } else {
            // NOTE: when `use_blended == false` (default flags or non-NN eval),
            // we take the original Plan-pre-I path. `heuristic_prior_mix_side2`
            // is intentionally NOT consulted here — the opp prior remains
            // whatever `new_with_eval` constructs.
            // engine-seed-plumbing: route through the seeded variant.
            MctsSearch::new_with_eval_seeded(
                state.clone(),
                s1_options,
                s2_options,
                &eval_kind,
                c_puct,
                seed,
            )
        };
        // Plan I: forced-playouts root constant. 0.0 (default) is a no-op.
        search.set_c_forced(forced_playouts_c);
        // Plan I Side2 (Bug #3 fix): per-side forced-playouts c-constant on
        // the opp dimension. 0.0 (default) preserves Side1-only behavior.
        search.set_c_forced_side2(forced_playouts_c_side2);
        let start = Instant::now();
        let time_limit = Duration::from_millis(time_limit_ms);
        let interval = Duration::from_millis(update_interval_ms);

        loop {
            let elapsed = start.elapsed();
            if elapsed >= time_limit {
                break;
            }
            // Clamp this slice to whatever budget remains.
            let remaining = time_limit - elapsed;
            let slice = if remaining < interval { remaining } else { interval };
            search.run_for(slice);

            let snap = search.snapshot(start.elapsed().as_millis() as u64);
            let update = build_stream_update("update", &snap, &s1_move_names);
            if !emit(&tx, &update) {
                // Client disconnected. Abort the search.
                return;
            }
        }

        // Final snapshot.
        let snap = search.snapshot(start.elapsed().as_millis() as u64);
        // [ENGINE-INSTRUMENT] structured log: emit ONE JSON line per request
        // capturing the root eval breakdown + top-K MCTS branches. Plan I
        // (Task 8b) — telemetry is now populated when the blended path ran;
        // empty captures + the search's forced-playouts counter otherwise.
        let telemetry = InstrumentTelemetry {
            raw_nn_probs: &telemetry_raw_nn_probs,
            heuristic: telemetry_heuristic.as_ref(),
            s1_priors_blended: &telemetry_s1_priors_blended,
            forced_playouts_triggered: search.forced_playouts_triggered,
            state: Some(&state_for_telemetry),
            s1_options: &s1_options_for_telemetry,
            s2_priors_blended: &telemetry_s2_priors_blended,
            heuristic_s2: telemetry_heuristic_s2.as_ref(),
            forced_playouts_triggered_s2: search.forced_playouts_triggered_side2(),
            battle_id: battle_id.as_deref(),
            turn,
        };
        emit_engine_instrument(
            &pre_search_breakdown,
            &snap,
            &s1_move_names,
            &s2_move_names,
            "single",
            &telemetry,
        );
        let final_update = build_stream_update("final", &snap, &s1_move_names);
        let _ = emit(&tx, &final_update);
    });

    // Bridge the blocking channel into an async Body stream. Each item is
    // already a complete NDJSON line (serialized JSON + '\n').
    let body_stream = async_stream::stream! {
        while let Some(line) = rx.recv().await {
            yield Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(line));
        }
    };

    // Content-Type: application/x-ndjson is the de facto standard for
    // newline-delimited JSON.
    (
        [("content-type", "application/x-ndjson")],
        Body::from_stream(body_stream),
    )
}

// -- Main --

#[tokio::main]
async fn main() {
    // Initialize logging (RUST_LOG=info,poke_engine=debug to crank verbosity).
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    let cli = Cli::parse();

    // engine-prior-tuning: install runtime knobs before any search runs.
    // Defaults match pre-branch behavior bit-identically.
    poke_engine::tuning::init_tuning(poke_engine::tuning::TuningConfig {
        prior_cap: cli.prior_cap,
        dirichlet_alpha: cli.dirichlet_alpha,
        dirichlet_eps: cli.dirichlet_eps,
        eval_slope: cli.eval_slope,
    });
    log::info!(
        "engine-prior-tuning: prior_cap={} dirichlet_alpha={} dirichlet_eps={} eval_slope={}",
        cli.prior_cap, cli.dirichlet_alpha, cli.dirichlet_eps, cli.eval_slope,
    );

    let eval_kind = if cli.nn_eval {
        let client = NnClient::new(
            cli.nn_url.clone(),
            Duration::from_millis(cli.nn_timeout_ms),
        );
        // Best-effort health check; do NOT fail-loudly here — engine should
        // run even if sidecar isn't up yet (per-request fallback covers it).
        match client.healthz() {
            Ok(()) => log::info!("sidecar /healthz OK at {}", cli.nn_url),
            Err(e) => log::warn!(
                "sidecar /healthz FAILED at {}: {} — engine will still run, requests will fall back to heuristic",
                cli.nn_url,
                e
            ),
        }
        log::info!(
            "Plan E NN-prior mode ENABLED: nn_url={} nn_timeout_ms={} c_puct={}",
            cli.nn_url,
            cli.nn_timeout_ms,
            cli.c_puct,
        );
        EvalKind::Nn(Arc::new(client))
    } else {
        log::info!(
            "Heuristic-only mode (Plan E flags inert): c_puct={}",
            cli.c_puct
        );
        EvalKind::Heuristic
    };

    let app_state = AppState {
        eval_kind,
        c_puct: cli.c_puct,
        heuristic_prior_mix: cli.heuristic_prior_mix,
        forced_playouts_c: cli.forced_playouts_c,
        heuristic_prior_mix_side2: cli.heuristic_prior_mix_side2,
        forced_playouts_c_side2: cli.forced_playouts_c_side2,
        heuristic_prior_mass_dmg: cli.heuristic_prior_mass_dmg,
        heuristic_prior_mass_switch: cli.heuristic_prior_mass_switch,
    };

    // Permissive CORS: server is localhost-only, no security concern.
    // Needed so browser extensions / userscripts / dev tools can fetch directly.
    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    let app = Router::new()
        .route("/status", get(status_handler))
        .route("/analyze", post(analyze_handler))
        .route("/analyze/stream", post(analyze_stream_handler))
        .layer(cors)
        .with_state(app_state);

    let addr = format!("0.0.0.0:{}", cli.port);
    println!("poke-engine MCTS server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    println!("Listening on http://{}", addr);

    axum::serve(listener, app)
        .await
        .expect("Server error");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_hypotheses_present() {
        let body = r#"{"hypotheses":[{"a":1},{"b":2}],"timeLimitMs":1000}"#;
        let h = extract_hypotheses(body).expect("should detect hypotheses");
        assert_eq!(h.len(), 2);
        assert!(h[0].contains("\"a\""));
        assert!(h[1].contains("\"b\""));
    }

    #[test]
    fn test_extract_hypotheses_absent() {
        let body = r#"{"sideOne":{},"timeLimitMs":1000}"#;
        assert!(extract_hypotheses(body).is_none());
    }

    #[test]
    fn test_extract_hypotheses_empty_array_returns_none() {
        let body = r#"{"hypotheses":[]}"#;
        assert!(extract_hypotheses(body).is_none());
    }
}
