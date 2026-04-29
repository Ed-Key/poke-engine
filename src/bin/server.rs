use axum::{
    body::Body,
    extract::{Json, State as AxumState},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
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
}

/// Shared per-process state plumbed through axum handlers.
///
/// Cheap to clone (Arc-wrapped client). Created once at startup; passed by
/// value into the Router via `with_state`.
#[derive(Clone)]
pub struct AppState {
    pub eval_kind: EvalKind,
    pub c_puct: f32,
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
    let time_limit_ms = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("timeLimit")?.as_u64())
        .unwrap_or(DEFAULT_TIME_LIMIT_MS);

    let raw_json = body;
    let eval_kind = app.eval_kind.clone();
    let c_puct = app.c_puct;

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

        // Run MCTS via the Plan E entry point (handles NN dispatch + PUCT).
        let start = Instant::now();
        let mut search = MctsSearch::new_with_eval(
            state,
            s1_options,
            s2_options,
            &eval_kind,
            c_puct,
        );
        search.run_for(Duration::from_millis(time_limit_ms));
        let mcts_result = search.snapshot(start.elapsed().as_millis() as u64);
        let elapsed_ms = start.elapsed().as_millis() as u64;

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

    let raw_json = body;
    let eval_kind = app.eval_kind.clone();
    let c_puct = app.c_puct;

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
            let s1_names = match poke_engine::translate::auto_detect_and_parse(&first_hyp_for_names) {
                Ok(state) => {
                    let (s1_options, _) = state.root_get_all_options();
                    s1_options.iter().map(|mc| mc.to_string(&state.side_one)).collect::<Vec<_>>()
                }
                Err(_) => Vec::new(),
            };
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

        // MctsSearch owns the State. Cloning is cheap relative to a
        // multi-second MCTS run.
        let mut search = MctsSearch::new_with_eval(
            state.clone(),
            s1_options,
            s2_options,
            &eval_kind,
            c_puct,
        );
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
