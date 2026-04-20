use axum::{
    body::Body,
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use poke_engine::mcts::{perform_mcts, MctsResult, MctsSearch};
use poke_engine::translate::auto_detect_and_parse;
use serde::Serialize;
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 7267;
const DEFAULT_TIME_LIMIT_MS: u64 = 5000;
const DEFAULT_UPDATE_INTERVAL_MS: u64 = 250;

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
    body: String,
) -> Result<Json<AnalyzeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Extract timeLimit if present in the JSON
    let time_limit_ms = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("timeLimit")?.as_u64())
        .unwrap_or(DEFAULT_TIME_LIMIT_MS);

    let raw_json = body;

    // Translate to poke-engine State — catch panics from deserialization
    let result = tokio::task::spawn_blocking(move || {
        // Translate JSON -> State (auto-detects format)
        let mut state = auto_detect_and_parse(&raw_json)
            .map_err(|e| format!("State parse error: {}", e))?;

        // Get legal options (root includes tera/mega)
        let (s1_options, s2_options) = state.root_get_all_options();

        if s1_options.is_empty() {
            return Err("No legal moves for side one".to_string());
        }

        // Snapshot side_one for move name resolution
        let side_one_ref = &state.side_one;
        let s1_move_names: Vec<String> = s1_options
            .iter()
            .map(|mc| mc.to_string(side_one_ref))
            .collect();

        // Run MCTS
        let start = Instant::now();
        let mcts_result = perform_mcts(
            &mut state,
            s1_options,
            s2_options,
            Duration::from_millis(time_limit_ms),
        );
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

async fn analyze_stream_handler(body: String) -> impl IntoResponse {
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

    // Channel between the blocking search thread and the async streaming
    // response body. Buffer 32 is enough for ~30s of updates at 1 Hz.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);

    // Dedicated OS thread. Node uses raw *mut Node parent pointers and is
    // !Send, so we cannot use tokio::task::spawn_blocking (which schedules
    // onto a Send-requiring pool). std::thread::spawn gives us a pinned
    // thread that owns the search for its entire lifetime.
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
        let mut search = MctsSearch::new(state.clone(), s1_options, s2_options);
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
    let app = Router::new()
        .route("/status", get(status_handler))
        .route("/analyze", post(analyze_handler))
        .route("/analyze/stream", post(analyze_stream_handler));

    let addr = format!("0.0.0.0:{}", DEFAULT_PORT);
    println!("poke-engine MCTS server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    println!("Listening on http://{}", addr);

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
