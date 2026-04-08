use axum::{
    extract::Json,
    http::StatusCode,
    routing::{get, post},
    Router,
};
use poke_engine::mcts::perform_mcts;
use poke_engine::translate::auto_detect_and_parse;
use serde::Serialize;
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 7267;
const DEFAULT_TIME_LIMIT_MS: u64 = 5000;

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

// -- Main --

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/status", get(status_handler))
        .route("/analyze", post(analyze_handler));

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
