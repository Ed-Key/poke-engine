use axum::{
    extract::Json,
    http::StatusCode,
    routing::{get, post},
    Router,
};
use poke_engine::mcts::perform_mcts;
use poke_engine::translate::BattleRequest;
use poke_engine::translate::to_poke_state;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 7267;
const DEFAULT_TIME_LIMIT_MS: u64 = 5000;

// -- Request / Response types --

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnalyzeRequest {
    #[serde(flatten)]
    battle: BattleRequest,
    #[serde(default)]
    time_limit: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalyzeResponse {
    best_move: String,
    confidence: f32,
    simulations: u32,
    depth: u32,
    time_ms: u64,
    reasoning: Vec<String>,
    alternatives: Vec<Alternative>,
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
    Json(request): Json<AnalyzeRequest>,
) -> Result<Json<AnalyzeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let time_limit_ms = request.time_limit.unwrap_or(DEFAULT_TIME_LIMIT_MS);

    // Translate to poke-engine State — catch panics from deserialization
    let battle = request.battle;
    let result = tokio::task::spawn_blocking(move || {
        // Translate JSON -> State
        let mut state = to_poke_state(&battle);

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

        Ok(AnalyzeResponse {
            best_move: best.0.to_uppercase(),
            confidence: best.1,
            simulations: mcts_result.iteration_count,
            depth: 4, // MCTS doesn't track explicit depth; placeholder
            time_ms: elapsed_ms,
            reasoning: vec![],
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
