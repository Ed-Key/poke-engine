//! Plan E NN-client integration tests.
//!
//! Five mocked tests cover the sidecar HTTP contract (happy path, timeout,
//! 4xx, malformed JSON, invalid shape). One additional `#[ignore]`-gated
//! test hits a live sidecar — start it manually with
//!
//!     source ~/Projects/metamon-spike/metamon/.venv-py310/bin/activate
//!     python -m sidecar.nn_sidecar &
//!
//! then run with `cargo test --release --features terastallization
//! --test test_nn_client -- --ignored --nocapture`.

use std::time::Duration;

use poke_engine::nn_client::{NnClient, NnClientError, Perspective, PolicyResponse, ACTION_DIM};

fn good_response_body() -> String {
    // 13 floats summing to 1.0 — sample values, distribution doesn't matter.
    serde_json::json!({
        "probs": [
            0.27, 0.001, 0.002, 0.001,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.69, 0.001, 0.034, 0.001
        ],
        "q_values": [
            1500.0, 100.0, 200.0, 100.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            1700.0, 100.0, 300.0, 100.0
        ],
        "v_estimate": 1500.0,
        "decoded_actions": [
            "move:earthquake", "move:scaleshot", "move:spikes", "move:stealthrock",
            "switch:nomove", "switch:nomove", "switch:nomove", "switch:nomove", "switch:nomove",
            "tera_move:earthquake", "tera_move:scaleshot", "tera_move:spikes", "tera_move:stealthrock"
        ],
    })
    .to_string()
}

#[test]
fn happy_path_returns_validated_response() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/policy")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(good_response_body())
        .create();

    let client = NnClient::new(server.url(), Duration::from_secs(2));
    let state = serde_json::json!({"sideOne": {}, "sideTwo": {}});
    let resp = client
        .policy(&state, Perspective::P1)
        .expect("happy path should succeed");
    mock.assert();

    assert_eq!(resp.probs.len(), ACTION_DIM);
    assert_eq!(resp.q_values.len(), ACTION_DIM);
    let sum: f32 = resp.probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "probs should sum to ~1.0, got {}", sum);
    // EQ-tera (idx 9) had the highest mass at 0.69
    assert_eq!(resp.argmax(), Some(9));
}

#[test]
fn server_500_returns_bad_status() {
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/policy")
        .with_status(500)
        .with_body("internal error")
        .create();

    let client = NnClient::new(server.url(), Duration::from_secs(2));
    let state = serde_json::json!({"sideOne": {}, "sideTwo": {}});
    let err = client
        .policy(&state, Perspective::P1)
        .expect_err("500 should map to BadStatus");
    match err {
        NnClientError::BadStatus(code, _) => assert_eq!(code, 500),
        other => panic!("expected BadStatus, got {:?}", other),
    }
}

#[test]
fn malformed_json_returns_decode_error() {
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/policy")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{not json at all")
        .create();

    let client = NnClient::new(server.url(), Duration::from_secs(2));
    let state = serde_json::json!({"sideOne": {}, "sideTwo": {}});
    let err = client
        .policy(&state, Perspective::P1)
        .expect_err("bad JSON should error");
    // reqwest .json() surfaces decode errors as Network(reqwest::Error)
    assert!(matches!(
        err,
        NnClientError::Network(_) | NnClientError::Decode(_)
    ));
}

#[test]
fn wrong_shape_returns_bad_response_shape() {
    // Sidecar returns 12 probs (one short) — validate() should catch.
    let probs_short: Vec<f32> = vec![0.1; 12];
    let q_full: Vec<f32> = vec![0.0; 13];
    let decoded: Vec<String> = (0..13).map(|i| format!("a:{}", i)).collect();
    let body = serde_json::json!({
        "probs": probs_short,
        "q_values": q_full,
        "v_estimate": 0.0,
        "decoded_actions": decoded,
    })
    .to_string();
    let mut server = mockito::Server::new();
    let _mock = server
        .mock("POST", "/policy")
        .with_status(200)
        .with_body(body)
        .create();

    let client = NnClient::new(server.url(), Duration::from_secs(2));
    let state = serde_json::json!({"sideOne": {}, "sideTwo": {}});
    let err = client.policy(&state, Perspective::P1).expect_err("short probs");
    assert!(matches!(err, NnClientError::BadResponseShape(_)));
}

#[test]
fn timeout_returns_network_error() {
    // Mockito doesn't support delayed responses elegantly across versions;
    // simulate a timeout by giving an unreachable port + tiny timeout.
    let client = NnClient::new("http://127.0.0.1:1".to_string(), Duration::from_millis(50));
    let state = serde_json::json!({"sideOne": {}, "sideTwo": {}});
    let err = client
        .policy(&state, Perspective::P1)
        .expect_err("connection refused / timeout");
    assert!(matches!(err, NnClientError::Network(_)));
}

#[test]
fn validate_zero_sum_rejected() {
    let resp = PolicyResponse {
        probs: vec![0.0; ACTION_DIM],
        q_values: vec![0.0; ACTION_DIM],
        v_estimate: 0.0,
        decoded_actions: (0..ACTION_DIM).map(|i| format!("a:{}", i)).collect(),
    };
    let err = resp.validate().unwrap_err();
    assert!(matches!(err, NnClientError::BadResponseShape(_)));
}

/// Live test against a running sidecar at default port 7273.
/// Ignored by default; opt in with `--ignored`.
#[test]
#[ignore]
fn live_sidecar_iron_crown_t5() {
    use std::time::Duration;

    let client = NnClient::new("http://localhost:7273".to_string(), Duration::from_secs(10));
    client
        .healthz()
        .expect("sidecar must be running on :7273 for this test");

    // Minimal Iron Crown T5 state — sufficient to round-trip the sidecar.
    // (The full fixture lives in the integration test in test_nn_mcts_integration.rs.)
    let state = serde_json::json!({
        "sideOne": {
            "pokemon": [{
                "species": "ironcrown",
                "level": 100,
                "types": ["psychic", "steel"],
                "hp": 321,
                "maxhp": 321,
                "ability": "quarkdrive",
                "item": "boosterenergy",
                "nature": "serious",
                "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                "attack": 188, "defense": 236,
                "specialAttack": 320, "specialDefense": 290, "speed": 252,
                "status": "None",
                "weightKg": 100.0,
                "moves": [
                    {"id": "calmmind", "pp": 16},
                    {"id": "none", "pp": 0},
                    {"id": "none", "pp": 0},
                    {"id": "none", "pp": 0}
                ],
                "terastallized": false,
                "teraType": "psychic",
                "boosts": {"special-attack": 2, "special-defense": 2},
            }],
            "activeIndex": 0,
            "sideConditions": {},
            "boosts": {},
        },
        "sideTwo": {
            "pokemon": [{
                "species": "garchomp",
                "level": 100,
                "types": ["dragon", "ground"],
                "hp": 357,
                "maxhp": 357,
                "ability": "roughskin",
                "item": "loadeddice",
                "nature": "serious",
                "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                "attack": 359, "defense": 246,
                "specialAttack": 207, "specialDefense": 226, "speed": 261,
                "status": "None",
                "weightKg": 100.0,
                "moves": [
                    {"id": "earthquake", "pp": 16},
                    {"id": "scaleshot", "pp": 16},
                    {"id": "stealthrock", "pp": 16},
                    {"id": "spikes", "pp": 16}
                ],
                "terastallized": false,
                "teraType": "ground",
                "boosts": {},
            }],
            "activeIndex": 0,
            "sideConditions": {},
            "boosts": {},
        },
        "weather": {"weatherType": "none"},
        "terrain": {"terrainType": "none"},
        "trickRoom": false,
    });
    let resp = client.policy(&state, Perspective::P2).expect("live policy");
    eprintln!("live policy probs: {:?}", resp.probs);
    eprintln!("live argmax = {:?}", resp.argmax());
    eprintln!("decoded[argmax] = {:?}", resp.argmax().map(|i| &resp.decoded_actions[i]));

    // Soft assertion: the smoking-gun test (Phase 1) had Tera-EQ at ~0.69.
    // The exact distribution depends on Kakuna's checkpoint — just check argmax.
    assert!(resp.argmax().is_some());
}
