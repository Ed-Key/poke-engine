//! Plan E state-encoder + action-index-mapping tests.
//!
//! Eight tests for the encoder + mapping. Together with
//! `tests/test_nn_client.rs` they cover the load-bearing path: any silent
//! drift in the state-encoder or the alphabetical permutation would corrupt
//! every NN-driven recommendation.

use poke_engine::engine::state::MoveChoice;
use poke_engine::nn_client::ACTION_DIM;
use poke_engine::nn_state_encoder::{
    encode, map_policy_to_options, move_name_norm, pokemon_name_norm, SidePerspective,
};
use poke_engine::translate::auto_detect_and_parse;

const IRON_CROWN_T5_FIXTURE: &str = include_str!("fixtures/iron_crown_t5.json");

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_round_trip_top_level_keys() {
    // Fixture JSON → State → encode_state → JSON. Top-level keys must match.
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    let encoded = encode(&state, SidePerspective::Side1);
    let obj = encoded.as_object().expect("top-level object");
    for k in ["sideOne", "sideTwo", "weather", "terrain", "trickRoom"] {
        assert!(obj.contains_key(k), "missing top-level key: {}", k);
    }
    // Confirm pokemon arrays both have 6 entries.
    assert_eq!(obj["sideOne"]["pokemon"].as_array().unwrap().len(), 6);
    assert_eq!(obj["sideTwo"]["pokemon"].as_array().unwrap().len(), 6);
}

#[test]
fn encode_active_index_preserved() {
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    let encoded = encode(&state, SidePerspective::Side1);
    assert_eq!(encoded["sideOne"]["activeIndex"].as_u64(), Some(0));
    assert_eq!(encoded["sideTwo"]["activeIndex"].as_u64(), Some(0));
}

#[test]
fn encode_active_pokemon_species_preserved() {
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    let encoded = encode(&state, SidePerspective::Side1);
    assert_eq!(
        encoded["sideOne"]["pokemon"][0]["species"].as_str(),
        Some("ironcrown")
    );
    assert_eq!(
        encoded["sideTwo"]["pokemon"][0]["species"].as_str(),
        Some("garchomp")
    );
}

#[test]
fn encode_garchomp_moves_preserved() {
    // Garchomp's 4 moves in slot order: earthquake, scaleshot, stealthrock, spikes.
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    let encoded = encode(&state, SidePerspective::Side1);
    let moves = encoded["sideTwo"]["pokemon"][0]["moves"].as_array().unwrap();
    let ids: Vec<&str> = moves.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["earthquake", "scaleshot", "stealthrock", "spikes"]);
}

#[test]
fn encode_side_level_boosts_round_trip() {
    // `BoostsInput` in translate.rs is keyed at the side level. Build a
    // minimal BattleRequest with side-level boosts and confirm they survive
    // the State -> JSON round-trip.
    let json = r#"{
        "sideOne": {
            "pokemon": [{
                "species": "Garchomp", "level": 100,
                "types": ["Dragon", "Ground"],
                "hp": 357, "maxhp": 357,
                "ability": "RoughSkin", "item": "LoadedDice",
                "nature": "Serious",
                "attack": 359, "defense": 246,
                "specialAttack": 207, "specialDefense": 226, "speed": 261,
                "status": "None", "weightKg": 100.0,
                "moves": [
                    {"id": "Earthquake", "pp": 16},
                    {"id": "ScaleShot", "pp": 16},
                    {"id": "StealthRock", "pp": 16},
                    {"id": "Spikes", "pp": 16}
                ],
                "teraType": "Ground"
            }],
            "activeIndex": 0,
            "boosts": {"attack": 2, "speed": 1}
        },
        "sideTwo": {
            "pokemon": [{
                "species": "Tapulele", "level": 100,
                "types": ["Fairy", "Psychic"],
                "hp": 281, "maxhp": 281,
                "ability": "PsychicSurge", "item": "ChoiceSpecs",
                "nature": "Serious",
                "attack": 207, "defense": 196,
                "specialAttack": 333, "specialDefense": 287, "speed": 251,
                "status": "None", "weightKg": 100.0,
                "moves": [
                    {"id": "Psychic", "pp": 16},
                    {"id": "Moonblast", "pp": 16},
                    {"id": "FocusBlast", "pp": 16},
                    {"id": "Psyshock", "pp": 16}
                ],
                "teraType": "Fairy"
            }],
            "activeIndex": 0
        }
    }"#;
    let state = auto_detect_and_parse(json).expect("parse boosted state");
    let encoded = encode(&state, SidePerspective::Side1);

    // Side-level boosts: encoder's `boosts` block on the side.
    let boosts = &encoded["sideOne"]["boosts"];
    assert_eq!(boosts["attack"].as_i64(), Some(2));
    assert_eq!(boosts["speed"].as_i64(), Some(1));

    // Per-active-mon boosts: also present on the active mon dict (sidecar
    // format). Non-zero entries only.
    let active = &encoded["sideOne"]["pokemon"][0];
    assert_eq!(active["boosts"]["attack"].as_i64(), Some(2));
    assert_eq!(active["boosts"]["speed"].as_i64(), Some(1));
    // Reserve mons get empty boosts.
    // (only one mon in this fixture; nothing to check for reserves.)
}

#[test]
fn encode_weather_terrain_default_none() {
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    let encoded = encode(&state, SidePerspective::Side1);
    assert_eq!(encoded["weather"]["weatherType"].as_str(), Some("none"));
    assert_eq!(encoded["terrain"]["terrainType"].as_str(), Some("none"));
    assert_eq!(encoded["trickRoom"].as_bool(), Some(false));
}

// ---------------------------------------------------------------------------
// Action-index mapping
// ---------------------------------------------------------------------------

#[test]
fn map_policy_iron_crown_t5_garchomp_moves_alphabetical() {
    // Garchomp on Side2 — perspective Side2 is Mariga's POV. Slot order:
    //   M0=earthquake, M1=scaleshot, M2=stealthrock, M3=spikes
    // Alphabetical: earthquake(0), scaleshot(1), spikes(2), stealthrock(3)
    //
    // Build a fake probs vector where:
    //   probs[0] (alpha 0 = earthquake) = 0.27
    //   probs[1] (alpha 1 = scaleshot)  = 0.001
    //   probs[2] (alpha 2 = spikes)     = 0.001
    //   probs[3] (alpha 3 = stealthrock)= 0.001
    //   probs[9] (tera-EQ)              = 0.69
    //   probs[10] (tera-scaleshot)      = 0.001
    //   probs[11] (tera-spikes)         = 0.001
    //   probs[12] (tera-SR)             = 0.001
    //   probs[4..9] (switches)          = small leftover
    let mut probs = vec![0.0_f32; ACTION_DIM];
    probs[0] = 0.27; // EQ
    probs[1] = 0.001; // ScaleShot
    probs[2] = 0.001; // Spikes (alpha-pos)
    probs[3] = 0.001; // SR (alpha-pos)
    probs[4] = 0.005; // Switch slot 0 (Tapu Lele alpha first)
    probs[5] = 0.005;
    probs[6] = 0.005;
    probs[7] = 0.005;
    probs[8] = 0.005;
    probs[9] = 0.69; // tera-EQ
    probs[10] = 0.001;
    probs[11] = 0.001;
    probs[12] = 0.001;

    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    // Build options in slot order: M0..M3, MoveTera M0..M3, Switch P1..P5
    use poke_engine::state::{PokemonIndex, PokemonMoveIndex};
    let opts = vec![
        MoveChoice::Move(PokemonMoveIndex::M0),       // EQ
        MoveChoice::Move(PokemonMoveIndex::M1),       // ScaleShot
        MoveChoice::Move(PokemonMoveIndex::M2),       // SR
        MoveChoice::Move(PokemonMoveIndex::M3),       // Spikes
        MoveChoice::MoveTera(PokemonMoveIndex::M0),   // tera-EQ
        MoveChoice::MoveTera(PokemonMoveIndex::M1),   // tera-ScaleShot
        MoveChoice::MoveTera(PokemonMoveIndex::M2),   // tera-SR
        MoveChoice::MoveTera(PokemonMoveIndex::M3),   // tera-Spikes
        MoveChoice::Switch(PokemonIndex::P1),         // Tapu Lele (reserve slot 0)
    ];
    let priors = map_policy_to_options(&probs, &state, SidePerspective::Side2, &opts);

    // Renormalized; total mass is 0.27+0.001*3+0.005*5+0.69+0.001*3 = ~0.991
    // After renormalization, EQ (~0.272) and tera-EQ (~0.696) are biggest.
    assert_eq!(priors.len(), opts.len());
    let sum: f32 = priors.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "priors should sum to ~1.0, got {}", sum);

    // Tera-EQ (index 4 in opts) should be largest.
    let argmax = priors
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax, 4, "tera-EQ should win (priors={:?})", priors);

    // EQ (index 0 in opts) should be #2.
    let mut sorted: Vec<(usize, f32)> = priors.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(sorted[0].0, 4, "first place: tera-EQ");
    assert_eq!(sorted[1].0, 0, "second place: EQ");

    // SR (index 2 in opts; alpha pos 3) should be near-zero relative to EQ.
    let sr_share = priors[2];
    let eq_share = priors[0];
    assert!(
        eq_share > 100.0 * sr_share,
        "EQ should dwarf SR (eq={}, sr={})",
        eq_share,
        sr_share
    );
}

#[test]
fn map_policy_zero_sum_falls_back_to_uniform() {
    let probs = vec![0.0_f32; ACTION_DIM];
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    use poke_engine::state::PokemonMoveIndex;
    let opts = vec![
        MoveChoice::Move(PokemonMoveIndex::M0),
        MoveChoice::Move(PokemonMoveIndex::M1),
    ];
    let priors = map_policy_to_options(&probs, &state, SidePerspective::Side2, &opts);
    assert_eq!(priors.len(), 2);
    // All zeros → uniform 0.5/0.5.
    for p in &priors {
        assert!((p - 0.5).abs() < 1e-5, "uniform expected, got {:?}", priors);
    }
}

#[test]
fn map_policy_wrong_dim_falls_back_to_uniform() {
    // Probs has wrong length → uniform.
    let probs = vec![1.0_f32; 5];
    let state = auto_detect_and_parse(IRON_CROWN_T5_FIXTURE).expect("parse fixture");
    use poke_engine::state::PokemonMoveIndex;
    let opts = vec![
        MoveChoice::Move(PokemonMoveIndex::M0),
        MoveChoice::Move(PokemonMoveIndex::M1),
        MoveChoice::Move(PokemonMoveIndex::M2),
        MoveChoice::Move(PokemonMoveIndex::M3),
    ];
    let priors = map_policy_to_options(&probs, &state, SidePerspective::Side2, &opts);
    assert_eq!(priors.len(), 4);
    for p in &priors {
        assert!((p - 0.25).abs() < 1e-5);
    }
}

#[test]
fn norm_helpers_match_metamon_semantics() {
    // metamon's clean_no_numbers → alpha-only, lower
    assert_eq!(move_name_norm("Hidden Power 70"), "hiddenpower");
    assert_eq!(move_name_norm("U-turn"), "uturn");
    // metamon's clean_name → alphanumeric, lower
    assert_eq!(pokemon_name_norm("Iron Crown"), "ironcrown");
    assert_eq!(pokemon_name_norm("Porygon-Z"), "porygonz");
    assert_eq!(pokemon_name_norm("Kommo-o"), "kommoo");
}
