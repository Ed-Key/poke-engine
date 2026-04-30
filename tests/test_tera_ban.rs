//! Format-level Tera ban suppresses MoveTera at root.
#![cfg(feature = "terastallization")]

use poke_engine::engine::state::MoveChoice;
use poke_engine::translate::json_to_poke_state;

const SAMPLE_JSON_TEMPLATE: &str = r#"{
    "sideOne": {
        "pokemon": [{
            "species": "Dragapult",
            "level": 100,
            "types": ["Dragon", "Ghost"],
            "hp": 297, "maxhp": 297,
            "ability": "Clear Body", "item": "Choice Specs", "nature": "Timid",
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            "attack": 200, "defense": 200, "specialAttack": 350, "specialDefense": 200, "speed": 421,
            "status": "None", "weightKg": 50.0,
            "moves": [
                {"id": "Draco Meteor", "pp": 8},
                {"id": "Shadow Ball", "pp": 24},
                {"id": "Flamethrower", "pp": 24},
                {"id": "U-turn", "pp": 32}
            ],
            "teraType": "Fairy"
        }],
        "activeIndex": 0
    },
    "sideTwo": {
        "pokemon": [{
            "species": "Tyranitar",
            "level": 100,
            "types": ["Rock", "Dark"],
            "hp": 341, "maxhp": 341,
            "ability": "Sand Stream", "item": "Choice Band", "nature": "Adamant",
            "evs": {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252},
            "attack": 403, "defense": 256, "specialAttack": 200, "specialDefense": 286, "speed": 243,
            "status": "None", "weightKg": 202.0,
            "moves": [
                {"id": "Crunch", "pp": 24},
                {"id": "Stone Edge", "pp": 8},
                {"id": "Earthquake", "pp": 16},
                {"id": "Pursuit", "pp": 32}
            ],
            "teraType": "Dark"
        }],
        "activeIndex": 0
    },
    "weather": {"weatherType": "none", "turnsRemaining": -1},
    "terrain": {"terrainType": "none", "turnsRemaining": -1},
    "trickRoom": false__TERA_BANNED__
}"#;

fn build(tera_banned: Option<bool>) -> poke_engine::state::State {
    let placeholder = match tera_banned {
        Some(b) => format!(", \"teraBanned\": {}", b),
        None => "".to_string(),
    };
    let json = SAMPLE_JSON_TEMPLATE.replace("__TERA_BANNED__", &placeholder);
    json_to_poke_state(&json).expect("parse")
}

#[test]
fn tera_actions_default_present() {
    // Default (no field) → field defaults to false → tera options exist.
    let state = build(None);
    let (s1_opts, s2_opts) = state.root_get_all_options();
    let s1_has_tera = s1_opts.iter().any(|o| matches!(o, MoveChoice::MoveTera(_)));
    let s2_has_tera = s2_opts.iter().any(|o| matches!(o, MoveChoice::MoveTera(_)));
    assert!(s1_has_tera, "side_one should have MoveTera options when not banned");
    assert!(s2_has_tera, "side_two should have MoveTera options when not banned");
    assert!(!state.tera_banned);
}

#[test]
fn tera_ban_suppresses_tera_actions() {
    let state = build(Some(true));
    assert!(state.tera_banned);
    let (s1_opts, s2_opts) = state.root_get_all_options();
    let s1_has_tera = s1_opts.iter().any(|o| matches!(o, MoveChoice::MoveTera(_)));
    let s2_has_tera = s2_opts.iter().any(|o| matches!(o, MoveChoice::MoveTera(_)));
    assert!(!s1_has_tera, "side_one should have NO MoveTera options when banned");
    assert!(!s2_has_tera, "side_two should have NO MoveTera options when banned");
    // Regular moves and switches must still be there.
    assert!(s1_opts.iter().any(|o| matches!(o, MoveChoice::Move(_))));
    assert!(s2_opts.iter().any(|o| matches!(o, MoveChoice::Move(_))));
}

#[test]
fn tera_ban_false_explicit_keeps_actions() {
    let state = build(Some(false));
    assert!(!state.tera_banned);
    let (s1_opts, _) = state.root_get_all_options();
    let s1_has_tera = s1_opts.iter().any(|o| matches!(o, MoveChoice::MoveTera(_)));
    assert!(s1_has_tera);
}
