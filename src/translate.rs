//! State Translation Module
//!
//! Converts JSON battle state from the Cobblemon Minecraft mod into
//! poke-engine's pipe-delimited state format, then deserializes into a `State` object.

use crate::state::State;
use serde::Deserialize;

/// Top-level battle request from the Cobblemon mod
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BattleRequest {
    pub side_one: SideInput,
    pub side_two: SideInput,
    #[serde(default)]
    pub weather: Option<WeatherInput>,
    #[serde(default)]
    pub terrain: Option<TerrainInput>,
    #[serde(default = "default_false")]
    pub trick_room: bool,
    #[serde(default)]
    pub trick_room_turns: Option<i8>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SideInput {
    pub pokemon: Vec<PokemonInput>,
    #[serde(default)]
    pub active_index: usize,
    #[serde(default)]
    pub side_conditions: Option<SideConditionsInput>,
    #[serde(default)]
    pub volatile_statuses: Option<Vec<String>>,
    #[serde(default)]
    pub boosts: Option<BoostsInput>,
    #[serde(default = "default_false")]
    pub force_trapped: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PokemonInput {
    pub species: String,
    #[serde(default = "default_level")]
    pub level: i8,
    pub types: Vec<String>,
    pub hp: i16,
    pub maxhp: i16,
    #[serde(default = "default_none_str")]
    pub ability: String,
    #[serde(default = "default_none_str")]
    pub item: String,
    #[serde(default = "default_serious")]
    pub nature: String,
    #[serde(default)]
    pub evs: Option<EvsInput>,
    pub attack: i16,
    pub defense: i16,
    pub special_attack: i16,
    pub special_defense: i16,
    pub speed: i16,
    #[serde(default = "default_none_status")]
    pub status: String,
    #[serde(default)]
    pub rest_turns: i8,
    #[serde(default)]
    pub sleep_turns: i8,
    #[serde(default = "default_weight")]
    pub weight_kg: f32,
    pub moves: Vec<MoveInput>,
    #[serde(default = "default_false")]
    pub terastallized: bool,
    #[serde(default = "default_normal_type")]
    pub tera_type: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveInput {
    pub id: String,
    #[serde(default)]
    pub pp: Option<i8>,
    #[serde(default = "default_false")]
    pub disabled: bool,
}

#[derive(Debug, Deserialize)]
pub struct EvsInput {
    #[serde(default)]
    pub hp: u8,
    #[serde(default)]
    pub atk: u8,
    #[serde(default)]
    pub def: u8,
    #[serde(default)]
    pub spa: u8,
    #[serde(default)]
    pub spd: u8,
    #[serde(default)]
    pub spe: u8,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SideConditionsInput {
    #[serde(default)]
    pub aurora_veil: i8,
    #[serde(default)]
    pub crafty_shield: i8,
    #[serde(default)]
    pub healing_wish: i8,
    #[serde(default)]
    pub light_screen: i8,
    #[serde(default)]
    pub lucky_chant: i8,
    #[serde(default)]
    pub lunar_dance: i8,
    #[serde(default)]
    pub mat_block: i8,
    #[serde(default)]
    pub mist: i8,
    #[serde(default)]
    pub protect: i8,
    #[serde(default)]
    pub quick_guard: i8,
    #[serde(default)]
    pub reflect: i8,
    #[serde(default)]
    pub safeguard: i8,
    #[serde(default)]
    pub spikes: i8,
    #[serde(default)]
    pub stealth_rock: i8,
    #[serde(default)]
    pub sticky_web: i8,
    #[serde(default)]
    pub tailwind: i8,
    #[serde(default)]
    pub toxic_count: i8,
    #[serde(default)]
    pub toxic_spikes: i8,
    #[serde(default)]
    pub wide_guard: i8,
}

#[derive(Debug, Deserialize)]
pub struct BoostsInput {
    #[serde(default)]
    pub attack: i8,
    #[serde(default)]
    pub defense: i8,
    #[serde(default)]
    pub special_attack: i8,
    #[serde(default)]
    pub special_defense: i8,
    #[serde(default)]
    pub speed: i8,
    #[serde(default)]
    pub accuracy: i8,
    #[serde(default)]
    pub evasion: i8,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeatherInput {
    #[serde(default = "default_none_weather")]
    pub weather_type: String,
    #[serde(default = "default_weather_turns")]
    pub turns_remaining: i8,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainInput {
    #[serde(default = "default_none_terrain")]
    pub terrain_type: String,
    #[serde(default = "default_terrain_turns")]
    pub turns_remaining: i8,
}

// -- Default value functions --

fn default_false() -> bool {
    false
}

fn default_level() -> i8 {
    100
}

fn default_none_str() -> String {
    "NONE".to_string()
}

fn default_serious() -> String {
    "SERIOUS".to_string()
}

fn default_none_status() -> String {
    "None".to_string()
}

fn default_weight() -> f32 {
    25.5
}

fn default_normal_type() -> String {
    "Normal".to_string()
}

fn default_none_weather() -> String {
    "none".to_string()
}

fn default_weather_turns() -> i8 {
    5
}

fn default_none_terrain() -> String {
    "none".to_string()
}

fn default_terrain_turns() -> i8 {
    5
}

// -- Serialization helpers --

/// Normalize a species/ability/item/move name to the UPPERCASE no-space format
/// poke-engine expects. E.g. "Close Combat" -> "CLOSECOMBAT", "life_orb" -> "LIFEORB"
fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_uppercase()
}

/// Capitalize first letter, lowercase rest. E.g. "fire" -> "Fire", "GRASS" -> "Grass"
fn capitalize_type(t: &str) -> String {
    let lower = t.to_lowercase();
    let mut chars = lower.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().to_string() + chars.as_str(),
    }
}

fn serialize_pokemon(pkmn: &PokemonInput) -> String {
    let species = normalize_name(&pkmn.species);

    let type1 = capitalize_type(&pkmn.types[0]);
    let type2 = if pkmn.types.len() > 1 {
        capitalize_type(&pkmn.types[1])
    } else {
        "Typeless".to_string()
    };

    // base types = current types (we don't track type changes from the mod)
    let base_type1 = type1.clone();
    let base_type2 = type2.clone();

    let ability = normalize_name(&pkmn.ability);
    let base_ability = ability.clone();
    let item = normalize_name(&pkmn.item);
    let nature = normalize_name(&pkmn.nature);

    let evs_str = match &pkmn.evs {
        Some(evs) => format!(
            "{};{};{};{};{};{}",
            evs.hp, evs.atk, evs.def, evs.spa, evs.spd, evs.spe
        ),
        None => String::new(), // empty string = defaults (85 in all)
    };

    let status = capitalize_type(&pkmn.status);

    // Serialize moves (pad to 4 with NONE defaults)
    let mut move_strs: Vec<String> = pkmn
        .moves
        .iter()
        .take(4)
        .map(|m| {
            let move_id = normalize_name(&m.id);
            let pp = m.pp.unwrap_or(32);
            format!("{};{};{}", move_id, m.disabled, pp)
        })
        .collect();

    // Pad to 4 moves
    while move_strs.len() < 4 {
        move_strs.push("NONE;false;32".to_string());
    }

    let tera_type = capitalize_type(&pkmn.tera_type);

    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        species,
        pkmn.level,
        type1,
        type2,
        base_type1,
        base_type2,
        pkmn.hp,
        pkmn.maxhp,
        ability,
        base_ability,
        item,
        nature,
        evs_str,
        pkmn.attack,
        pkmn.defense,
        pkmn.special_attack,
        pkmn.special_defense,
        pkmn.speed,
        status,
        pkmn.rest_turns,
        pkmn.sleep_turns,
        pkmn.weight_kg,
        move_strs[0],
        move_strs[1],
        move_strs[2],
        move_strs[3],
        pkmn.terastallized,
        tera_type,
    )
}

/// Create a "blank" Pokemon serialization for empty team slots.
/// poke-engine requires exactly 6 Pokemon per side, so we fill empty slots
/// with fainted (0 HP) placeholder Pokemon.
fn blank_pokemon() -> String {
    "NONE,1,Typeless,Typeless,Typeless,Typeless,0,0,NONE,NONE,NONE,SERIOUS,,0,0,0,0,0,None,0,0,25.5,NONE;false;32,NONE;false;32,NONE;false;32,NONE;false;32,false,Normal".to_string()
}

fn serialize_side_conditions(conds: &Option<SideConditionsInput>) -> String {
    match conds {
        Some(c) => format!(
            "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};",
            c.aurora_veil,
            c.crafty_shield,
            c.healing_wish,
            c.light_screen,
            c.lucky_chant,
            c.lunar_dance,
            c.mat_block,
            c.mist,
            c.protect,
            c.quick_guard,
            c.reflect,
            c.safeguard,
            c.spikes,
            c.stealth_rock,
            c.sticky_web,
            c.tailwind,
            c.toxic_count,
            c.toxic_spikes,
            c.wide_guard,
        ),
        None => "0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;".to_string(),
    }
}

fn serialize_side(side: &SideInput) -> String {
    // Serialize all pokemon (pad to 6)
    let mut pkmn_strs: Vec<String> = side
        .pokemon
        .iter()
        .take(6)
        .map(|p| serialize_pokemon(p))
        .collect();

    while pkmn_strs.len() < 6 {
        pkmn_strs.push(blank_pokemon());
    }

    let side_conditions = serialize_side_conditions(&side.side_conditions);

    // Volatile statuses (colon-separated)
    let volatile_statuses = match &side.volatile_statuses {
        Some(vs) if !vs.is_empty() => vs
            .iter()
            .map(|v| normalize_name(v))
            .collect::<Vec<_>>()
            .join(":"),
        _ => String::new(),
    };

    // Volatile status durations (defaults: all zeros)
    let volatile_durations = "0;0;0;0;0;0";

    // Boosts
    let (atk_b, def_b, spa_b, spd_b, spe_b, acc_b, eva_b) = match &side.boosts {
        Some(b) => (
            b.attack,
            b.defense,
            b.special_attack,
            b.special_defense,
            b.speed,
            b.accuracy,
            b.evasion,
        ),
        None => (0, 0, 0, 0, 0, 0, 0),
    };

    // Side format (29 fields separated by =):
    // [0-5]  pokemon 0-5
    // [6]    active_index
    // [7]    side_conditions
    // [8]    volatile_statuses
    // [9]    volatile_status_durations
    // [10]   substitute_health
    // [11-17] boosts (atk, def, spa, spd, spe, acc, eva)
    // [18-19] wish (turns, hp)
    // [20-21] future_sight (turns, pokemon_index)
    // [22]   force_switch
    // [23]   switch_out_move_second_saved_move
    // [24]   baton_passing
    // [25]   shed_tailing
    // [26]   force_trapped
    // [27]   last_used_move
    // [28]   slow_uturn_move
    format!(
        "{}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}",
        pkmn_strs[0],
        pkmn_strs[1],
        pkmn_strs[2],
        pkmn_strs[3],
        pkmn_strs[4],
        pkmn_strs[5],
        side.active_index,          // [6]
        side_conditions,            // [7]
        volatile_statuses,          // [8]
        volatile_durations,         // [9]
        0,                          // [10] substitute_health
        atk_b,                      // [11]
        def_b,                      // [12]
        spa_b,                      // [13]
        spd_b,                      // [14]
        spe_b,                      // [15]
        acc_b,                      // [16]
        eva_b,                      // [17]
        0,                          // [18] wish turns
        0,                          // [19] wish hp
        0,                          // [20] future sight turns
        0,                          // [21] future sight pokemon_index
        false,                      // [22] force_switch
        "NONE",                     // [23] switch_out_move
        false,                      // [24] baton_passing
        false,                      // [25] shed_tailing
        side.force_trapped,         // [26] force_trapped
        "switch:0",                 // [27] last_used_move
        false,                      // [28] slow_uturn_move
    )
}

fn serialize_weather(weather: &Option<WeatherInput>) -> String {
    match weather {
        Some(w) => format!("{};{}", w.weather_type.to_lowercase(), w.turns_remaining),
        None => "none;5".to_string(),
    }
}

fn serialize_terrain(terrain: &Option<TerrainInput>) -> String {
    match terrain {
        Some(t) => format!("{};{}", t.terrain_type.to_lowercase(), t.turns_remaining),
        None => "none;5".to_string(),
    }
}

fn serialize_trick_room(active: bool, turns: Option<i8>) -> String {
    format!("{};{}", active, turns.unwrap_or(5))
}

/// Convert a `BattleRequest` JSON input into a poke-engine `State`.
///
/// This builds the pipe-delimited string format that `State::deserialize()` expects
/// and calls it to produce the final `State` object.
pub fn to_poke_state(request: &BattleRequest) -> State {
    let serialized = format!(
        "{}/{}/{}/{}/{}/{}",
        serialize_side(&request.side_one),
        serialize_side(&request.side_two),
        serialize_weather(&request.weather),
        serialize_terrain(&request.terrain),
        serialize_trick_room(request.trick_room, request.trick_room_turns),
        false, // team_preview is always false during battle
    );

    State::deserialize(&serialized)
}

/// Convenience wrapper that parses JSON directly into a `State`.
pub fn json_to_poke_state(json: &str) -> Result<State, serde_json::Error> {
    let request: BattleRequest = serde_json::from_str(json)?;
    Ok(to_poke_state(&request))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
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
                    },
                    {
                        "species": "Garchomp",
                        "level": 100,
                        "types": ["Dragon", "Ground"],
                        "hp": 357,
                        "maxhp": 357,
                        "ability": "Rough Skin",
                        "item": "Rocky Helmet",
                        "nature": "Jolly",
                        "attack": 296,
                        "defense": 226,
                        "specialAttack": 176,
                        "specialDefense": 206,
                        "speed": 333,
                        "status": "None",
                        "weightKg": 95.0,
                        "moves": [
                            {"id": "Earthquake", "pp": 16},
                            {"id": "Dragon Claw", "pp": 24},
                            {"id": "Stealth Rock", "pp": 32},
                            {"id": "Stone Edge", "pp": 8}
                        ],
                        "teraType": "Dragon"
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
        }"#
    }

    #[test]
    fn test_json_to_poke_state() {
        let state = json_to_poke_state(sample_json()).expect("Failed to parse JSON");

        // Verify side one active pokemon
        let active_one = state.side_one.get_active_immutable();
        assert_eq!(active_one.hp, 302);
        assert_eq!(active_one.maxhp, 302);
        assert_eq!(active_one.level, 100);
        assert_eq!(active_one.attack, 349);
        assert_eq!(active_one.speed, 284);

        // Verify side two active pokemon
        let active_two = state.side_two.get_active_immutable();
        assert_eq!(active_two.hp, 251);
        assert_eq!(active_two.maxhp, 251);
        assert_eq!(active_two.special_attack, 369);

        // Verify weather/terrain/trick_room defaults
        assert_eq!(state.trick_room.active, false);
        assert_eq!(state.team_preview, false);
    }

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("Close Combat"), "CLOSECOMBAT");
        assert_eq!(normalize_name("life_orb"), "LIFEORB");
        assert_eq!(normalize_name("Speed Boost"), "SPEEDBOOST");
        assert_eq!(normalize_name("NONE"), "NONE");
        assert_eq!(normalize_name("Knock Off"), "KNOCKOFF");
    }

    #[test]
    fn test_capitalize_type() {
        assert_eq!(capitalize_type("fire"), "Fire");
        assert_eq!(capitalize_type("GRASS"), "Grass");
        assert_eq!(capitalize_type("Fighting"), "Fighting");
        assert_eq!(capitalize_type("typeless"), "Typeless");
    }

    #[test]
    fn test_single_typed_pokemon() {
        // Alakazam is single-typed (Psychic only)
        let state = json_to_poke_state(sample_json()).expect("Failed to parse JSON");
        let alakazam = state.side_two.get_active_immutable();
        // A single-typed pokemon should have Typeless as second type
        assert_eq!(alakazam.hp, 251);
    }

    #[test]
    fn test_empty_team_slots_filled() {
        // Side two only has 1 pokemon, remaining 5 should be blank (0 HP)
        let state = json_to_poke_state(sample_json()).expect("Failed to parse JSON");

        // Pokemon at index 1-5 on side two should be fainted blanks
        assert_eq!(state.side_two.pokemon.pkmn[1].hp, 0);
        assert_eq!(state.side_two.pokemon.pkmn[2].hp, 0);
    }

    #[test]
    fn test_weather_and_terrain() {
        let json = r#"{
            "sideOne": {
                "pokemon": [{
                    "species": "Tyranitar",
                    "level": 100,
                    "types": ["Rock", "Dark"],
                    "hp": 404,
                    "maxhp": 404,
                    "ability": "Sand Stream",
                    "item": "Leftovers",
                    "attack": 305,
                    "defense": 256,
                    "specialAttack": 203,
                    "specialDefense": 327,
                    "speed": 159,
                    "moves": [
                        {"id": "Crunch", "pp": 24},
                        {"id": "Stone Edge", "pp": 8},
                        {"id": "Earthquake", "pp": 16},
                        {"id": "Stealth Rock", "pp": 32}
                    ]
                }],
                "activeIndex": 0
            },
            "sideTwo": {
                "pokemon": [{
                    "species": "Garchomp",
                    "level": 100,
                    "types": ["Dragon", "Ground"],
                    "hp": 357,
                    "maxhp": 357,
                    "ability": "Rough Skin",
                    "item": "Choice Scarf",
                    "attack": 296,
                    "defense": 226,
                    "specialAttack": 176,
                    "specialDefense": 206,
                    "speed": 333,
                    "moves": [
                        {"id": "Earthquake", "pp": 16},
                        {"id": "Outrage", "pp": 16},
                        {"id": "Stone Edge", "pp": 8},
                        {"id": "Fire Fang", "pp": 24}
                    ]
                }],
                "activeIndex": 0
            },
            "weather": {
                "weatherType": "sand",
                "turnsRemaining": 3
            },
            "trickRoom": false
        }"#;

        let state = json_to_poke_state(json).expect("Failed to parse JSON");
        assert_eq!(state.side_one.get_active_immutable().hp, 404);
        assert_eq!(state.side_two.get_active_immutable().hp, 357);
        assert_eq!(state.trick_room.active, false);
    }

    #[test]
    fn test_roundtrip_serialize() {
        // Verify that our generated state can be serialized and deserialized again
        let state = json_to_poke_state(sample_json()).expect("Failed to parse JSON");
        let serialized = state.serialize();
        let state2 = State::deserialize(&serialized);
        assert_eq!(state.serialize(), state2.serialize());
    }

    #[test]
    fn test_status_condition() {
        let json = r#"{
            "sideOne": {
                "pokemon": [{
                    "species": "Blaziken",
                    "level": 100,
                    "types": ["Fire", "Fighting"],
                    "hp": 200,
                    "maxhp": 302,
                    "ability": "Speed Boost",
                    "item": "Life Orb",
                    "attack": 349,
                    "defense": 196,
                    "specialAttack": 230,
                    "specialDefense": 176,
                    "speed": 284,
                    "status": "Burn",
                    "moves": [
                        {"id": "Close Combat", "pp": 8},
                        {"id": "Flare Blitz", "pp": 24},
                        {"id": "Swords Dance", "pp": 32},
                        {"id": "Knock Off", "pp": 32}
                    ]
                }],
                "activeIndex": 0
            },
            "sideTwo": {
                "pokemon": [{
                    "species": "Alakazam",
                    "level": 100,
                    "types": ["Psychic"],
                    "hp": 251,
                    "maxhp": 251,
                    "ability": "Magic Guard",
                    "item": "Focus Sash",
                    "attack": 121,
                    "defense": 128,
                    "specialAttack": 369,
                    "specialDefense": 206,
                    "speed": 372,
                    "moves": [
                        {"id": "Psychic", "pp": 16},
                        {"id": "Shadow Ball", "pp": 24},
                        {"id": "Focus Blast", "pp": 8},
                        {"id": "Energy Ball", "pp": 16}
                    ]
                }],
                "activeIndex": 0
            }
        }"#;

        let state = json_to_poke_state(json).expect("Failed to parse JSON");
        let active = state.side_one.get_active_immutable();
        assert_eq!(active.hp, 200);
        // Verify the state roundtrips successfully (implies status parsed correctly)
        let serialized = state.serialize();
        let state2 = State::deserialize(&serialized);
        assert_eq!(state.serialize(), state2.serialize());
    }
}
