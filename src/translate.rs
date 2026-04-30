//! State Translation Module
//!
//! Converts JSON battle state from the Cobblemon Minecraft mod into
//! poke-engine's pipe-delimited state format, then deserializes into a `State` object.
//!
//! Supports two input formats:
//! 1. **BattleRequest** — the original format with `sideOne`/`sideTwo` objects
//! 2. **FabricAuxState** — the fabric-aux `/api/battle-state` format with a `sides` array

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
    /// Format-level Tera ban (e.g. NatDex OU 2026). When true, MoveTera
    /// actions are excluded from search.
    #[serde(default = "default_false")]
    pub tera_banned: bool,
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
    #[serde(default = "default_false")]
    pub force_switch: bool,
    /// Substitute HP for the active Pokemon. When the SUBSTITUTE
    /// volatile_status is set but this is None/0, the engine cannot
    /// model damage absorption correctly. Showdown-side proxies should
    /// derive this as `active.maxhp / 4` when the sub volatile is
    /// present (standard sub HP at creation).
    #[serde(default)]
    pub substitute_health: Option<i16>,
    /// Last move used by the active Pokemon, in poke-engine's
    /// `LastUsedMove::deserialize` format: `"move:<idx>"` for an active
    /// move at index 0..3, `"switch:<idx>"` for a just-came-in switch,
    /// or `"move:none"` for "nothing previously". Without this, choice-
    /// locked opponents (Scarf Urshifu locked into Surging Strikes, CB
    /// Dragonite locked into Outrage) are evaluated as if all 4 moves
    /// are usable. Engine reads via `Side.last_used_move` (state.rs:1010).
    #[serde(default)]
    pub last_used_move: Option<String>,
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

#[derive(Debug, Clone, Deserialize)]
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

// -- Fabric-aux format structs --

/// Top-level battle state from the fabric-aux `/api/battle-state` endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FabricAuxState {
    #[serde(default)]
    pub turn: u32,
    pub weather: Option<String>,
    pub terrain: Option<String>,
    pub sides: Vec<FabricAuxSide>,
}

/// One side (p1 or p2) from fabric-aux.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FabricAuxSide {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub pokemon: Vec<FabricAuxPokemon>,
}

/// A single Pokemon from the fabric-aux state.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FabricAuxPokemon {
    pub species: String,
    #[serde(default = "default_level")]
    pub level: i8,
    pub hp: i16,
    pub maxhp: i16,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub item: Option<String>,
    #[serde(default = "default_none_str")]
    pub ability: String,
    #[serde(default)]
    pub base_ability: Option<String>,
    pub types: Vec<String>,
    pub stats: FabricAuxStats,
    #[serde(default)]
    pub ivs: Option<EvsInput>, // Reuse EvsInput since the shape is identical
    #[serde(default)]
    pub evs: Option<EvsInput>,
    #[serde(default = "default_serious")]
    pub nature: String,
    #[serde(default)]
    pub gender: Option<String>,
    #[serde(default)]
    pub boosts: Option<FabricAuxBoosts>,
    pub moves: Vec<FabricAuxMove>,
    #[serde(default)]
    pub fainted: bool,
    #[serde(default)]
    pub is_active: bool,
    #[serde(default = "default_weight_hectograms")]
    pub weight: f32,
}

/// Stat block from fabric-aux (computed stats, not base stats).
#[derive(Debug, Deserialize)]
pub struct FabricAuxStats {
    pub atk: i16,
    pub def: i16,
    pub spa: i16,
    pub spd: i16,
    pub spe: i16,
}

/// Boosts from fabric-aux (per-pokemon).
#[derive(Debug, Deserialize)]
pub struct FabricAuxBoosts {
    #[serde(default)]
    pub atk: i8,
    #[serde(default)]
    pub def: i8,
    #[serde(default)]
    pub spa: i8,
    #[serde(default)]
    pub spd: i8,
    #[serde(default)]
    pub spe: i8,
    #[serde(default)]
    pub accuracy: i8,
    #[serde(default)]
    pub evasion: i8,
}

/// A move from fabric-aux state.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FabricAuxMove {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub pp: Option<i8>,
    #[serde(default)]
    pub maxpp: Option<i8>,
    #[serde(default)]
    pub disabled: Option<bool>,
    #[serde(default)]
    pub base_power: Option<i16>,
}

fn default_weight_hectograms() -> f32 {
    255.0 // 25.5 kg in hectograms
}

// -- Fabric-aux conversion --

/// Convert a `FabricAuxPokemon` into a `PokemonInput` so we can reuse existing
/// serialization logic.
fn fabric_aux_pokemon_to_input(pkmn: &FabricAuxPokemon) -> PokemonInput {
    // fabric-aux weight is in hectograms (e.g. 520 = 52.0 kg)
    let weight_kg = pkmn.weight / 10.0;

    let hp = if pkmn.fainted { 0 } else { pkmn.hp };

    let status = match &pkmn.status {
        Some(s) if !s.is_empty() => s.clone(),
        _ => "None".to_string(),
    };

    let item = match &pkmn.item {
        Some(i) if !i.is_empty() => i.clone(),
        _ => "NONE".to_string(),
    };

    let moves: Vec<MoveInput> = pkmn
        .moves
        .iter()
        .map(|m| MoveInput {
            id: m.id.clone(),
            pp: m.pp,
            disabled: m.disabled.unwrap_or(false),
        })
        .collect();

    PokemonInput {
        species: pkmn.species.clone(),
        level: pkmn.level,
        types: pkmn.types.clone(),
        hp,
        maxhp: pkmn.maxhp,
        ability: pkmn.ability.clone(),
        item,
        nature: pkmn.nature.clone(),
        evs: pkmn.evs.clone(),
        attack: pkmn.stats.atk,
        defense: pkmn.stats.def,
        special_attack: pkmn.stats.spa,
        special_defense: pkmn.stats.spd,
        speed: pkmn.stats.spe,
        status,
        rest_turns: 0,
        sleep_turns: 0,
        weight_kg,
        moves,
        terastallized: false,
        tera_type: "Normal".to_string(),
    }
}

/// Convert a `FabricAuxSide` into a `SideInput`, determining the active index
/// from the `isActive` flag on each Pokemon.
fn fabric_aux_side_to_input(side: &FabricAuxSide) -> SideInput {
    let active_index = side
        .pokemon
        .iter()
        .position(|p| p.is_active)
        .unwrap_or(0);

    let pokemon: Vec<PokemonInput> = side
        .pokemon
        .iter()
        .map(|p| fabric_aux_pokemon_to_input(p))
        .collect();

    // Use the ACTIVE Pokemon's boosts for the side boosts
    let boosts = side
        .pokemon
        .get(active_index)
        .and_then(|p| p.boosts.as_ref())
        .map(|b| BoostsInput {
            attack: b.atk,
            defense: b.def,
            special_attack: b.spa,
            special_defense: b.spd,
            speed: b.spe,
            accuracy: b.accuracy,
            evasion: b.evasion,
        });

    SideInput {
        pokemon,
        active_index,
        side_conditions: None,
        volatile_statuses: None,
        boosts,
        force_trapped: false,
        force_switch: false,
        substitute_health: None,
        last_used_move: None,
    }
}

/// Convert a `FabricAuxState` into a poke-engine `State`.
///
/// sides[0] (p1) becomes side_one, sides[1] (p2) becomes side_two.
pub fn from_fabric_aux(state: &FabricAuxState) -> State {
    let side_one = if !state.sides.is_empty() {
        fabric_aux_side_to_input(&state.sides[0])
    } else {
        SideInput {
            pokemon: vec![],
            active_index: 0,
            side_conditions: None,
            volatile_statuses: None,
            boosts: None,
            force_trapped: false,
            force_switch: false,
            substitute_health: None,
            last_used_move: None,
        }
    };

    let side_two = if state.sides.len() > 1 {
        fabric_aux_side_to_input(&state.sides[1])
    } else {
        SideInput {
            pokemon: vec![],
            active_index: 0,
            side_conditions: None,
            volatile_statuses: None,
            boosts: None,
            force_trapped: false,
            force_switch: false,
            substitute_health: None,
            last_used_move: None,
        }
    };

    // Convert weather string to WeatherInput
    let weather = state.weather.as_ref().and_then(|w| {
        if w.is_empty() || w == "none" {
            None
        } else {
            Some(WeatherInput {
                weather_type: w.clone(),
                turns_remaining: default_weather_turns(),
            })
        }
    });

    // Convert terrain string to TerrainInput
    let terrain = state.terrain.as_ref().and_then(|t| {
        if t.is_empty() || t == "none" {
            None
        } else {
            Some(TerrainInput {
                terrain_type: t.clone(),
                turns_remaining: default_terrain_turns(),
            })
        }
    });

    let request = BattleRequest {
        side_one,
        side_two,
        weather,
        terrain,
        trick_room: false,
        trick_room_turns: None,
        tera_banned: false,
    };

    to_poke_state(&request)
}

/// Parse fabric-aux JSON directly into a `State`.
pub fn fabric_aux_json_to_poke_state(json: &str) -> Result<State, serde_json::Error> {
    let state: FabricAuxState = serde_json::from_str(json)?;
    Ok(from_fabric_aux(&state))
}

/// Auto-detect the JSON format and parse into a `State`.
///
/// Tries fabric-aux format first (checks for `"sides"` key), then falls back
/// to the original `BattleRequest` format.
pub fn auto_detect_and_parse(json: &str) -> Result<State, String> {
    // Quick heuristic: if the JSON contains "sides", it's fabric-aux format
    // Use serde_json::Value for reliable detection
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("Invalid JSON: {}", e))?;

    if value.get("sides").is_some() {
        // Fabric-aux format
        fabric_aux_json_to_poke_state(json)
            .map_err(|e| format!("Failed to parse fabric-aux state: {}", e))
    } else if value.get("sideOne").is_some() || value.get("side_one").is_some() {
        // Original BattleRequest format
        json_to_poke_state(json)
            .map_err(|e| format!("Failed to parse BattleRequest: {}", e))
    } else {
        Err("Unrecognized battle state format: expected 'sides' (fabric-aux) or 'sideOne' (BattleRequest)".to_string())
    }
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

    // Guard empty types[] — battle-testing adapter pads teams to 6 with
    // placeholder pokemon that have species="none" and types=[].
    let type1 = if pkmn.types.is_empty() {
        "Typeless".to_string()
    } else {
        capitalize_type(&pkmn.types[0])
    };
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

    // Map Showdown status abbreviations to poke-engine status names
    let status = match pkmn.status.to_lowercase().as_str() {
        "brn" => "Burn".to_string(),
        "slp" => "Sleep".to_string(),
        "frz" => "Freeze".to_string(),
        "par" => "Paralyze".to_string(),
        "psn" => "Poison".to_string(),
        "tox" => "Toxic".to_string(),
        "fnt" | "none" | "" => "None".to_string(),
        _ => capitalize_type(&pkmn.status), // fallback
    };

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
        side.substitute_health.unwrap_or(0), // [10] substitute_health
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
        side.force_switch,          // [22] force_switch
        "NONE",                     // [23] switch_out_move
        false,                      // [24] baton_passing
        false,                      // [25] shed_tailing
        side.force_trapped,         // [26] force_trapped
        // [27] last_used_move — must be in LastUsedMove::deserialize format:
        // `move:<idx>` (0..3), `switch:<idx>`, or `move:none`. Bare `none`
        // panics the deserializer (state.rs:74).
        side.last_used_move.as_deref().unwrap_or("move:none"),
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

    let mut state = State::deserialize(&serialized);
    state.tera_banned = request.tera_banned;
    state
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

    // -- Fabric-aux format tests --

    fn fabric_aux_sample_json() -> &'static str {
        r#"{
            "turn": 1,
            "weather": null,
            "terrain": null,
            "sides": [
                {
                    "id": "p1",
                    "name": "player-uuid",
                    "pokemon": [
                        {
                            "species": "Blaziken",
                            "level": 26,
                            "hp": 81,
                            "maxhp": 85,
                            "status": null,
                            "item": null,
                            "ability": "speedboost",
                            "baseAbility": "speedboost",
                            "types": ["Fire", "Fighting"],
                            "stats": {"atk": 91, "def": 48, "spa": 63, "spd": 48, "spe": 78},
                            "ivs": {"hp": 31, "atk": 31, "def": 26, "spa": 31, "spd": 27, "spe": 31},
                            "evs": {"hp": 4, "atk": 252, "def": 2, "spa": 0, "spd": 0, "spe": 252},
                            "nature": "Jolly",
                            "gender": "M",
                            "boosts": {"atk": 0, "def": -1, "spa": 0, "spd": -1, "spe": 0, "accuracy": 0, "evasion": 0},
                            "moves": [
                                {"name": "Swords Dance", "id": "swordsdance", "pp": 20, "maxpp": 20, "basePower": 0},
                                {"name": "Close Combat", "id": "closecombat", "pp": 3, "maxpp": 5, "basePower": 0},
                                {"name": "Knock Off", "id": "knockoff", "pp": 20, "maxpp": 20, "basePower": 0},
                                {"name": "Blaze Kick", "id": "blazekick", "pp": 9, "maxpp": 10, "basePower": 0}
                            ],
                            "fainted": false,
                            "isActive": true,
                            "weight": 520
                        },
                        {
                            "species": "Pangoro",
                            "level": 25,
                            "hp": 106,
                            "maxhp": 106,
                            "status": null,
                            "item": null,
                            "ability": "ironfist",
                            "baseAbility": "ironfist",
                            "types": ["Fighting", "Dark"],
                            "stats": {"atk": 99, "def": 50, "spa": 42, "spd": 44, "spe": 41},
                            "ivs": {"hp": 31, "atk": 31, "def": 27, "spa": 31, "spd": 14, "spe": 31},
                            "evs": {"hp": 252, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                            "nature": "Adamant",
                            "gender": "F",
                            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0},
                            "moves": [
                                {"name": "Crunch", "id": "crunch", "pp": 15, "maxpp": 15, "basePower": 0},
                                {"name": "Drain Punch", "id": "drainpunch", "pp": 10, "maxpp": 10, "basePower": 0},
                                {"name": "Ice Punch", "id": "icepunch", "pp": 15, "maxpp": 15, "basePower": 0},
                                {"name": "Bullet Punch", "id": "bulletpunch", "pp": 30, "maxpp": 30, "basePower": 0}
                            ],
                            "fainted": false,
                            "isActive": false,
                            "weight": 1360
                        }
                    ]
                },
                {
                    "id": "p2",
                    "name": "opponent-uuid",
                    "pokemon": [
                        {
                            "species": "Meowth",
                            "level": 13,
                            "hp": 0,
                            "maxhp": 36,
                            "status": null,
                            "item": null,
                            "ability": "technician",
                            "baseAbility": "technician",
                            "types": ["Normal"],
                            "stats": {"atk": 18, "def": 16, "spa": 18, "spd": 18, "spe": 27},
                            "ivs": {"hp": 22, "atk": 15, "def": 20, "spa": 26, "spd": 18, "spe": 14},
                            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                            "nature": "Sassy",
                            "gender": "M",
                            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0},
                            "moves": [
                                {"name": "Pay Day", "id": "payday", "pp": 20, "maxpp": 20, "basePower": 0},
                                {"name": "Scratch", "id": "scratch", "pp": 35, "maxpp": 35, "basePower": 0},
                                {"name": "Feint", "id": "feint", "pp": 9, "maxpp": 10, "basePower": 0},
                                {"name": "Fake Out", "id": "fakeout", "pp": 10, "maxpp": 10, "basePower": 0}
                            ],
                            "fainted": true,
                            "isActive": false,
                            "weight": 42
                        }
                    ]
                }
            ]
        }"#
    }

    #[test]
    fn test_fabric_aux_basic_parse() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Side one active = Blaziken
        let active_one = state.side_one.get_active_immutable();
        assert_eq!(active_one.hp, 81);
        assert_eq!(active_one.maxhp, 85);
        assert_eq!(active_one.level, 26);
        assert_eq!(active_one.attack, 91);
        assert_eq!(active_one.defense, 48);
        assert_eq!(active_one.special_attack, 63);
        assert_eq!(active_one.special_defense, 48);
        assert_eq!(active_one.speed, 78);
    }

    #[test]
    fn test_fabric_aux_active_pokemon_detection() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Active index should be 0 (Blaziken has isActive: true)
        let active = state.side_one.get_active_immutable();
        assert_eq!(active.hp, 81); // Blaziken's HP
        assert_eq!(active.maxhp, 85);
    }

    #[test]
    fn test_fabric_aux_boosts_on_side() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Blaziken has def: -1 and spd: -1 boosts
        assert_eq!(state.side_one.defense_boost, -1);
        assert_eq!(state.side_one.special_defense_boost, -1);
        assert_eq!(state.side_one.attack_boost, 0);
        assert_eq!(state.side_one.speed_boost, 0);
    }

    #[test]
    fn test_fabric_aux_fainted_pokemon() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Meowth on side two is fainted — HP should be 0
        let meowth = &state.side_two.pokemon.pkmn[0];
        assert_eq!(meowth.hp, 0);
        assert_eq!(meowth.maxhp, 36);
    }

    #[test]
    fn test_fabric_aux_weight_conversion() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Blaziken weight: 520 hectograms = 52.0 kg
        let active = state.side_one.get_active_immutable();
        assert!((active.weight_kg - 52.0).abs() < 0.1);
    }

    #[test]
    fn test_fabric_aux_team_padding() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Side one has 2 pokemon, slots 2-5 should be blank (0 HP)
        assert_eq!(state.side_one.pokemon.pkmn[2].hp, 0);
        assert_eq!(state.side_one.pokemon.pkmn[3].hp, 0);

        // Side two has 1 pokemon, slots 1-5 should be blank
        assert_eq!(state.side_two.pokemon.pkmn[1].hp, 0);
    }

    #[test]
    fn test_fabric_aux_moves_parsed() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");

        // Blaziken should have swordsdance, closecombat, knockoff, blazekick
        let active = state.side_one.get_active_immutable();
        // Verify that moves were parsed (check that pp values are preserved)
        // Close Combat is move index 1 with pp: 3
        use crate::state::PokemonMoveIndex;
        assert_eq!(active.moves[&PokemonMoveIndex::M1].pp, 3);
    }

    #[test]
    fn test_fabric_aux_roundtrip() {
        let state =
            fabric_aux_json_to_poke_state(fabric_aux_sample_json()).expect("Failed to parse");
        let serialized = state.serialize();
        let state2 = State::deserialize(&serialized);
        assert_eq!(state.serialize(), state2.serialize());
    }

    #[test]
    fn test_auto_detect_fabric_aux() {
        let state =
            auto_detect_and_parse(fabric_aux_sample_json()).expect("Failed to auto-detect");
        let active = state.side_one.get_active_immutable();
        assert_eq!(active.hp, 81);
    }

    #[test]
    fn test_auto_detect_battle_request() {
        let state = auto_detect_and_parse(sample_json()).expect("Failed to auto-detect");
        let active = state.side_one.get_active_immutable();
        assert_eq!(active.hp, 302);
    }

    #[test]
    fn test_auto_detect_invalid_json() {
        let result = auto_detect_and_parse("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_detect_unrecognized_format() {
        let result = auto_detect_and_parse(r#"{"foo": "bar"}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unrecognized"));
    }

    #[test]
    fn test_fabric_aux_real_state_file() {
        // Test with the actual /tmp/real_state.json content if available
        let json = r#"{"turn":1,"weather":null,"terrain":null,"sides":[{"id":"p1","name":"adfb38d8-117b-4683-94b0-60d3cf7a31a8","pokemon":[{"species":"Blaziken","level":26,"hp":81,"maxhp":85,"status":null,"item":null,"ability":"speedboost","baseAbility":"speedboost","types":["Fire","Fighting"],"stats":{"atk":91,"def":48,"spa":63,"spd":48,"spe":78},"ivs":{"hp":31,"atk":31,"def":26,"spa":31,"spd":27,"spe":31},"evs":{"hp":4,"atk":252,"def":2,"spa":0,"spd":0,"spe":252},"nature":"Jolly","gender":"M","boosts":{"atk":0,"def":-1,"spa":0,"spd":-1,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Swords Dance","id":"swordsdance","pp":20,"maxpp":20,"basePower":0},{"name":"Close Combat","id":"closecombat","pp":3,"maxpp":5,"basePower":0},{"name":"Knock Off","id":"knockoff","pp":20,"maxpp":20,"basePower":0},{"name":"Blaze Kick","id":"blazekick","pp":9,"maxpp":10,"basePower":0}],"fainted":false,"isActive":true,"weight":520},{"species":"Pangoro","level":25,"hp":106,"maxhp":106,"status":null,"item":null,"ability":"ironfist","baseAbility":"ironfist","types":["Fighting","Dark"],"stats":{"atk":99,"def":50,"spa":42,"spd":44,"spe":41},"ivs":{"hp":31,"atk":31,"def":27,"spa":31,"spd":14,"spe":31},"evs":{"hp":252,"atk":252,"def":0,"spa":0,"spd":0,"spe":0},"nature":"Adamant","gender":"F","boosts":{"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Crunch","id":"crunch","pp":15,"maxpp":15,"basePower":0},{"name":"Drain Punch","id":"drainpunch","pp":10,"maxpp":10,"basePower":0},{"name":"Ice Punch","id":"icepunch","pp":15,"maxpp":15,"basePower":0},{"name":"Bullet Punch","id":"bulletpunch","pp":30,"maxpp":30,"basePower":0}],"fainted":false,"isActive":false,"weight":1360},{"species":"Lucario","level":25,"hp":78,"maxhp":78,"status":null,"item":null,"ability":"innerfocus","baseAbility":"innerfocus","types":["Fighting","Steel"],"stats":{"atk":60,"def":40,"spa":86,"spd":42,"spe":80},"ivs":{"hp":31,"atk":31,"def":0,"spa":31,"spd":9,"spe":31},"evs":{"hp":4,"atk":0,"def":0,"spa":252,"spd":0,"spe":252},"nature":"Timid","gender":"M","boosts":{"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Calm Mind","id":"calmmind","pp":20,"maxpp":20,"basePower":0},{"name":"Aura Sphere","id":"aurasphere","pp":20,"maxpp":20,"basePower":0},{"name":"Flash Cannon","id":"flashcannon","pp":10,"maxpp":10,"basePower":0},{"name":"Dark Pulse","id":"darkpulse","pp":15,"maxpp":15,"basePower":0}],"fainted":false,"isActive":false,"weight":540},{"species":"Hawlucha","level":25,"hp":82,"maxhp":82,"status":null,"item":null,"ability":"unburden","baseAbility":"unburden","types":["Fighting","Flying"],"stats":{"atk":74,"def":48,"spa":44,"spd":44,"spe":95},"ivs":{"hp":31,"atk":31,"def":24,"spa":31,"spd":31,"spe":31},"evs":{"hp":4,"atk":252,"def":0,"spa":0,"spd":0,"spe":252},"nature":"Jolly","gender":"M","boosts":{"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Swords Dance","id":"swordsdance","pp":20,"maxpp":20,"basePower":0},{"name":"Acrobatics","id":"acrobatics","pp":15,"maxpp":15,"basePower":0},{"name":"Close Combat","id":"closecombat","pp":5,"maxpp":5,"basePower":0},{"name":"Roost","id":"roost","pp":5,"maxpp":5,"basePower":0}],"fainted":false,"isActive":false,"weight":215},{"species":"Breloom","level":25,"hp":73,"maxhp":73,"status":null,"item":null,"ability":"poisonheal","baseAbility":"poisonheal","types":["Grass","Fighting"],"stats":{"atk":102,"def":48,"spa":37,"spd":41,"spe":63},"ivs":{"hp":31,"atk":31,"def":14,"spa":31,"spd":26,"spe":31},"evs":{"hp":4,"atk":252,"def":2,"spa":0,"spd":0,"spe":252},"nature":"Adamant","gender":"F","boosts":{"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Spore","id":"spore","pp":15,"maxpp":15,"basePower":0},{"name":"Mach Punch","id":"machpunch","pp":30,"maxpp":30,"basePower":0},{"name":"Seed Bomb","id":"seedbomb","pp":15,"maxpp":15,"basePower":0},{"name":"Leech Seed","id":"leechseed","pp":10,"maxpp":10,"basePower":0}],"fainted":false,"isActive":false,"weight":392},{"species":"Great Tusk","level":25,"hp":100,"maxhp":100,"status":null,"item":null,"ability":"protosynthesis","baseAbility":"protosynthesis","types":["Ground","Fighting"],"stats":{"atk":94,"def":77,"spa":35,"spd":39,"spe":79},"ivs":{"hp":31,"atk":31,"def":27,"spa":31,"spd":31,"spe":31},"evs":{"hp":4,"atk":252,"def":0,"spa":0,"spd":0,"spe":252},"nature":"Jolly","gender":null,"boosts":{"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Earthquake","id":"earthquake","pp":10,"maxpp":10,"basePower":0},{"name":"Close Combat","id":"closecombat","pp":5,"maxpp":5,"basePower":0},{"name":"Knock Off","id":"knockoff","pp":20,"maxpp":20,"basePower":0},{"name":"Rapid Spin","id":"rapidspin","pp":40,"maxpp":40,"basePower":0}],"fainted":false,"isActive":false,"weight":3200}]},{"id":"p2","name":"5723adc0-fdda-4730-9634-5bad49796435","pokemon":[{"species":"Meowth","level":13,"hp":0,"maxhp":36,"status":null,"item":null,"ability":"technician","baseAbility":"technician","types":["Normal"],"stats":{"atk":18,"def":16,"spa":18,"spd":18,"spe":27},"ivs":{"hp":22,"atk":15,"def":20,"spa":26,"spd":18,"spe":14},"evs":{"hp":0,"atk":0,"def":0,"spa":0,"spd":0,"spe":0},"nature":"Sassy","gender":"M","boosts":{"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},"moves":[{"name":"Pay Day","id":"payday","pp":20,"maxpp":20,"basePower":0},{"name":"Scratch","id":"scratch","pp":35,"maxpp":35,"basePower":0},{"name":"Feint","id":"feint","pp":9,"maxpp":10,"basePower":0},{"name":"Fake Out","id":"fakeout","pp":10,"maxpp":10,"basePower":0}],"fainted":true,"isActive":false,"weight":42}]}]}"#;

        let state = auto_detect_and_parse(json).expect("Failed to parse real state");

        // Verify p1 side (6 pokemon)
        let active = state.side_one.get_active_immutable();
        assert_eq!(active.hp, 81);
        assert_eq!(active.level, 26);

        // Verify Pangoro is at index 1
        assert_eq!(state.side_one.pokemon.pkmn[1].hp, 106);
        assert_eq!(state.side_one.pokemon.pkmn[1].maxhp, 106);

        // Verify all 6 pokemon on side one are populated
        assert_eq!(state.side_one.pokemon.pkmn[0].hp, 81);  // Blaziken
        assert_eq!(state.side_one.pokemon.pkmn[1].hp, 106); // Pangoro
        assert_eq!(state.side_one.pokemon.pkmn[2].hp, 78);  // Lucario
        assert_eq!(state.side_one.pokemon.pkmn[3].hp, 82);  // Hawlucha
        assert_eq!(state.side_one.pokemon.pkmn[4].hp, 73);  // Breloom
        assert_eq!(state.side_one.pokemon.pkmn[5].hp, 100); // Great Tusk

        // Verify p2 side (fainted Meowth)
        assert_eq!(state.side_two.pokemon.pkmn[0].hp, 0);
        assert_eq!(state.side_two.pokemon.pkmn[0].maxhp, 36);

        // Verify boosts from active Blaziken are on side_one
        assert_eq!(state.side_one.defense_boost, -1);
        assert_eq!(state.side_one.special_defense_boost, -1);

        // Roundtrip test
        let serialized = state.serialize();
        let state2 = State::deserialize(&serialized);
        assert_eq!(state.serialize(), state2.serialize());
    }
}
