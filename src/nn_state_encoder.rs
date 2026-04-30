//! Plan E state encoder: rust `State` -> `BattleRequest` JSON for the metamon
//! sidecar.
//!
//! The Phase 2 sidecar consumes the same `BattleRequest` shape that
//! `translate.rs` already parses inward. This module is the reverse direction.
//!
//! ## Output shape (matches `~/Projects/metamon-spike/sidecar/fixtures.py`):
//!
//! ```json
//! {
//!   "sideOne": { "pokemon": [...], "activeIndex": 0, "boosts": {...},
//!                "sideConditions": {...} },
//!   "sideTwo": { ... },
//!   "weather": {"weatherType": "none"},
//!   "terrain": {"terrainType": "none"},
//!   "trickRoom": false
//! }
//! ```
//!
//! ## `map_policy_to_options` (THE LOAD-BEARING FUNCTION)
//!
//! The sidecar returns a 13-element policy distribution indexed:
//!   - 0..3   moves alphabetical by `clean_no_numbers(name)` (lower, alpha-only)
//!   - 4..8   switches alphabetical by `clean_name(species)` (lower, alphanumeric)
//!   - 9..12  tera-variants of the same moves as 0..3
//!
//! The engine's `MoveChoice` list (`s1_options` / `s2_options`) is in slot
//! order — M0..M3, then switches by `PokemonIndex` order, then tera variants
//! also in slot order. The mapping from `s1_options[i]` to its sidecar
//! `probs[k]` index is NOT the identity; we must permute via the sort.
//!
//! Failures here silently corrupt every NN-driven recommendation, so the
//! tests in `tests/test_state_encoder.rs` lock down the alphabetical-sort
//! semantics with explicit fixtures.

use serde_json::{json, Value};

use crate::engine::state::MoveChoice;
use crate::nn_client::{Perspective, ACTION_DIM};
use crate::state::{
    Move, Pokemon, PokemonIndex, PokemonMoveIndex, Side, SideConditions, State, StateTerrain,
    StateTrickRoom, StateWeather,
};

/// Engine-internal side selector. Mirrors the search-loop's side enumeration
/// for the encoder's "whose perspective is this from?" arg.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SidePerspective {
    Side1,
    Side2,
}

impl SidePerspective {
    pub fn to_p1_p2(self) -> Perspective {
        // The Showdown `p1`/`p2` distinction maps directly to engine
        // `side_one`/`side_two`. The engine always optimizes for "side_one"
        // — when serving Plan H battles where the player is on p2, the
        // proxy already swaps the sides at the JSON layer (see
        // `project_pvp_perspective_fix`). So whatever side_one is, that's
        // p1 to the sidecar.
        match self {
            SidePerspective::Side1 => Perspective::P1,
            SidePerspective::Side2 => Perspective::P2,
        }
    }
}

// ---------------------------------------------------------------------------
// Normalization helpers — must match metamon's `clean_no_numbers` /
// `clean_name` exactly. Used by both the action-index mapping and (for
// sanity) any encoder-side string canonicalization.
// ---------------------------------------------------------------------------

/// metamon `move_name`: lowercase + filter to alphabetic only (NO digits).
/// Drives the alphabetical sort over move IDs in the policy index.
pub fn move_name_norm(s: &str) -> String {
    s.chars().filter(|c| c.is_alphabetic()).collect::<String>().to_lowercase()
}

/// metamon `pokemon_name`: lowercase + filter to alphanumeric (digits OK).
/// Drives the alphabetical sort over species in the policy index.
pub fn pokemon_name_norm(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Top-level `State` -> `BattleRequest` JSON.
///
/// `perspective` only affects logging today — the sidecar is told the
/// perspective via the request body, not the JSON state itself.
pub fn encode(state: &State, _perspective: SidePerspective) -> Value {
    json!({
        "sideOne": encode_side(&state.side_one),
        "sideTwo": encode_side(&state.side_two),
        "weather": encode_weather(&state.weather),
        "terrain": encode_terrain(&state.terrain),
        "trickRoom": encode_trickroom(&state.trick_room),
    })
}

fn encode_side(side: &Side) -> Value {
    let active_idx = pokemon_index_to_usize(side.active_index);
    let mut pkmn_arr: Vec<Value> = Vec::with_capacity(6);
    for (slot, p) in side.pokemon.into_iter().enumerate() {
        // Per-mon boosts: only the active mon has them (side-level boosts in
        // the engine; sidecar expects per-active-mon hash like
        // {"special-attack": 2}).
        let boosts = if slot == active_idx {
            encode_active_pokemon_boosts(side)
        } else {
            json!({})
        };
        pkmn_arr.push(encode_pokemon(p, boosts));
    }
    json!({
        "pokemon": pkmn_arr,
        "activeIndex": active_idx,
        "sideConditions": encode_side_conditions(&side.side_conditions),
        "boosts": encode_side_level_boosts(side),
        "forceTrapped": side.force_trapped,
        "forceSwitch": side.force_switch,
    })
}

fn encode_pokemon(p: &Pokemon, boosts: Value) -> Value {
    let species_lc = format!("{:?}", p.id).to_lowercase();
    let type1 = format!("{:?}", p.types.0).to_lowercase();
    let type2 = format!("{:?}", p.types.1).to_lowercase();
    let types = if type2 == "typeless" {
        vec![type1]
    } else {
        vec![type1, type2]
    };
    let ability = format!("{:?}", p.ability).to_lowercase();
    let item = format!("{:?}", p.item).to_lowercase();
    let nature = format!("{:?}", p.nature).to_lowercase();
    let status = format!("{:?}", p.status);
    let tera_type = format!("{:?}", p.tera_type).to_lowercase();

    json!({
        "species": species_lc,
        "level": p.level,
        "types": types,
        "hp": p.hp,
        "maxhp": p.maxhp,
        "ability": ability,
        "item": item,
        "nature": nature,
        "evs": {
            "hp": p.evs.0,
            "atk": p.evs.1,
            "def": p.evs.2,
            "spa": p.evs.3,
            "spd": p.evs.4,
            "spe": p.evs.5,
        },
        "attack": p.attack,
        "defense": p.defense,
        "specialAttack": p.special_attack,
        "specialDefense": p.special_defense,
        "speed": p.speed,
        "status": status,
        "restTurns": p.rest_turns,
        "sleepTurns": p.sleep_turns,
        "weightKg": p.weight_kg,
        "moves": [
            encode_move(&p.moves.m0),
            encode_move(&p.moves.m1),
            encode_move(&p.moves.m2),
            encode_move(&p.moves.m3),
        ],
        "terastallized": p.terastallized,
        "teraType": tera_type,
        "boosts": boosts,
    })
}

/// Sidecar-compatible per-active-mon boosts (the keys it expects are
/// `attack`/`defense`/`special-attack`/`special-defense`/`speed`/`accuracy`/`evasion`).
/// Zero entries are elided to match how the Python fixture is shaped.
fn encode_active_pokemon_boosts(side: &Side) -> Value {
    let mut m = serde_json::Map::new();
    if side.attack_boost != 0 {
        m.insert("attack".into(), json!(side.attack_boost));
    }
    if side.defense_boost != 0 {
        m.insert("defense".into(), json!(side.defense_boost));
    }
    if side.special_attack_boost != 0 {
        m.insert("special-attack".into(), json!(side.special_attack_boost));
    }
    if side.special_defense_boost != 0 {
        m.insert("special-defense".into(), json!(side.special_defense_boost));
    }
    if side.speed_boost != 0 {
        m.insert("speed".into(), json!(side.speed_boost));
    }
    if side.accuracy_boost != 0 {
        m.insert("accuracy".into(), json!(side.accuracy_boost));
    }
    if side.evasion_boost != 0 {
        m.insert("evasion".into(), json!(side.evasion_boost));
    }
    Value::Object(m)
}

fn encode_move(m: &Move) -> Value {
    // `Choices::NONE` is the empty-slot sentinel. The sidecar's translator
    // already understands "none"/"nomove"; either spelling works.
    let id = format!("{:?}", m.id).to_lowercase();
    let id = if id == "none" { "none".to_string() } else { id };
    json!({ "id": id, "pp": m.pp, "disabled": m.disabled })
}

fn encode_side_conditions(sc: &SideConditions) -> Value {
    json!({
        "auroraVeil": sc.aurora_veil,
        "craftyShield": sc.crafty_shield,
        "healingWish": sc.healing_wish,
        "lightScreen": sc.light_screen,
        "luckyChant": sc.lucky_chant,
        "lunarDance": sc.lunar_dance,
        "matBlock": sc.mat_block,
        "mist": sc.mist,
        "protect": sc.protect,
        "quickGuard": sc.quick_guard,
        "reflect": sc.reflect,
        "safeguard": sc.safeguard,
        "spikes": sc.spikes,
        "stealthRock": sc.stealth_rock,
        "stickyWeb": sc.sticky_web,
        "tailwind": sc.tailwind,
        "toxicCount": sc.toxic_count,
        "toxicSpikes": sc.toxic_spikes,
        "wideGuard": sc.wide_guard,
    })
}

/// Side-level boost block, matching `BoostsInput` in `translate.rs`. The
/// engine's translate path can read per-side boosts here and apply them to
/// the active mon — kept for round-trip symmetry.
fn encode_side_level_boosts(side: &Side) -> Value {
    json!({
        "attack": side.attack_boost,
        "defense": side.defense_boost,
        "specialAttack": side.special_attack_boost,
        "specialDefense": side.special_defense_boost,
        "speed": side.speed_boost,
        "accuracy": side.accuracy_boost,
        "evasion": side.evasion_boost,
    })
}

fn encode_weather(w: &StateWeather) -> Value {
    json!({
        "weatherType": format!("{:?}", w.weather_type).to_lowercase(),
        "turnsRemaining": w.turns_remaining,
    })
}

fn encode_terrain(t: &StateTerrain) -> Value {
    json!({
        "terrainType": format!("{:?}", t.terrain_type).to_lowercase(),
        "turnsRemaining": t.turns_remaining,
    })
}

fn encode_trickroom(tr: &StateTrickRoom) -> Value {
    // BattleRequest expects a bool. (translate.rs:24 takes default false.)
    json!(tr.active)
}

// ---------------------------------------------------------------------------
// Action-index mapping
// ---------------------------------------------------------------------------

/// Map the sidecar's 13-element `probs` vector to a per-`s1_options[i]` prior.
///
/// Steps:
/// 1. Get the active mon's 4 move slots and the side's 5 reserve switches.
/// 2. Compute alphabetical permutation for moves (alphabetic-only, lowercase)
///    and for switches (alphanumeric, lowercase).
/// 3. For each entry in `options`, find its index in the policy and copy the
///    corresponding probability. Unknown / out-of-range options get 0.
/// 4. Renormalize the result to sum to 1.0; if every option had 0 mass
///    (degenerate), fall back to uniform.
///
/// Returns a `Vec<f32>` of length `options.len()`.
pub fn map_policy_to_options(
    probs: &[f32],
    state: &State,
    perspective: SidePerspective,
    options: &[MoveChoice],
) -> Vec<f32> {
    let n = options.len();
    if probs.len() != ACTION_DIM || n == 0 {
        return vec![1.0 / n.max(1) as f32; n];
    }
    let side = match perspective {
        SidePerspective::Side1 => &state.side_one,
        SidePerspective::Side2 => &state.side_two,
    };
    let active = side.get_active_immutable();

    // 1. Build move-id -> alphabetical-policy-index map.
    let move_ids = active_move_ids(active);
    let move_alpha_perm = alpha_perm_with_norm(&move_ids, move_name_norm);

    // 2. Build switch-species -> alphabetical-policy-index map (over reserve
    //    slots; active is excluded, fainted included as a slot but with 0 mass).
    let switch_species = reserve_species(side);
    let switch_alpha_perm = alpha_perm_with_norm(&switch_species, pokemon_name_norm);

    // 3. Walk `options` and look up each one's probability.
    let mut priors = Vec::with_capacity(n);
    for opt in options {
        let p = match opt {
            MoveChoice::Move(idx) => {
                let slot = move_index_to_slot(*idx);
                match move_alpha_perm.get(slot).copied() {
                    Some(policy_idx) if policy_idx < 4 => probs[policy_idx],
                    _ => 0.0,
                }
            }
            MoveChoice::MoveTera(idx) => {
                let slot = move_index_to_slot(*idx);
                match move_alpha_perm.get(slot).copied() {
                    Some(policy_idx) if policy_idx < 4 => probs[9 + policy_idx],
                    _ => 0.0,
                }
            }
            MoveChoice::MoveMega(idx) => {
                // Kakuna's policy doesn't have a Mega slot; treat as the
                // base move slot's mass (best approximation).
                let slot = move_index_to_slot(*idx);
                match move_alpha_perm.get(slot).copied() {
                    Some(policy_idx) if policy_idx < 4 => probs[policy_idx],
                    _ => 0.0,
                }
            }
            MoveChoice::Switch(pidx) => {
                // Reserves only: the active mon doesn't appear in the
                // switch list. `reserve_species_slot_for(side, pidx)` -> Option<reserve_slot>.
                match reserve_slot_for(side, *pidx) {
                    Some(reserve_slot) => match switch_alpha_perm.get(reserve_slot).copied() {
                        Some(policy_idx) if policy_idx < 5 => probs[4 + policy_idx],
                        _ => 0.0,
                    },
                    None => 0.0,
                }
            }
            MoveChoice::None => 0.0,
        };
        priors.push(p);
    }

    // 4. Renormalize (or fall back to uniform).
    let sum: f32 = priors.iter().sum();
    if sum > 1e-6 && sum.is_finite() {
        for p in priors.iter_mut() {
            *p /= sum;
        }
    } else {
        let u = 1.0 / n as f32;
        for p in priors.iter_mut() {
            *p = u;
        }
    }
    priors
}

/// Variant of `map_policy_to_options` that optionally blends an external
/// heuristic prior with the NN policy before mapping to options.
///
/// Final per-slot prior is `(1-λ)·P_NN + λ·P_heuristic`, then mapped to
/// the options list using the same alphabetical-slot logic.
///
/// When `heuristic` is `None` or `lambda_mix <= 0.0`, returns the same
/// result as `map_policy_to_options(probs, state, perspective, options)`
/// bit-identically (regression guard for the default-off CLI flag).
///
/// `lambda_mix` is clamped to `[0.0, 1.0]`.
pub fn map_policy_to_options_blended(
    probs: &[f32],
    state: &State,
    perspective: SidePerspective,
    options: &[MoveChoice],
    heuristic: Option<&crate::heuristic_prior::HeuristicPrior>,
    lambda_mix: f32,
) -> Vec<f32> {
    if heuristic.is_none() || lambda_mix <= 0.0 {
        return map_policy_to_options(probs, state, perspective, options);
    }
    let h = heuristic.unwrap();
    let lam = lambda_mix.clamp(0.0, 1.0);

    let mut blended = [0.0_f32; ACTION_DIM];
    for i in 0..ACTION_DIM {
        let nn = probs.get(i).copied().unwrap_or(0.0);
        let hp = h.probs[i];
        blended[i] = (1.0 - lam) * nn + lam * hp;
    }

    map_policy_to_options(&blended, state, perspective, options)
}

/// Return the 4 move IDs (raw `Choices` debug strings, lowercase) for the
/// active Pokemon — including blank slots so the indexing matches `M0..M3`.
pub(crate) fn active_move_ids(p: &Pokemon) -> Vec<String> {
    vec![
        format!("{:?}", p.moves.m0.id).to_lowercase(),
        format!("{:?}", p.moves.m1.id).to_lowercase(),
        format!("{:?}", p.moves.m2.id).to_lowercase(),
        format!("{:?}", p.moves.m3.id).to_lowercase(),
    ]
}

/// Return reserve species names in `PokemonIndex` order, EXCLUDING the active
/// slot. Length is 5 (six total minus active).
pub(crate) fn reserve_species(side: &Side) -> Vec<String> {
    let mut out = Vec::with_capacity(5);
    for idx in 0..6 {
        let pidx = match idx {
            0 => PokemonIndex::P0,
            1 => PokemonIndex::P1,
            2 => PokemonIndex::P2,
            3 => PokemonIndex::P3,
            4 => PokemonIndex::P4,
            _ => PokemonIndex::P5,
        };
        if pidx == side.active_index {
            continue;
        }
        out.push(format!("{:?}", side.pokemon[pidx].id).to_lowercase());
    }
    out
}

/// Reverse of `reserve_species`: given a `PokemonIndex` for a (non-active)
/// reserve slot, find its 0..4 position in the reserves list. Returns
/// `None` if `pidx` is the active slot.
pub(crate) fn reserve_slot_for(side: &Side, pidx: PokemonIndex) -> Option<usize> {
    if pidx == side.active_index {
        return None;
    }
    let mut count = 0usize;
    for idx in 0..6 {
        let candidate = match idx {
            0 => PokemonIndex::P0,
            1 => PokemonIndex::P1,
            2 => PokemonIndex::P2,
            3 => PokemonIndex::P3,
            4 => PokemonIndex::P4,
            _ => PokemonIndex::P5,
        };
        if candidate == side.active_index {
            continue;
        }
        if candidate == pidx {
            return Some(count);
        }
        count += 1;
    }
    None
}

pub(crate) fn move_index_to_slot(idx: PokemonMoveIndex) -> usize {
    match idx {
        PokemonMoveIndex::M0 => 0,
        PokemonMoveIndex::M1 => 1,
        PokemonMoveIndex::M2 => 2,
        PokemonMoveIndex::M3 => 3,
    }
}

fn pokemon_index_to_usize(idx: PokemonIndex) -> usize {
    match idx {
        PokemonIndex::P0 => 0,
        PokemonIndex::P1 => 1,
        PokemonIndex::P2 => 2,
        PokemonIndex::P3 => 3,
        PokemonIndex::P4 => 4,
        PokemonIndex::P5 => 5,
    }
}

/// Given `items` and a normalization function, return for each input slot its
/// alphabetical rank under that normalization.
///
/// e.g. items = ["earthquake", "scaleshot", "stealthrock", "spikes"] →
///      ranks = [0, 2, 3, 1]   (alphabetical: earthquake < scaleshot < spikes
///                              < stealthrock; but wait we need "scaleshot"
///                              after "spikes" — let me redo.)
/// Actually the alphabetical sort: earthquake(0) < scaleshot(2) — no:
///   earthquake(e), scaleshot(s), spikes(s), stealthrock(s)
///   so e < s; among the s's: scaleshot, spikes, stealthrock alphabetical
///   ⇒ ranks for input order [earthquake, scaleshot, stealthrock, spikes]:
///     earthquake → 0 (smallest), scaleshot → 1 (next), stealthrock → 3,
///     spikes → 2.
pub(crate) fn alpha_perm_with_norm<F>(items: &[String], norm: F) -> Vec<usize>
where
    F: Fn(&str) -> String,
{
    let mut indexed: Vec<(usize, String)> = items
        .iter()
        .enumerate()
        .map(|(i, s)| (i, norm(s)))
        .collect();
    // Stable sort so equal keys preserve input order — matches Python's sorted().
    indexed.sort_by(|a, b| a.1.cmp(&b.1));
    let mut rank = vec![0usize; items.len()];
    for (rank_i, (orig_i, _)) in indexed.into_iter().enumerate() {
        rank[orig_i] = rank_i;
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_norm_drops_digits_and_dashes() {
        assert_eq!(move_name_norm("Stealth Rock"), "stealthrock");
        assert_eq!(move_name_norm("U-turn"), "uturn");
        // moves with digits in their canonical id strip them too
        assert_eq!(move_name_norm("hiddenpower70"), "hiddenpower");
    }

    #[test]
    fn pokemon_norm_keeps_digits() {
        assert_eq!(pokemon_name_norm("Iron Crown"), "ironcrown");
        // some species have digits in their canonical id (e.g., porygon2)
        assert_eq!(pokemon_name_norm("Porygon2"), "porygon2");
    }

    #[test]
    fn alpha_perm_iron_crown_t5_moves() {
        // Garchomp's slot order: [earthquake, scaleshot, stealthrock, spikes]
        let moves = vec![
            "earthquake".to_string(),
            "scaleshot".to_string(),
            "stealthrock".to_string(),
            "spikes".to_string(),
        ];
        let perm = alpha_perm_with_norm(&moves, move_name_norm);
        // alphabetical: earthquake(0), scaleshot(1), spikes(2), stealthrock(3)
        // input order:  earthquake → 0 (alpha rank 0)
        //               scaleshot  → 1 (alpha rank 1)
        //               stealthrock→ 3 (alpha rank 3)
        //               spikes     → 2 (alpha rank 2)
        assert_eq!(perm, vec![0, 1, 3, 2]);
    }
}
