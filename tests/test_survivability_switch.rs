//! Plan I, Fix #1: survivability-aware matchup_switch_pick.
//!
//! Red-phase failing tests. The current `matchup_switch_pick` only sums
//! type-effectiveness multipliers — it ignores HP, hazards, status, and
//! predicted incoming damage. These tests pin the desired survivability
//! behavior.

use poke_engine::choices::Choices;
use poke_engine::engine::items::Items;
use poke_engine::heuristic_prior::matchup_switch_pick;
use poke_engine::nn_state_encoder::SidePerspective;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, PokemonType, State};

/// Two bench mons with the SAME type matchup score vs opp, but the
/// "naturally first" candidate (P1) is at low HP — would die to opp's
/// predicted move — while P2 is at full HP. Survivability-aware
/// heuristic must reject P1 and pick P2.
///
/// The current `matchup_switch_pick` ties by iteration order (first
/// score wins because the comparison uses strict `<`), so it returns
/// HEATRAN (P1). The new behavior must return TOXAPEX (P2).
///
/// Both candidates share Bug-resistance (Heatran 0.25× Bug, Toxapex 0.5×
/// Bug — NOT identical, but Toxapex's 0.5× still resists). To force a
/// HP-based tiebreak we use two FIRE/STEEL clones renamed via
/// PokemonName so the type-eff sums match exactly. We use HEATRAN (P1,
/// low HP) and DIALGA (P2, full HP) — both Steel-type with secondary
/// types that give the same type-eff against Volcarona's Bug Buzz. To
/// guarantee a tie we hand-set both as FIRE/STEEL.
#[test]
fn switch_picks_higher_hp_among_equal_matchups() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::GARCHOMP;
    state.side_one.active_index = PokemonIndex::P0;

    // P1 = Heatran clone at 5% HP, OHKO'd by opp's Surging Strikes.
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_one.pokemon[PokemonIndex::P1].hp = 19;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P1].defense = 248;
    state.side_one.pokemon[PokemonIndex::P1].special_defense = 384;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::STONEEDGE);

    // P2 = Dialga (also FIRE/STEEL forced) at full HP, same matchup
    // score, survives.
    state.side_one.pokemon[PokemonIndex::P2].id = PokemonName::DIALGA;
    state.side_one.pokemon[PokemonIndex::P2].types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_one.pokemon[PokemonIndex::P2].hp = 385;
    state.side_one.pokemon[PokemonIndex::P2].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P2].defense = 248;
    state.side_one.pokemon[PokemonIndex::P2].special_defense = 384;
    state.side_one.pokemon[PokemonIndex::P2]
        .replace_move(PokemonMoveIndex::M0, Choices::STONEEDGE);

    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;
    state.side_one.pokemon[PokemonIndex::P5].hp = 0;

    // Volcarona running Bug Buzz — 0.25× vs FIRE/STEEL (Bug ½× × Steel
    // ½×, Fire neutral). With ~150 BP equivalent and 100 SpA, max-roll
    // damage vs 19-HP Heatran will OHKO; vs 385-HP Dialga will not.
    state.side_two.get_active().id = PokemonName::VOLCARONA;
    state.side_two.get_active().types = (PokemonType::BUG, PokemonType::FIRE);
    state.side_two.get_active().level = 100;
    state.side_two.get_active().special_attack = 369;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BUGBUZZ);

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(PokemonName::DIALGA),
        "tie on type matchup → must pick the higher-HP candidate (Dialga survives, Heatran dies)"
    );
}

/// Bench[1] = low-HP Toxapex (would die to spikes on entry).
/// Bench[2] = Dragonite at full HP, slightly worse type matchup.
/// Side has 3 spikes layers. Toxapex is grounded → eats 25% of maxhp on
/// entry, which kills it. The survivability-aware heuristic must avoid
/// the dying switch and pick Dragonite.
#[test]
fn switch_avoids_low_hp_into_spikes() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::GARCHOMP;
    state.side_one.active_index = PokemonIndex::P0;

    // 3 layers of spikes on side_one — switch-ins eat 25% maxhp.
    state.side_one.side_conditions.spikes = 3;

    // P1 = Toxapex. Poison/Water (grounded). HP at 20% (below the 25%
    // spikes threshold) → dies on entry. Best type matchup vs opp.
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::TOXAPEX;
    state.side_one.pokemon[PokemonIndex::P1].types = (PokemonType::POISON, PokemonType::WATER);
    state.side_one.pokemon[PokemonIndex::P1].hp = 50;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 300;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::SCALD);

    // P2 = Dragonite. Dragon/Flying (NOT grounded → spikes immune).
    // Worse type matchup vs Volcarona's Bug Buzz, but survives entry.
    state.side_one.pokemon[PokemonIndex::P2].id = PokemonName::DRAGONITE;
    state.side_one.pokemon[PokemonIndex::P2].types = (PokemonType::DRAGON, PokemonType::FLYING);
    state.side_one.pokemon[PokemonIndex::P2].hp = 350;
    state.side_one.pokemon[PokemonIndex::P2].maxhp = 350;
    state.side_one.pokemon[PokemonIndex::P2]
        .replace_move(PokemonMoveIndex::M0, Choices::DRAGONCLAW);

    // Faint others so only P1 and P2 are candidates.
    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;
    state.side_one.pokemon[PokemonIndex::P5].hp = 0;

    state.side_two.get_active().id = PokemonName::VOLCARONA;
    state.side_two.get_active().types = (PokemonType::BUG, PokemonType::FIRE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BUGBUZZ);

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(PokemonName::DRAGONITE),
        "should avoid Toxapex (dies to 3-spikes on entry) even though its type matchup is better"
    );
}

/// Bench[1] = Pawmot at 25% HP (Electric/Fighting — has type advantage:
/// Fighting hits opp Tyranitar 4× via Close Combat, also resists
/// Tyranitar's Crunch via Fighting). Bench[2] = Slowbro at full HP
/// (Water/Psychic — neutral matchup at best). Opp Tyranitar runs Stone
/// Edge which is 4× SE vs Pawmot's Fighting/Electric typing (Rock vs
/// Fighting 2× × Rock vs Electric 1× hmm wait — actually Rock vs
/// Fighting is 0.5×). Re-pick: Use opp = Garchomp running Stone Edge
/// (Rock vs Pawmot Fighting/Electric: 0.5× × 0.5× = 0.25× — bad). Try
/// opp = Hydreigon running Draco Meteor: vs Pawmot E/F = 1× × 1× = 1×.
///
/// Simpler fixture: opp = Galarian Zapdos running Close Combat. Pawmot
/// (Electric/Fighting) takes 0.5× from CC (Fighting × Fighting = 0.5).
/// Wait — Fighting vs Electric = 1×, Fighting vs Fighting = 0.5×, so
/// Pawmot takes 0.5× CC.
///
/// Cleanest version: bench candidates with reversed survivability vs
/// type-matchup ranking. Use:
///   - P1 = LANDORUSTHERIAN (Ground/Flying) at 30 HP / 320 maxhp.
///     Best type matchup vs opp: Fighting moves immune (Ghost? no).
///     Actually: vs opp Heatran — Ground hits Heatran 4×. Fire vs
///     Ground/Flying neutral. Score wins.
///   - P2 = TOXAPEX (Poison/Water) at full HP. Water vs Heatran 2×.
///     Worse matchup score than Lando-T.
/// Opp Heatran running Magma Storm — neutral vs Lando-T (Fire vs
/// Ground 1× × Fire vs Flying 1× = 1×). High SpA + no resistance →
/// 30-HP Lando-T dies on the predicted hit.
///
/// Current (broken) heuristic picks Lando-T by type matchup, ignoring
/// the lethal incoming. New heuristic must pick Toxapex.
#[test]
fn switch_avoids_dying_to_predicted_opp_move() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::CORVIKNIGHT;
    state.side_one.active_index = PokemonIndex::P0;

    // P1 = Landorus-Therian. Ground/Flying. Low HP. Best type matchup
    // (Ground 4× SE on Heatran's Fire/Steel, takes neutral Magma Storm).
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::LANDORUSTHERIAN;
    state.side_one.pokemon[PokemonIndex::P1].types =
        (PokemonType::GROUND, PokemonType::FLYING);
    state.side_one.pokemon[PokemonIndex::P1].hp = 30;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 319;
    state.side_one.pokemon[PokemonIndex::P1].defense = 200;
    state.side_one.pokemon[PokemonIndex::P1].special_defense = 200;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::EARTHQUAKE);

    // P2 = Toxapex. Poison/Water. Full HP. Worse matchup but survives.
    state.side_one.pokemon[PokemonIndex::P2].id = PokemonName::TOXAPEX;
    state.side_one.pokemon[PokemonIndex::P2].types =
        (PokemonType::POISON, PokemonType::WATER);
    state.side_one.pokemon[PokemonIndex::P2].hp = 300;
    state.side_one.pokemon[PokemonIndex::P2].maxhp = 300;
    state.side_one.pokemon[PokemonIndex::P2].defense = 245;
    state.side_one.pokemon[PokemonIndex::P2].special_defense = 245;
    state.side_one.pokemon[PokemonIndex::P2]
        .replace_move(PokemonMoveIndex::M0, Choices::SCALD);

    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;
    state.side_one.pokemon[PokemonIndex::P5].hp = 0;

    // Opp Heatran running Magma Storm — high BP, will OHKO 30-HP
    // Lando-T at high SpA roll, but cannot dent full-HP Toxapex.
    state.side_two.get_active().id = PokemonName::HEATRAN;
    state.side_two.get_active().types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_two.get_active().special_attack = 369;
    state.side_two.get_active().level = 100;
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::MAGMASTORM);

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(PokemonName::TOXAPEX),
        "should pick full-HP Toxapex over Lando-T that dies to predicted Magma Storm despite better type matchup"
    );
}

/// Every bench Pokemon dies on entry (3 spikes + low HP). The
/// heuristic must return None so the caller falls back to staying in.
#[test]
fn switch_returns_none_when_no_viable_candidate() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::GARCHOMP;
    state.side_one.active_index = PokemonIndex::P0;
    state.side_one.side_conditions.spikes = 3; // 25% maxhp on entry

    // All bench: grounded, HP < 25% maxhp → dies to spikes.
    for idx in [
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ] {
        state.side_one.pokemon[idx].id = PokemonName::TOXAPEX;
        state.side_one.pokemon[idx].types = (PokemonType::POISON, PokemonType::WATER);
        state.side_one.pokemon[idx].hp = 30; // 10% of 300
        state.side_one.pokemon[idx].maxhp = 300;
        state.side_one.pokemon[idx].item = Items::NONE; // not boots
        state.side_one.pokemon[idx]
            .replace_move(PokemonMoveIndex::M0, Choices::SCALD);
    }

    state.side_two.get_active().id = PokemonName::VOLCARONA;
    state.side_two.get_active().types = (PokemonType::BUG, PokemonType::FIRE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BUGBUZZ);

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(
        pick, None,
        "no viable switch — all candidates die on entry; expected None"
    );
}

/// Regression guard: when all bench mons survive entry and the
/// predicted incoming move, the heuristic must still pick by type
/// matchup (preserving the existing behavior of
/// `matchup_switch_picks_best_resist_profile`).
#[test]
fn switch_still_picks_best_matchup_when_all_viable() {
    let mut state = State::default();

    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.active_index = PokemonIndex::P0;

    // Bench[1] = Heatran (full HP, no hazards). Best resist vs Volcarona.
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].types = (PokemonType::FIRE, PokemonType::STEEL);
    state.side_one.pokemon[PokemonIndex::P1].hp = 385;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::STONEEDGE);

    // Bench[2] = Toxapex (full HP). Worse type matchup.
    state.side_one.pokemon[PokemonIndex::P2].id = PokemonName::TOXAPEX;
    state.side_one.pokemon[PokemonIndex::P2].types = (PokemonType::POISON, PokemonType::WATER);
    state.side_one.pokemon[PokemonIndex::P2].hp = 300;
    state.side_one.pokemon[PokemonIndex::P2].maxhp = 300;
    state.side_one.pokemon[PokemonIndex::P2]
        .replace_move(PokemonMoveIndex::M0, Choices::SCALD);

    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;
    state.side_one.pokemon[PokemonIndex::P5].hp = 0;

    state.side_two.get_active().id = PokemonName::VOLCARONA;
    state.side_two.get_active().types = (PokemonType::BUG, PokemonType::FIRE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BUGBUZZ);

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(
        pick,
        Some(PokemonName::HEATRAN),
        "all viable: must still pick by type matchup (Heatran beats Toxapex)"
    );
}
