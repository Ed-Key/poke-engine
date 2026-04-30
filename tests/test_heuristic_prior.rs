//! Plan I: heuristic_prior module unit tests.

use poke_engine::choices::Choices;
use poke_engine::heuristic_prior::{damage_calc_top_move, matchup_switch_pick};
use poke_engine::nn_state_encoder::SidePerspective;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonIndex, PokemonMoveIndex, State};

/// Garchomp vs Heatran: EQ is 4× super-effective, deals far more than
/// Dragon Claw / Stone Edge. Damage-calc heuristic must pick Earthquake.
#[test]
fn damage_calc_top_move_picks_super_effective_ko() {
    let mut state = State::default();
    state.side_one.get_active().id = PokemonName::GARCHOMP;
    state.side_one.get_active().types = (
        poke_engine::state::PokemonType::DRAGON,
        poke_engine::state::PokemonType::GROUND,
    );
    state.side_one.get_active().attack = 359;
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::DRAGONCLAW);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::EARTHQUAKE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::STONEEDGE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M3, Choices::SWORDSDANCE);

    state.side_two.get_active().id = PokemonName::HEATRAN;
    state.side_two.get_active().types = (
        poke_engine::state::PokemonType::FIRE,
        poke_engine::state::PokemonType::STEEL,
    );
    state.side_two.get_active().hp = 385;
    state.side_two.get_active().maxhp = 385;
    state.side_two.get_active().defense = 248;
    state.side_two.get_active().special_defense = 384;

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(pick, Some(Choices::EARTHQUAKE));
}

/// Two damaging moves with identical damage output: pick the one with
/// higher base_power × accuracy. The MOVES table mutates BPs based on
/// gen features, so picking move IDs alone wasn't enough to guarantee
/// a damage tie. Instead we pick TACKLE and POUND (both 40 BP / Normal /
/// Physical / 100% acc by default) and *override* TACKLE's accuracy to
/// 80% directly on the Move struct, making the damages tie exactly.
/// BP×acc: TACKLE 40×0.8 = 32, POUND 40×1.0 = 40 → POUND wins tiebreak.
///
/// NOTE: Spec originally proposed Body Slam vs Hyper Beam, then we tried
/// COLLISIONCOURSE vs HIGHJUMPKICK, but neither put the damages within
/// the 10% window for gen9 BP values. The current fixture engineers an
/// exact tie by overriding `choice.accuracy` post-`replace_move`.
#[test]
fn damage_calc_top_move_tiebreaks_with_base_power_x_acc() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::POUND);

    // Force the M0 (TACKLE) move to have lower accuracy than M1 (POUND)
    // so the two damages tie but BP×acc differs.
    state.side_one.get_active().moves.m0.choice.accuracy = 80.0;

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(pick, Some(Choices::POUND));
}

/// All legal moves are status moves: returns None.
#[test]
fn damage_calc_top_move_returns_none_when_only_status_moves() {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SWORDSDANCE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::RECOVER);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M2, Choices::TAUNT);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M3, Choices::PROTECT);

    let pick = damage_calc_top_move(&state, SidePerspective::Side1);
    assert_eq!(pick, None);
}

/// Bench has Heatran + Toxapex; opp is Volcarona (Bug/Fire) running Bug
/// Buzz. Heatran (Fire/Steel) takes 0.25× from Bug (Steel ½× × Fire ½×)
/// and retaliates with Stone Edge — Rock vs Bug = 2× and Rock vs Fire =
/// 2× → outgoing 4.0. Toxapex (Poison/Water) takes 0.5× from Bug
/// (Poison ½× × Water 1×) and retaliates with Scald — Water vs Bug = 1×
/// × Water vs Fire = 2× → outgoing 2.0. Heatran score 0.25 − 4.0 =
/// −3.75 vs Toxapex 0.5 − 2.0 = −1.5; lower wins, so Heatran is picked.
///
/// NOTE: The spec's original fixture used Earth Power (Heatran) and
/// added Fiery Dance to Volcarona, asserting Heatran wins. Two type-chart
/// errors in the spec docstring caused that fixture to actually pick
/// Toxapex: (1) Steel does NOT resist Fire — it takes 2× from Fire (so
/// Heatran has no Fire-incoming advantage); (2) Ground vs Bug is 0.5×
/// (gen 6+ chart), so Earth Power against Bug/Fire is only neutral
/// (0.5 × 2.0 = 1.0), not 2× SE. Switching Heatran's attack to Stone
/// Edge gives a clean 4× outgoing that dominates the score difference,
/// preserving the spec's intent ("Heatran's matchup wins") under the
/// formula as implemented.
#[test]
fn matchup_switch_picks_best_resist_profile() {
    let mut state = State::default();

    // Active is Dragonite (P0). Bench[1]=Heatran, Bench[2]=Toxapex.
    state.side_one.get_active().id = PokemonName::DRAGONITE;
    state.side_one.active_index = PokemonIndex::P0;

    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].types = (
        poke_engine::state::PokemonType::FIRE,
        poke_engine::state::PokemonType::STEEL,
    );
    state.side_one.pokemon[PokemonIndex::P1].hp = 385;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;
    state.side_one.pokemon[PokemonIndex::P1]
        .replace_move(PokemonMoveIndex::M0, Choices::STONEEDGE);

    state.side_one.pokemon[PokemonIndex::P2].id = PokemonName::TOXAPEX;
    state.side_one.pokemon[PokemonIndex::P2].types = (
        poke_engine::state::PokemonType::POISON,
        poke_engine::state::PokemonType::WATER,
    );
    state.side_one.pokemon[PokemonIndex::P2].hp = 100;
    state.side_one.pokemon[PokemonIndex::P2].maxhp = 100;
    state.side_one.pokemon[PokemonIndex::P2]
        .replace_move(PokemonMoveIndex::M0, Choices::SCALD);

    state.side_two.get_active().id = PokemonName::VOLCARONA;
    state.side_two.get_active().types = (
        poke_engine::state::PokemonType::BUG,
        poke_engine::state::PokemonType::FIRE,
    );
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::BUGBUZZ);

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(pick, Some(PokemonName::HEATRAN));
}

#[test]
fn matchup_switch_returns_none_when_force_trapped() {
    let mut state = State::default();
    state.side_one.force_trapped = true;
    state.side_one.pokemon[PokemonIndex::P1].id = PokemonName::HEATRAN;
    state.side_one.pokemon[PokemonIndex::P1].hp = 385;
    state.side_one.pokemon[PokemonIndex::P1].maxhp = 385;

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(pick, None);
}

#[test]
fn matchup_switch_returns_none_when_last_alive() {
    let mut state = State::default();
    // Default state has all 6 Pokemon at hp=100 (alive). Faint everyone
    // except P0 so the only living Pokemon is the active and there is
    // nobody to switch to.
    state.side_one.pokemon[PokemonIndex::P1].hp = 0;
    state.side_one.pokemon[PokemonIndex::P2].hp = 0;
    state.side_one.pokemon[PokemonIndex::P3].hp = 0;
    state.side_one.pokemon[PokemonIndex::P4].hp = 0;
    state.side_one.pokemon[PokemonIndex::P5].hp = 0;

    let pick = matchup_switch_pick(&state, SidePerspective::Side1);
    assert_eq!(pick, None);
}
