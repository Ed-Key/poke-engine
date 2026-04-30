//! Plan I: heuristic_prior module unit tests.

use poke_engine::choices::Choices;
use poke_engine::heuristic_prior::damage_calc_top_move;
use poke_engine::nn_state_encoder::SidePerspective;
use poke_engine::pokemon::PokemonName;
use poke_engine::state::{PokemonMoveIndex, State};

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
