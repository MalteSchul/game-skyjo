import json
from dataclasses import replace
from pathlib import Path

import pytest

from skyjo.domain.engine import (
    BOARD_SIZE,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
    apply_action,
    legal_actions,
    new_match,
)
from skyjo.domain.engine import legal_actions as state_legal_actions
from skyjo.domain.state_json import game_state_from_dict, game_state_to_dict

EXAMPLE_STATES_DIR = Path(__file__).parent.parent / "scripts" / "example_states"

# --- fixture helpers -----------------------------------------------------------


def _board_from_values(values, *, face_up: bool = False) -> PlayerBoard:
    assert len(values) == BOARD_SIZE
    return PlayerBoard(cards=tuple(Card(value=v, face_up=face_up) for v in values))


# --- happy path -----------------------------------------------------------------


def test_round_trip_preserves_a_fresh_new_match_state():
    state = new_match(player_count=3, seed=1)

    restored = game_state_from_dict(game_state_to_dict(state))

    assert restored == state


def test_round_trip_preserves_an_advanced_state_with_hidden_and_revealed_cards():
    state = new_match(player_count=2, seed=5)
    while state.phase == "initial_flip":
        action = next(a for a in state_legal_actions(state) if a.type is ActionType.FLIP_INITIAL)
        state = apply_action(state, action)

    restored = game_state_from_dict(game_state_to_dict(state))

    assert restored == state
    assert any(card is not None and card.face_up for board in state.boards for card in board.cards)
    assert any(card is not None and not card.face_up for board in state.boards for card in board.cards)


def test_round_trip_preserves_none_cards_and_none_round_scores():
    board = replace(_board_from_values([1] * BOARD_SIZE), cards=(None,) * BOARD_SIZE)
    state = GameState(
        boards=(board, board),
        stock=(1, 2, 3),
        discard=(4,),
        current_player=0,
        drawn_card=None,
        finisher=None,
        players_awaiting_final_turn=frozenset(),
        round_scores=None,
        total_scores=(0, 0),
        phase="awaiting_draw",
        reshuffle_seed=None,
    )

    restored = game_state_from_dict(game_state_to_dict(state))

    assert restored == state


def test_dict_is_json_serializable_end_to_end():
    state = new_match(player_count=2, seed=2)

    dumped = json.dumps(game_state_to_dict(state))
    restored = game_state_from_dict(json.loads(dumped))

    assert restored == state


def test_players_awaiting_final_turn_round_trips_as_a_frozenset():
    state = new_match(player_count=3, seed=1)
    state = replace(state, players_awaiting_final_turn=frozenset({0, 2}))

    restored = game_state_from_dict(game_state_to_dict(state))

    assert restored.players_awaiting_final_turn == frozenset({0, 2})


def test_drawn_card_source_round_trips():
    state = new_match(player_count=2, seed=1)
    state = replace(state, phase="awaiting_placement", drawn_card=7, drawn_card_source="discard")

    restored = game_state_from_dict(game_state_to_dict(state))

    assert restored.drawn_card_source == "discard"


def test_drawn_card_source_defaults_to_none_when_absent_from_the_dict():
    # Guards a state file dumped before this field existed (e.g. the checked-in
    # example_states fixtures) - it should still load, as an unrestricted state.
    state = new_match(player_count=2, seed=1)
    data = game_state_to_dict(state)
    del data["drawn_card_source"]

    restored = game_state_from_dict(data)

    assert restored.drawn_card_source is None


# --- checked-in example fixtures -----------------------------------------------


def test_example_fixture_midgame_awaiting_draw_loads_correctly():
    data = json.loads((EXAMPLE_STATES_DIR / "midgame_awaiting_draw.json").read_text())

    state = game_state_from_dict(data)

    # Matches this file's own derivation recipe (see example_states/README.md):
    # both players finish their two initial flips, seed=5. If this ever
    # drifts, either the fixture is stale or GameState/state_json changed
    # incompatibly - regenerate the fixture per the README, don't just widen
    # this assertion.
    expected = new_match(player_count=2, seed=5)
    while expected.phase == "initial_flip":
        action = next(a for a in legal_actions(expected) if a.type is ActionType.FLIP_INITIAL)
        expected = apply_action(expected, action)

    assert state == expected
    assert legal_actions(state)  # still a playable position


# --- bad path -----------------------------------------------------------------


def test_from_dict_raises_on_a_missing_required_field():
    state = new_match(player_count=2, seed=1)
    data = game_state_to_dict(state)
    del data["current_player"]

    with pytest.raises(KeyError):
        game_state_from_dict(data)
