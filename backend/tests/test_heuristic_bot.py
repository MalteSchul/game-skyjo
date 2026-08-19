from dataclasses import replace

import pytest

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.bots.random_bot import RandomBot
from skyjo.domain.engine import (
    Action,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
    apply_action,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn

# --- fixture helpers ---------------------------------------------------------


def _play_match_to_completion(
    state: GameState, bots: list[HeuristicBot | RandomBot], *, max_steps: int = 5000
) -> GameState:
    for _ in range(max_steps):
        if state.phase == "game_over":
            return state
        if state.phase == "round_over":
            state = start_next_round(state)
            continue

        turn = Turn.from_state(state)
        action = bots[turn.acting_player].choose_action(turn)
        state = apply_action(state, action)

    raise AssertionError(f"match did not reach game_over within {max_steps} steps")


def _turn_with_boards(
    boards: tuple[PlayerBoard, ...], *, drawn_card: int | None, phase: str = "awaiting_placement"
) -> Turn:
    state = new_match(player_count=len(boards), seed=1)
    state = replace(state, boards=boards, drawn_card=drawn_card, phase=phase)
    return Turn.from_state(state)


# --- choose_action: general ---------------------------------------------------


def test_choose_action_always_picks_a_legal_action():
    state = new_match(player_count=2, seed=1)
    bot = HeuristicBot(seed=1)

    for _ in range(200):
        if state.phase in ("round_over", "game_over"):
            break
        turn = Turn.from_state(state)
        action = bot.choose_action(turn)
        assert action in turn.legal_actions
        state = apply_action(state, action)


def test_same_seed_produces_the_same_sequence_of_actions():
    state = new_match(player_count=2, seed=2)
    turn = Turn.from_state(state)

    first = HeuristicBot(seed=7).choose_action(turn)
    second = HeuristicBot(seed=7).choose_action(turn)

    assert first == second


def test_rejects_non_int_seed():
    with pytest.raises(TypeError):
        HeuristicBot(seed="not-a-seed")  # type: ignore[arg-type]


def test_choose_action_accepts_and_ignores_a_report_progress_callback():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    progress_calls: list[float] = []

    action = HeuristicBot(seed=1).choose_action(turn, report_progress=progress_calls.append)

    assert action in turn.legal_actions
    assert progress_calls == []


# --- choose_action: draw heuristic --------------------------------------------


def test_draws_from_discard_when_it_is_a_low_card():
    state = new_match(player_count=2, seed=1)
    state = replace(state, phase="awaiting_draw", discard=(9, 2))
    turn = Turn.from_state(state)

    action = HeuristicBot(seed=1).choose_action(turn)

    assert action == Action(ActionType.DRAW_DISCARD)


def test_draws_from_discard_when_it_matches_an_own_face_up_card():
    boards = tuple(
        PlayerBoard(cards=(Card(value=8, face_up=True),) + (None,) * 11) if i == 0 else PlayerBoard(cards=(None,) * 12)
        for i in range(2)
    )
    state = new_match(player_count=2, seed=1)
    state = replace(state, boards=boards, phase="awaiting_draw", discard=(1, 8))
    turn = Turn.from_state(state)

    action = HeuristicBot(seed=1).choose_action(turn)

    assert action == Action(ActionType.DRAW_DISCARD)


def test_draws_from_stock_when_discard_is_high_and_unmatched():
    state = new_match(player_count=2, seed=1)
    state = replace(state, phase="awaiting_draw", discard=(1, 9))
    turn = Turn.from_state(state)

    action = HeuristicBot(seed=1).choose_action(turn)

    assert action == Action(ActionType.DRAW_STOCK)


# --- choose_action: placement heuristic ---------------------------------------


def test_swaps_drawn_card_into_the_worst_face_up_slot():
    board = PlayerBoard(cards=(Card(value=11, face_up=True), Card(value=1, face_up=True)) + (None,) * 10)
    turn = _turn_with_boards((board, PlayerBoard(cards=(None,) * 12)), drawn_card=2)

    action = HeuristicBot(seed=1).choose_action(turn)

    assert action == Action(ActionType.PLACE, position=0)


def test_places_on_a_face_down_card_when_no_face_up_swap_helps():
    board = PlayerBoard(cards=(Card(value=1, face_up=True), Card(value=6, face_up=False)) + (None,) * 10)
    turn = _turn_with_boards((board, PlayerBoard(cards=(None,) * 12)), drawn_card=5)

    action = HeuristicBot(seed=1).choose_action(turn)

    assert action == Action(ActionType.PLACE, position=1)


def test_places_over_worst_slot_when_board_is_fully_face_up_and_drawn_card_does_not_help():
    board = PlayerBoard(cards=(Card(value=1, face_up=True), Card(value=2, face_up=True)) + (None,) * 10)
    turn = _turn_with_boards((board, PlayerBoard(cards=(None,) * 12)), drawn_card=10)

    action = HeuristicBot(seed=1).choose_action(turn)

    assert action == Action(ActionType.PLACE, position=1)


# --- integration ---------------------------------------------------------------


def test_two_heuristic_bots_can_play_a_full_match_to_completion():
    state = new_match(player_count=2, seed=3)
    bots = [HeuristicBot(seed=10), HeuristicBot(seed=11)]

    final_state = _play_match_to_completion(state, bots)

    assert final_state.phase == "game_over"
    assert any(score >= final_state.target_score for score in final_state.total_scores)


def test_heuristic_bot_beats_random_bot_on_average():
    heuristic_wins = 0
    matches = 20
    for seed in range(matches):
        state = new_match(player_count=2, seed=seed)
        bots = [HeuristicBot(seed=seed), RandomBot(seed=seed + 1000)]
        final_state = _play_match_to_completion(state, bots)
        if final_state.total_scores[0] < final_state.total_scores[1]:
            heuristic_wins += 1

    assert heuristic_wins > matches / 2
