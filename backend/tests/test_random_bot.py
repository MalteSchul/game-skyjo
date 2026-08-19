import pytest

from skyjo.bots.random_bot import RandomBot
from skyjo.domain.engine import (
    GameState,
    apply_action,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn

# --- fixture helpers ---------------------------------------------------------


def _play_match_to_completion(state: GameState, bots: list[RandomBot], *, max_steps: int = 5000) -> GameState:
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


# --- choose_action -----------------------------------------------------------


def test_choose_action_always_picks_a_legal_action():
    state = new_match(player_count=2, seed=1)
    bot = RandomBot(seed=1)

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

    first = RandomBot(seed=7).choose_action(turn)
    second = RandomBot(seed=7).choose_action(turn)

    assert first == second


def test_rejects_non_int_seed():
    with pytest.raises(TypeError):
        RandomBot(seed="not-a-seed")  # type: ignore[arg-type]


def test_choose_action_accepts_and_ignores_a_report_progress_callback():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    progress_calls: list[float] = []

    action = RandomBot(seed=1).choose_action(turn, report_progress=progress_calls.append)

    assert action in turn.legal_actions
    assert progress_calls == []


# --- integration: a full match played entirely by RandomBot ------------------


def test_two_random_bots_can_play_a_full_match_to_completion():
    state = new_match(player_count=2, seed=3)
    bots = [RandomBot(seed=10), RandomBot(seed=11)]

    final_state = _play_match_to_completion(state, bots)

    assert final_state.phase == "game_over"
    assert any(score >= final_state.target_score for score in final_state.total_scores)


def test_four_random_bots_can_play_a_full_match_to_completion():
    state = new_match(player_count=4, seed=4)
    bots = [RandomBot(seed=i) for i in range(4)]

    final_state = _play_match_to_completion(state, bots)

    assert final_state.phase == "game_over"
