import time

import pytest

from skyjo.bots.thinking_bot import ThinkingBot
from skyjo.domain.engine import (
    GameState,
    apply_action,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn

# --- fixture helpers ---------------------------------------------------------


def _play_match_to_completion(state: GameState, bots: list[ThinkingBot], *, max_steps: int = 5000) -> GameState:
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
    bot = ThinkingBot(seed=1, think_seconds=0)

    for _ in range(200):
        if state.phase in ("round_over", "game_over"):
            break
        turn = Turn.from_state(state)
        action = bot.choose_action(turn)
        assert action in turn.legal_actions
        state = apply_action(state, action)


def test_same_seed_produces_the_same_action():
    state = new_match(player_count=2, seed=2)
    turn = Turn.from_state(state)

    first = ThinkingBot(seed=7, think_seconds=0).choose_action(turn)
    second = ThinkingBot(seed=7, think_seconds=0).choose_action(turn)

    assert first == second


def test_rejects_non_int_seed():
    with pytest.raises(TypeError):
        ThinkingBot(seed="not-a-seed")  # type: ignore[arg-type]


def test_rejects_a_negative_think_seconds():
    with pytest.raises(ValueError):
        ThinkingBot(think_seconds=-1)


def test_choose_action_takes_roughly_think_seconds_and_reports_progress_up_to_1():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    progress_calls: list[float] = []

    started = time.monotonic()
    action = ThinkingBot(seed=1, think_seconds=0.2).choose_action(turn, report_progress=progress_calls.append)
    elapsed = time.monotonic() - started

    assert action in turn.legal_actions
    assert elapsed >= 0.18  # small tolerance for scheduling jitter
    assert progress_calls == sorted(progress_calls)
    assert progress_calls[-1] == 1.0


def test_choose_action_still_thinks_without_a_report_progress_callback():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    started = time.monotonic()
    action = ThinkingBot(seed=1, think_seconds=0.1).choose_action(turn)
    elapsed = time.monotonic() - started

    assert action in turn.legal_actions
    assert elapsed >= 0.09


# --- integration: a full match played entirely by ThinkingBot ---------------


def test_two_thinking_bots_can_play_a_full_match_to_completion():
    state = new_match(player_count=2, seed=3)
    bots = [ThinkingBot(seed=10, think_seconds=0), ThinkingBot(seed=11, think_seconds=0)]

    final_state = _play_match_to_completion(state, bots)

    assert final_state.phase == "game_over"
    assert any(score >= final_state.target_score for score in final_state.total_scores)
