import numpy as np
import pytest

from skyjo.bots.mcts_bot import CHECKPOINT_PATH_ENV_VAR, MctsBot, default_evaluator
from skyjo.domain.engine import GameState, apply_action, new_match
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.network import AlphaZeroNet

# --- fixture helpers -----------------------------------------------------------


def _uniform_evaluate(state: GameState):
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


# --- choose_action -----------------------------------------------------------


def test_choose_action_always_picks_a_legal_action():
    state = new_match(player_count=2, seed=1)
    bot = MctsBot(_uniform_evaluate, num_simulations=4, seed=1)

    for _ in range(50):
        if state.phase in ("round_over", "game_over"):
            break
        turn = Turn.from_state(state)
        action = bot.choose_action(turn)
        assert action in turn.legal_actions
        state = apply_action(state, action)


def test_same_seed_produces_the_same_action():
    state = new_match(player_count=2, seed=2)
    turn = Turn.from_state(state)

    first = MctsBot(_uniform_evaluate, num_simulations=8, seed=7).choose_action(turn)
    second = MctsBot(_uniform_evaluate, num_simulations=8, seed=7).choose_action(turn)

    assert first == second


def test_rejects_non_int_seed():
    with pytest.raises(TypeError):
        MctsBot(_uniform_evaluate, seed="not-a-seed")  # type: ignore[arg-type]


def test_rejects_a_negative_num_simulations():
    with pytest.raises(ValueError):
        MctsBot(_uniform_evaluate, num_simulations=-1)


def test_choose_action_reports_progress_up_to_1_when_simulating():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    progress_calls: list[float] = []

    action = MctsBot(_uniform_evaluate, num_simulations=6, seed=1).choose_action(
        turn, report_progress=progress_calls.append
    )

    assert action in turn.legal_actions
    assert progress_calls == sorted(progress_calls)
    assert progress_calls[-1] == 1.0
    assert len(progress_calls) == 6


def test_choose_action_still_reports_a_final_1_with_zero_simulations():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    progress_calls: list[float] = []

    action = MctsBot(_uniform_evaluate, num_simulations=0, seed=1).choose_action(
        turn, report_progress=progress_calls.append
    )

    assert action in turn.legal_actions
    assert progress_calls == [1.0]


def test_choose_action_works_without_a_report_progress_callback():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    action = MctsBot(_uniform_evaluate, num_simulations=4, seed=1).choose_action(turn)

    assert action in turn.legal_actions


# --- integration: real (untrained) network end-to-end -------------------------


def test_works_end_to_end_with_a_real_untrained_network():
    """Not just MctsBot's own logic against a fake evaluator: this exercises
    the real AlphaZeroNet -> make_network_evaluator -> run_mcts path, the
    piece that actually proves the mcts_bot wiring works."""
    state = new_match(player_count=2, seed=1)
    evaluate = make_network_evaluator(AlphaZeroNet())
    bot = MctsBot(evaluate, num_simulations=3, seed=1)

    turn = Turn.from_state(state)
    action = bot.choose_action(turn)

    assert action in turn.legal_actions


# --- default_evaluator() / checkpoint loading ---------------------------------


def test_default_evaluator_loads_a_real_training_checkpoint(tmp_path, monkeypatch):
    """Regression test: skyjo.rl.checkpoint.save_checkpoint (what
    scripts/train_mcts.py actually writes to --checkpoint-dir) wraps the
    net's state_dict inside a payload with optimizer/iteration/extra keys -
    default_evaluator must unwrap that via load_checkpoint, not hand the
    whole payload straight to net.load_state_dict."""
    trained_net = AlphaZeroNet()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, trained_net, None, iteration=1, total_train_steps=1)
    monkeypatch.setenv(CHECKPOINT_PATH_ENV_VAR, str(checkpoint_path))
    default_evaluator.cache_clear()

    try:
        evaluate = default_evaluator()
        priors, value = evaluate(new_match(player_count=2, seed=1))
    finally:
        default_evaluator.cache_clear()

    assert priors
    assert value.shape == (2,)


def test_default_evaluator_raises_for_a_bad_checkpoint_path(tmp_path, monkeypatch):
    monkeypatch.setenv(CHECKPOINT_PATH_ENV_VAR, str(tmp_path / "does_not_exist.pt"))
    default_evaluator.cache_clear()

    try:
        with pytest.raises(FileNotFoundError):
            default_evaluator()
    finally:
        default_evaluator.cache_clear()


# --- integration: many consecutive turns, alternating seats -------------------


def test_two_mcts_bots_can_play_many_consecutive_turns_without_crashing():
    # Not a full-match-to-completion test (unlike RandomBot's/ThinkingBot's):
    # against an untrained/uninformative evaluator, a low-simulation MCTS
    # search can take an unbounded number of turns to happen to close a
    # round by chance, since it has no learned preference for matching
    # values. This instead checks the same thing that matters for the API
    # integration - many consecutive real bot-vs-bot decisions, across every
    # phase a round passes through, never produce an illegal action.
    state = new_match(player_count=2, seed=3)
    bots = [MctsBot(_uniform_evaluate, num_simulations=3, seed=10), MctsBot(_uniform_evaluate, num_simulations=3, seed=11)]

    for _ in range(300):
        if state.phase in ("round_over", "game_over"):
            break
        turn = Turn.from_state(state)
        action = bots[turn.acting_player].choose_action(turn)
        assert action in turn.legal_actions
        state = apply_action(state, action)
