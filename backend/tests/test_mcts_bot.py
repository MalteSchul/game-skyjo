from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from skyjo.bots.mcts_bot import CHECKPOINT_PATH_ENV_VAR, MctsBot, default_evaluator
from skyjo.domain.engine import (
    Action,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
    apply_action,
    new_match,
)
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.hidden_info import gamestate_from_turn
from skyjo.rl.mcts import ChanceEdge, ChanceNode, MCTSEdge, MCTSNode
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


# --- observe_transition: tree reuse --------------------------------------------


def _awaiting_draw_state(seed: int) -> GameState:
    state = new_match(player_count=2, seed=seed)
    while state.phase == "initial_flip":
        state = apply_action(state, engine_legal_actions(state)[0])
    return state


def _near_closing_state(*, target_score: int) -> GameState:
    board0 = PlayerBoard(cards=tuple(Card(value=v, face_up=True) for v in range(12)))
    board1 = PlayerBoard(cards=tuple(Card(value=1, face_up=True) for _ in range(9)) + (Card(3, True), Card(4, True), Card(5, True)))
    board1 = replace(
        board1, cards=board1.cards[:9] + tuple(replace(c, face_up=False) for c in board1.cards[9:])
    )
    return GameState(
        boards=(board0, board1),
        stock=(5, 6, 7),
        discard=(2,),
        current_player=1,
        drawn_card=9,
        finisher=0,
        players_awaiting_final_turn=frozenset({1}),
        round_scores=None,
        total_scores=(66, 0),
        phase="awaiting_placement",
        reshuffle_seed=None,
        target_score=target_score,
    )


def test_observe_transition_is_a_no_op_when_there_is_no_cached_root():
    bot = MctsBot(_uniform_evaluate, num_simulations=1, seed=1)
    turn = Turn.from_state(_awaiting_draw_state(seed=1))

    bot.observe_transition(turn, turn.legal_actions[0], turn)  # must not raise

    assert bot._cached_root is None


def test_observe_transition_clears_the_cache_when_turn_before_does_not_match():
    bot = MctsBot(_uniform_evaluate, num_simulations=1, seed=1)
    turn = Turn.from_state(_awaiting_draw_state(seed=1))
    bot._cached_root = MCTSNode(state=gamestate_from_turn(turn), n_act=2, is_terminal=False)

    other_turn = Turn.from_state(_awaiting_draw_state(seed=2))
    bot.observe_transition(other_turn, other_turn.legal_actions[0], other_turn)

    assert bot._cached_root is None


def test_observe_transition_advances_the_cache_through_a_deterministic_action():
    # DRAW_DISCARD is never a reveal (the discard top is already public) -
    # this exercises the plain edge.child path, no ChanceNode involved.
    state = _awaiting_draw_state(seed=1)
    turn_before = Turn.from_state(state)
    action = Action(type=ActionType.DRAW_DISCARD)
    assert action in turn_before.legal_actions
    bot = MctsBot(_uniform_evaluate, num_simulations=10, seed=1)
    bot.choose_action(turn_before)
    edge = bot._cached_root.edges[action]
    assert edge.child is not None  # sanity: DRAW_DISCARD was actually visited

    turn_after = Turn.from_state(apply_action(state, action))
    bot.observe_transition(turn_before, action, turn_after)

    assert bot._cached_root is edge.child


def test_observe_transition_advances_the_cache_through_a_matched_reveal():
    state = _awaiting_draw_state(seed=1)
    turn_before = Turn.from_state(state)
    action = Action(type=ActionType.DRAW_STOCK)
    turn_after = Turn.from_state(apply_action(state, action))
    real_value = turn_after.drawn_card
    assert real_value is not None

    bot = MctsBot(_uniform_evaluate, num_simulations=1, seed=1)
    cached_state = gamestate_from_turn(turn_before)
    root = MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    expected_child = MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    chance = ChanceNode(n_act=2, counts=Counter({real_value: 1}))
    chance.edges[real_value] = ChanceEdge(value=real_value, prior=1.0, n_act=2, child=expected_child)
    root.edges[action] = MCTSEdge(action=action, prior=1.0, n_act=2, child=chance)
    bot._cached_root = root

    bot.observe_transition(turn_before, action, turn_after)

    assert bot._cached_root is expected_child


def test_observe_transition_clears_the_cache_when_the_real_reveal_was_never_visited():
    state = _awaiting_draw_state(seed=1)
    turn_before = Turn.from_state(state)
    action = Action(type=ActionType.DRAW_STOCK)
    turn_after = Turn.from_state(apply_action(state, action))
    real_value = turn_after.drawn_card
    other_value = next(v for v in range(-2, 13) if v != real_value)

    bot = MctsBot(_uniform_evaluate, num_simulations=1, seed=1)
    cached_state = gamestate_from_turn(turn_before)
    root = MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    chance = ChanceNode(n_act=2, counts=Counter({other_value: 1}))
    chance.edges[other_value] = ChanceEdge(
        value=other_value, prior=1.0, n_act=2, child=MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    )
    root.edges[action] = MCTSEdge(action=action, prior=1.0, n_act=2, child=chance)
    bot._cached_root = root

    bot.observe_transition(turn_before, action, turn_after)

    assert bot._cached_root is None


def test_observe_transition_clears_the_cache_when_the_edge_was_never_expanded():
    state = _awaiting_draw_state(seed=1)
    turn_before = Turn.from_state(state)
    action = Action(type=ActionType.DRAW_DISCARD)

    bot = MctsBot(_uniform_evaluate, num_simulations=1, seed=1)
    cached_state = gamestate_from_turn(turn_before)
    root = MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    root.edges[action] = MCTSEdge(action=action, prior=1.0, n_act=2)  # child left None: never visited
    bot._cached_root = root

    turn_after = Turn.from_state(apply_action(state, action))
    bot.observe_transition(turn_before, action, turn_after)

    assert bot._cached_root is None


def test_observe_transition_clears_the_cache_for_a_round_closing_action():
    state = _near_closing_state(target_score=50)
    turn_before = Turn.from_state(state)
    action = Action(ActionType.PLACE, position=11)

    bot = MctsBot(_uniform_evaluate, num_simulations=1, seed=1)
    cached_state = gamestate_from_turn(turn_before)
    root = MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    root.edges[action] = MCTSEdge(
        action=action, prior=1.0, n_act=2, child=MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    )
    bot._cached_root = root

    # game_over follows this action (target_score=50 is already exceeded) -
    # no legal actions left, so there's no real turn_after to build.
    bot.observe_transition(turn_before, action, turn_before)

    assert bot._cached_root is None


def test_choose_action_reuses_the_cached_tree_after_a_matching_observe_transition():
    # num_simulations is a *target total* per decision, not "always this many
    # more": the reused child already carries some visits from the parent's
    # own 10-simulation search, so the second choose_action should only need
    # to run the shortfall to reach 10 total, not add another 10 on top.
    state = _awaiting_draw_state(seed=1)
    turn_before = Turn.from_state(state)
    action = Action(type=ActionType.DRAW_DISCARD)
    bot = MctsBot(_uniform_evaluate, num_simulations=10, seed=1)

    bot.choose_action(turn_before)
    edge = bot._cached_root.edges[action]
    expected_prior_visits = edge.child.visit_count
    assert expected_prior_visits < 10  # sanity: a genuine shortfall exists to fill

    turn_after = Turn.from_state(apply_action(state, action))
    bot.observe_transition(turn_before, action, turn_after)
    assert bot._cached_root is edge.child  # sanity: reuse actually kicked in

    bot.choose_action(turn_after)

    assert bot._cached_root.visit_count == 10


def test_choose_action_progress_reflects_visits_already_carried_over_on_reuse():
    state = _awaiting_draw_state(seed=1)
    turn_before = Turn.from_state(state)
    action = Action(type=ActionType.DRAW_DISCARD)
    bot = MctsBot(_uniform_evaluate, num_simulations=10, seed=1)
    bot.choose_action(turn_before)
    already_visited = bot._cached_root.edges[action].child.visit_count
    assert already_visited < 10  # sanity: a genuine shortfall exists to fill

    turn_after = Turn.from_state(apply_action(state, action))
    bot.observe_transition(turn_before, action, turn_after)
    progress_calls: list[float] = []

    bot.choose_action(turn_after, report_progress=progress_calls.append)

    # Progress should account for the visits already carried over, not
    # restart from 1/10 as if nothing had been reused.
    assert len(progress_calls) == 10 - already_visited
    assert progress_calls[0] == pytest.approx((already_visited + 1) / 10)
    assert progress_calls[-1] == 1.0


def test_choose_action_runs_zero_new_simulations_when_the_reused_root_already_meets_the_target():
    turn = Turn.from_state(_awaiting_draw_state(seed=1))
    cached_state = gamestate_from_turn(turn)
    action = turn.legal_actions[0]
    root = MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    root.edges[action] = MCTSEdge(
        action=action, prior=1.0, n_act=2, visit_count=50, child=MCTSNode(state=cached_state, n_act=2, is_terminal=False)
    )
    bot = MctsBot(_uniform_evaluate, num_simulations=5, seed=1)  # target well below the 50 already cached
    bot._cached_root = root
    progress_calls: list[float] = []

    result_action = bot.choose_action(turn, report_progress=progress_calls.append)

    assert bot._cached_root is root  # unchanged: run_mcts(num_simulations=0) returns it as-is
    assert bot._cached_root.visit_count == 50  # no new simulations were added
    assert progress_calls == [1.0]
    assert result_action in turn.legal_actions


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
