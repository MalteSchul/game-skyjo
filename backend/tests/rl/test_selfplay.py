from dataclasses import replace

import numpy as np
import pytest

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.domain.engine import Action, ActionType, new_match
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.rl.action_space import ACTION_SPACE_SIZE, action_to_index
from skyjo.rl.selfplay import generate_bot_episode, generate_episode, generate_episodes_batch


def _uniform_evaluate(state):
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


def _batch_evaluate(states):
    return [_uniform_evaluate(s) for s in states]


def _degenerate_place_zero_evaluate(state):
    """Deterministically prefers DRAW_STOCK, then PLACE at position 0 - never
    reveals anything past the initial two flips and position 0, so a round
    genuinely never closes on its own (nothing else on the board ever
    becomes face-up or clears). Real, reliable trap for exercising
    round_max_steps/max_rounds, not a contrived assertion about internals.
    """
    actions = engine_legal_actions(state)
    preferred = next((a for a in actions if a.type == ActionType.DRAW_STOCK), None)
    if preferred is None:
        preferred = next((a for a in actions if a.type == ActionType.PLACE and a.position == 0), None)
    if preferred is None:
        preferred = actions[0]
    priors = {a: (1.0 if a == preferred else 0.0) for a in actions}
    return priors, np.zeros(len(state.boards))


def _swap_into_position_zero(turn: Turn) -> Action:
    """Always takes the discard over the stock, and always tries to PLACE it
    at position 0 regardless of whether that's an improvement - TODO.md's
    "take the discard, swap it for a same-or-near-value card, discard that;
    the opposing bot mirrors it" pattern in its most literal form, as a
    direct `Turn -> Action` callable (no MCTS/network involved, so nothing
    about it is randomized by root noise - see
    `test_generate_bot_episode_finishes_despite_two_players_mirroring_a_discard_swap`).
    Before `engine.legal_actions` blocked a discard-sourced non-improving
    swap, two players both running this could keep trading position 0's
    value back and forth without ever revealing a new face-down card, i.e.
    never finishing the round on their own.
    """
    if turn.phase == "initial_flip":
        return next(a for a in turn.legal_actions if a.type == ActionType.FLIP_INITIAL)
    if turn.phase == "awaiting_draw":
        draw_discard = Action(ActionType.DRAW_DISCARD)
        return draw_discard if draw_discard in turn.legal_actions else Action(ActionType.DRAW_STOCK)
    preferred = Action(ActionType.PLACE, position=0)
    return preferred if preferred in turn.legal_actions else turn.legal_actions[0]


# --- happy path ------------------------------------------------------------


def test_generate_episode_produces_consistent_samples_for_a_full_match():
    state = new_match(player_count=2, seed=1)

    samples = generate_episode(
        state, _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(0), max_steps=3000
    )

    assert len(samples) > 0
    for sample in samples:
        assert sample.n_act == 2
        assert sample.pi.shape == (ACTION_SPACE_SIZE,)
        assert sample.pi.sum() == pytest.approx(1.0, abs=1e-4)
        assert sample.y.shape == (2,)
        assert sample.points_y.shape == (2,)
    # the final ranks/points are the same object recorded at every decision step
    assert {tuple(s.y) for s in samples} == {tuple(samples[-1].y)}
    assert {tuple(s.points_y.tolist()) for s in samples} == {tuple(samples[-1].points_y.tolist())}
    assert sorted(samples[-1].y.tolist()) == [0, 1]  # a strict permutation for 2 players
    # y and points_y must agree on who did better: the lower-ranked (y=0) player
    # has the lower (better, since Skyjo scores low-to-win) final points.
    better, worse = list(samples[-1].y).index(0), list(samples[-1].y).index(1)
    assert samples[-1].points_y[better] <= samples[-1].points_y[worse]


def test_generate_episode_accepts_a_callable_tau_schedule():
    # A near-zero tau combined with an evaluator that gives no real signal (as
    # here) can make deterministic argmax play cycle indefinitely - that's a
    # property of the degenerate test evaluator, not of generate_episode, so
    # this only exercises that the callable-schedule code path runs; the
    # near-zero-tau behavior itself is unit-tested directly against
    # visit_distribution in test_mcts.py.
    state = new_match(player_count=3, seed=2)
    calls: list[int] = []

    def tau_schedule(step: int) -> float:
        calls.append(step)
        return 1.0

    samples = generate_episode(
        state, _uniform_evaluate, num_simulations=2, tau_schedule=tau_schedule, rng=np.random.default_rng(1),
        max_steps=3000,
    )

    assert all(s.n_act == 3 for s in samples)
    assert sorted(samples[-1].y.tolist()) == [0, 1, 2]
    # one call per recorded decision; step skips ahead across round transitions
    # (which don't call tau_schedule), so calls is increasing but not contiguous.
    assert len(calls) == len(samples)
    assert calls == sorted(set(calls))


def test_generate_episode_widens_a_tied_representative_to_a_uniformly_sampled_real_position():
    # The very first decision (fully-unrevealed board) is one tied class
    # spanning all 12 positions, so search only ever scores a single fixed
    # representative for it - the recorded pi should point at that same
    # representative regardless of seed. The position actually flipped on
    # the real board should still vary across seeds, though: tied_actions
    # widens the applied action back out across the whole class instead of
    # always taking that same representative verbatim.
    first_decision_argmax = set()
    revealed_positions = set()
    for seed in range(40):
        state = new_match(player_count=2, seed=1)
        turn = Turn.from_state(state)
        samples = generate_episode(
            state,
            _uniform_evaluate,
            num_simulations=1,
            rng=np.random.default_rng(seed),
            max_steps=3000,
            round_max_steps=200,
            max_rounds=10,
        )
        first_decision_argmax.add(int(np.argmax(samples[0].pi)))
        after_board = samples[1].state.boards[turn.acting_player]
        flipped = next(i for i, c in enumerate(after_board.cards) if c is not None and c.face_up)
        revealed_positions.add(flipped)

    assert len(first_decision_argmax) == 1
    assert len(revealed_positions) > 1


# --- bad path ------------------------------------------------------------


def test_generate_episode_raises_if_the_match_does_not_finish_within_max_steps():
    state = new_match(player_count=2, seed=1)

    with pytest.raises(RuntimeError):
        generate_episode(state, _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0), max_steps=1)


# --- generate_episode: round_max_steps / max_rounds -------------------------


def test_generate_episode_force_closes_a_stuck_round_instead_of_raising():
    state = new_match(player_count=2, seed=1)

    samples = generate_episode(
        state,
        _degenerate_place_zero_evaluate,
        num_simulations=2,
        tau_schedule=0.0,
        rng=np.random.default_rng(0),
        max_steps=5000,
        round_max_steps=20,
        max_rounds=3,
    )

    assert len(samples) > 0
    # a strict permutation for 2 players, same as a naturally-finished game -
    # force-closing still produces a legitimate final outcome to label with.
    assert sorted(samples[-1].y.tolist()) == [0, 1]


def test_generate_episode_without_round_max_steps_hits_the_same_trap_generate_episode_force_close_solves():
    # Confirms the degenerate evaluator is genuinely pathological (not just
    # "the test didn't wait long enough") - with round-level forcing
    # effectively disabled, this is exactly today's failure mode.
    state = new_match(player_count=2, seed=1)

    with pytest.raises(RuntimeError):
        generate_episode(
            state,
            _degenerate_place_zero_evaluate,
            num_simulations=2,
            tau_schedule=0.0,
            rng=np.random.default_rng(0),
            max_steps=50,
            round_max_steps=100_000,
            max_rounds=100_000,
        )


def test_generate_episode_stops_after_max_rounds_even_short_of_target_score():
    # target_score set unreachably high so max_rounds - not target_score - is
    # what ends the game; round_max_steps is small so every round is forced.
    state = replace(new_match(player_count=2, seed=1), target_score=1_000_000)

    samples = generate_episode(
        state,
        _degenerate_place_zero_evaluate,
        num_simulations=2,
        tau_schedule=0.0,
        rng=np.random.default_rng(0),
        max_steps=5000,
        round_max_steps=10,
        max_rounds=4,
    )

    assert len(samples) > 0
    assert sum(samples[-1].y.tolist()) == 1  # still a valid 2-player rank permutation


# --- generate_bot_episode: the discard-swap restriction itself prevents a stall


def test_generate_bot_episode_finishes_despite_two_players_mirroring_a_discard_swap():
    # generate_bot_episode has no round_max_steps/force-close safety valve at
    # all (only the global max_steps) - reaching game_over here is possible
    # only because engine.legal_actions's discard-swap restriction itself
    # breaks the mirrored cycle, not because anything papers over a stall.
    state = new_match(player_count=2, seed=1)

    samples = generate_bot_episode(
        state, [_swap_into_position_zero, _swap_into_position_zero], max_steps=5000
    )

    assert len(samples) > 0
    assert sorted(samples[-1].y.tolist()) == [0, 1]


# --- generate_episodes_batch ------------------------------------------------


def test_generate_episodes_batch_matches_generate_episode_played_one_at_a_time():
    # Same property as run_mcts_batch's equivalence test, one level up: playing
    # N games concurrently must produce exactly the samples/outcomes that
    # playing each one alone (with the same seed) would - batching only
    # changes when the evaluator is called, never the result.
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=3, seed=2)

    solo_a = generate_episode(state_a, _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(10), max_steps=3000)
    solo_b = generate_episode(state_b, _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(20), max_steps=3000)

    batch_a, batch_b = generate_episodes_batch(
        [state_a, state_b],
        _batch_evaluate,
        num_simulations=2,
        rngs=[np.random.default_rng(10), np.random.default_rng(20)],
        max_steps=3000,
    )

    assert [s.pi.tolist() for s in batch_a] == [s.pi.tolist() for s in solo_a]
    assert [s.y.tolist() for s in batch_a] == [s.y.tolist() for s in solo_a]
    assert [s.pi.tolist() for s in batch_b] == [s.pi.tolist() for s in solo_b]
    assert [s.y.tolist() for s in batch_b] == [s.y.tolist() for s in solo_b]


def test_generate_episodes_batch_handles_games_finishing_at_different_times():
    # player counts differ (2 vs 4), so the games are very unlikely to reach
    # game_over in the same number of decisions - this exercises the "batch
    # shrinks as games finish early" path rather than every game ending on
    # the same round.
    states = [new_match(player_count=2, seed=1), new_match(player_count=4, seed=2)]

    results = generate_episodes_batch(
        states,
        _batch_evaluate,
        num_simulations=2,
        rngs=[np.random.default_rng(0), np.random.default_rng(1)],
        max_steps=3000,
    )

    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0
    assert sorted(results[0][-1].y.tolist()) == [0, 1]
    assert sorted(results[1][-1].y.tolist()) == [0, 1, 2, 3]


def test_generate_episodes_batch_widens_a_tied_representative_to_a_uniformly_sampled_real_position():
    # Same property as generate_episode's own version of this test - the
    # batched path must widen ties in its applied action too, not just when
    # played solo.
    revealed_positions = set()
    for seed in range(40):
        state = new_match(player_count=2, seed=1)
        turn = Turn.from_state(state)
        results = generate_episodes_batch(
            [state],
            _batch_evaluate,
            num_simulations=1,
            rngs=[np.random.default_rng(seed)],
            max_steps=3000,
            round_max_steps=200,
            max_rounds=10,
        )
        after_board = results[0][1].state.boards[turn.acting_player]
        flipped = next(i for i, c in enumerate(after_board.cards) if c is not None and c.face_up)
        revealed_positions.add(flipped)

    assert len(revealed_positions) > 1


def test_generate_episodes_batch_with_no_games_returns_an_empty_list():
    assert generate_episodes_batch([], _batch_evaluate, num_simulations=2) == []


def test_generate_episodes_batch_force_closes_stuck_rounds_per_game_independently():
    # Only the first game is stuck (degenerate evaluator); the second plays
    # normally - both must finish, independently, without either raising.
    def mixed_evaluate(states):
        return [
            _degenerate_place_zero_evaluate(s) if i == 0 else _uniform_evaluate(s) for i, s in enumerate(states)
        ]

    states = [new_match(player_count=2, seed=1), new_match(player_count=2, seed=2)]

    results = generate_episodes_batch(
        states,
        mixed_evaluate,
        num_simulations=2,
        tau_schedule=0.0,
        rngs=[np.random.default_rng(0), np.random.default_rng(1)],
        max_steps=5000,
        round_max_steps=20,
        max_rounds=3,
    )

    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0
    assert sorted(results[0][-1].y.tolist()) == [0, 1]
    assert sorted(results[1][-1].y.tolist()) == [0, 1]


# --- generate_episodes_batch: bad path --------------------------------------


def test_generate_episodes_batch_raises_if_a_game_does_not_finish_within_max_steps():
    states = [new_match(player_count=2, seed=1)]

    with pytest.raises(RuntimeError):
        generate_episodes_batch(states, _batch_evaluate, num_simulations=1, rngs=[np.random.default_rng(0)], max_steps=1)


def test_generate_episodes_batch_rejects_mismatched_rngs_length():
    states = [new_match(player_count=2, seed=1), new_match(player_count=2, seed=2)]

    with pytest.raises(ValueError):
        generate_episodes_batch(states, _batch_evaluate, num_simulations=1, rngs=[np.random.default_rng(0)])


# --- generate_bot_episode --------------------------------------------------


def test_generate_bot_episode_produces_one_hot_pi_matching_the_bots_actual_choices():
    state = new_match(player_count=2, seed=1)
    bots = [HeuristicBot(seed=1), HeuristicBot(seed=2)]

    samples = generate_bot_episode(state, [b.choose_action for b in bots], max_steps=3000)

    assert len(samples) > 0
    for sample in samples:
        assert sample.n_act == 2
        assert sample.pi.shape == (ACTION_SPACE_SIZE,)
        # one-hot: exactly one action carries all the probability mass
        assert sample.pi.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.count_nonzero(sample.pi) == 1
    assert sorted(samples[-1].y.tolist()) == [0, 1]


def test_generate_bot_episode_pi_is_always_a_legal_action_for_its_own_state():
    state = new_match(player_count=3, seed=3)
    bots = [HeuristicBot(seed=1), HeuristicBot(seed=2), HeuristicBot(seed=3)]

    samples = generate_bot_episode(state, [b.choose_action for b in bots], max_steps=3000)

    for sample in samples:
        legal_indices = {action_to_index(a) for a in engine_legal_actions(sample.state)}
        chosen_index = int(np.argmax(sample.pi))
        assert chosen_index in legal_indices


def test_generate_bot_episode_is_deterministic_for_the_same_seeded_bots():
    def play(seed: int) -> list[int]:
        state = new_match(player_count=2, seed=seed)
        bots = [HeuristicBot(seed=1), HeuristicBot(seed=2)]
        samples = generate_bot_episode(state, [b.choose_action for b in bots], max_steps=3000)
        return [int(np.argmax(s.pi)) for s in samples]

    assert play(7) == play(7)


# --- bad path ------------------------------------------------------------


def test_generate_bot_episode_raises_if_the_match_does_not_finish_within_max_steps():
    state = new_match(player_count=2, seed=1)
    bots = [HeuristicBot(seed=1), HeuristicBot(seed=2)]

    with pytest.raises(RuntimeError):
        generate_bot_episode(state, [b.choose_action for b in bots], max_steps=1)
