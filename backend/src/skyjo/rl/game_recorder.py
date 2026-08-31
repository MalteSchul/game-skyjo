"""Records a fully-analyzable Skyjo game: every decision keeps the
ground-truth `GameState`, the raw policy prior (no search), the full MCTS
root visit distribution (not just the greedy pick), and the network's
rank/points estimates - everything needed to reconstruct "what did the net
think, what did search find, what was actually played" after the fact,
without rerunning anything. `game_record_export.game_record_to_dict` turns a
`GameRecord` into the JSON the frontend's game-replay tool loads; see
`scripts/play_and_record_game.py` for the CLI that produces one.

`play_and_record` plays evaluation-style (greedy, no root noise) - the same
regime `evaluator.evaluate_vs_heuristic` uses. `record_training_selfplay_game`
is the training-faithful counterpart: Dirichlet root noise, tau-tempered
sampling, tied-action widening, exactly as `rl.selfplay.generate_episode`
actually generates a training sample - instrumented to keep what that
function normally discards once it has computed `pi`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.domain.action_equivalence import group_representatives, tied_actions
from skyjo.domain.engine import (
    Action,
    GameState,
    apply_action,
    force_close_round,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn
from skyjo.rl.encoding import StateEncoding, encode_state
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.hidden_info import gamestate_from_turn
from skyjo.rl.mcts import (
    MCTSNode,
    advance_cached_root,
    greedy_action,
    run_mcts,
    sample_action,
    visit_distribution,
)
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import final_ranks


@dataclass
class DecisionRecord:
    step: int
    actor_seat: int
    actor_name: str
    phase: str
    total_scores: tuple[int, ...]
    state: GameState  # ground-truth state before this decision (all hidden info visible to us)
    encoding: StateEncoding  # the exact features/mask/perm the network actually saw

    raw_policy_priors: dict[Action, float]  # level 1: network policy head, no search
    raw_prior_favorite: Action
    raw_rank_probs: np.ndarray  # (n_act, n_act) absolute seat order, from the same no-search forward pass
    raw_points_pred: np.ndarray  # (n_act,) absolute seat order

    mcts_num_simulations_requested: int
    mcts_visit_counts: dict[Action, int]  # level 2: full root visit distribution, not just top-3
    mcts_root_value: np.ndarray | None  # network's value estimate AT THE ROOT when it was expanded (n_act,)
    reused_tree_visits: int  # how many visits the reused root already had before this decision's search topped it up

    # Q-value (mean_value()[actor_seat], the acting player's own view) per
    # action - "initial" from partway into THIS decision's own search (an
    # early checkpoint, not just the first simulation - see
    # _EARLY_VALUE_SNAPSHOT_FRACTION), "final" after all requested
    # simulations. Shows how much search revised the network's early value
    # guess for each candidate action, not just which one it ended up
    # preferring (that's mcts_visit_counts). An action absent from the
    # snapshot (not yet explored at that point) is 0, matching
    # MCTSEdge.mean_value()'s own zero-visits convention. Equal to each
    # other when this decision's own search added zero new simulations (a
    # fully-reused root, eval-style play only) - no new information to have
    # revised anything.
    initial_action_values: dict[Action, float]
    final_action_values: dict[Action, float]

    chosen_action: Action  # what was actually played (greedy_action, or a tau-sample for training self-play)
    # Whether search's own most-visited action differs from, and actually
    # out-shares, the raw prior's favorite - see _search_overrode_prior.
    # Deliberately not `chosen_action != raw_prior_favorite`: chosen_action
    # can be a low-probability tau-sample that differs from the favorite by
    # pure chance even when search itself agrees with the prior.
    search_overrode_prior: bool

    # A purely hypothetical query - never fed into apply_action, never affects
    # the game - of what a fixed rule-based reference (no search of its own)
    # would have played from this exact turn. Lets "diffs from heuristic" be
    # computed after the fact without a second, separately-seeded rerun.
    # None for record_training_selfplay_game, which has no such reference.
    heuristic_action: Action | None = None
    # heuristic_action's own equivalence-class representative (see
    # domain.action_equivalence) - what to actually compare against
    # chosen_action/raw_prior_favorite, since those are themselves always
    # representatives (both the network and MCTS only ever see
    # distinct_actions(turn), never the raw board-position-specific action).
    # Comparing heuristic_action directly would flag e.g. every initial_flip
    # decision as "different" merely because HeuristicBot picks a uniformly
    # random real slot while the net's choice always renders as the same
    # representative slot - not a real disagreement, since every slot is
    # provably equivalent before any card is known.
    heuristic_action_representative: Action | None = None

    # Training self-play only (record_training_selfplay_game) - None for
    # eval-style play_and_record games, which use add_root_noise=False and
    # never compute a tau-tempered pi at all.
    #
    # The root prior actually searched by PUCT, AFTER Dirichlet(alpha) noise
    # was mixed in at weight epsilon - raw_policy_priors above is always
    # prior_before_noise (the network's own output; noise is the self-play
    # procedure's exploration mechanism, not something the network produces).
    dirichlet_noised_priors: dict[Action, float] | None = None
    # rl.selfplay.generate_episode's actual training target: the root visit
    # distribution tempered by `tau` - sharper (tau<1) or softer (tau>1) than
    # the raw visit-count ratio.
    pi_target: dict[Action, float] | None = None
    tau: float | None = None
    # How many real board-position actions chosen_action was uniformly
    # sampled from (domain.action_equivalence.tied_actions) - >1 whenever
    # multiple positions were still equivalent at search time (e.g. every
    # initial-flip slot, before any card is known, or two placement slots
    # holding the same already-revealed value).
    tied_group_size: int | None = None


@dataclass
class GameRecord:
    seat_names: tuple[str, str]
    checkpoint_paths: tuple[str, str]
    seed: int
    num_simulations: int
    c_puct: float
    decisions: list[DecisionRecord] = field(default_factory=list)
    final_total_scores: tuple[int, ...] | None = None
    final_ranks: list[int] | None = None
    winner_name: str | None = None
    rounds_played: int = 0


def _search_overrode_prior(
    raw_policy_priors: dict[Action, float],
    mcts_visit_counts: dict[Action, int],
    raw_prior_favorite: Action,
) -> bool:
    """Whether MCTS search itself - not the final sampled/played action -
    genuinely disagreed with the raw prior: the most-visited action differs
    from the prior's own favorite, AND search actually pushed that action's
    share above what the raw prior alone gave it (not just a hair above the
    favorite from noise/rounding).

    Deliberately NOT `chosen_action != raw_prior_favorite`: for training
    self-play, `chosen_action` is tau-sampled from `pi`, so a low-probability
    action can get played by chance even when search's own visit
    distribution fully agrees with the prior - that's sampling variance, not
    search overriding anything. This looks only at the deterministic
    post-search visit distribution.
    """
    if not mcts_visit_counts:
        return False
    top_visit_action = max(mcts_visit_counts, key=mcts_visit_counts.get)
    if top_visit_action == raw_prior_favorite:
        return False
    total_visits = sum(mcts_visit_counts.values()) or 1
    top_visit_share = mcts_visit_counts[top_visit_action] / total_visits
    top_visit_raw_prior_share = raw_policy_priors.get(top_visit_action, 0.0)
    return top_visit_share > top_visit_raw_prior_share


# How far into a decision's own search to snapshot each action's Q-value as
# "initial" - a fraction of that call's own requested simulation count, not
# just the first simulation (which would only have touched a single root
# edge; PUCT hasn't spread visits across the field yet at N=1). 20% is early
# enough to reflect a young, mostly-prior-driven estimate while still having
# touched most candidate actions at typical simulation counts (20-1000).
_EARLY_VALUE_SNAPSHOT_FRACTION = 0.2


def _value_snapshot_hooks(
    num_simulations: int, actor: int
) -> tuple[Callable[[MCTSNode], None], Callable[[int], None], dict[Action, float]]:
    """Returns `(on_root_ready, on_simulation, initial_values)` for `run_mcts`
    - `initial_values` starts empty and gets filled in place, once, at the
    early snapshot step (see `_EARLY_VALUE_SNAPSHOT_FRACTION`). Stays empty
    if `num_simulations` is 0 (a fully-reused root needing no new work) or
    the snapshot step is never reached for some other reason - callers
    should fall back to the final values in that case, since there's no new
    information to distinguish "initial" from "final" either way.
    """
    root_box: list[MCTSNode] = []
    initial_values: dict[Action, float] = {}
    snapshot_step = max(1, round(num_simulations * _EARLY_VALUE_SNAPSHOT_FRACTION))

    def on_root_ready(root: MCTSNode) -> None:
        root_box.append(root)

    def on_simulation(step: int) -> None:
        if step == snapshot_step and not initial_values and root_box:
            initial_values.update({a: float(e.mean_value()[actor]) for a, e in root_box[0].edges.items()})

    return on_root_ready, on_simulation, initial_values


def load_net(path: str, network_kwargs: dict[str, Any] | None = None) -> AlphaZeroNet:
    net = AlphaZeroNet(**(network_kwargs or {}))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return net


def play_and_record(
    name_a: str,
    checkpoint_a: str,
    name_b: str,
    checkpoint_b: str,
    *,
    network_kwargs: dict[str, Any] | None = None,
    seed: int,
    num_simulations: int,
    c_puct: float = 1.5,
    max_steps: int = 5000,
    round_max_steps: int = 200,
    max_rounds: int = 10,
    deterministic_torch: bool = True,
) -> GameRecord:
    """Plays one 2-player game, both seats net-driven MCTS (seat0=name_a/
    checkpoint_a, seat1=name_b/checkpoint_b - pass the same checkpoint twice
    for a true self-play mirror match), greedy/no-root-noise (evaluation-
    style, not training-style tau sampling), recording everything needed to
    fully reconstruct the game afterward.

    `deterministic_torch=True` pins `torch.set_num_threads(1)` for the
    duration of this call and restores the previous value after - CPU
    multi-threaded float reduction order isn't guaranteed bit-stable run to
    run, which would otherwise make two calls with the same seed produce
    different games. Single-threaded forward passes remove that source of
    nondeterminism (at some throughput cost); the MCTS rng itself is already
    explicitly seeded, so this closes the remaining gap.
    """
    prior_num_threads = torch.get_num_threads()
    if deterministic_torch:
        torch.set_num_threads(1)
    try:
        checkpoints = [checkpoint_a, checkpoint_b]
        names = [name_a, name_b]
        nets = [load_net(checkpoints[0], network_kwargs), load_net(checkpoints[1], network_kwargs)]
        rank_sinks: list[dict] = [{}, {}]
        points_sinks: list[dict] = [{}, {}]
        evaluates = [
            make_network_evaluator(nets[i], rank_probs_sink=rank_sinks[i], points_pred_sink=points_sinks[i])
            for i in range(2)
        ]
        # A fixed rule-based reference, queried but never played (see
        # DecisionRecord.heuristic_action) - seeded off the game's own seed so
        # a given (checkpoints, seed) pair always reports the same hypothetical
        # heuristic line, without affecting the game itself at all.
        heuristic_bots = [HeuristicBot(seed=seed * 1000 + 7 + i) for i in range(2)]

        record = GameRecord(
            seat_names=(name_a, name_b),
            checkpoint_paths=(checkpoint_a, checkpoint_b),
            seed=seed,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )

        rng = np.random.default_rng(seed)
        state = new_match(player_count=2, seed=seed)
        round_step = 0
        round_count = 0
        cached_roots: list[MCTSNode | None] = [None, None]

        for _ in range(max_steps):
            if state.phase == "round_over":
                round_count += 1
                if round_count >= max_rounds:
                    break
                state = start_next_round(state)
                round_step = 0
                cached_roots = [None, None]
                continue
            if state.phase == "game_over":
                break
            if round_step >= round_max_steps:
                state = force_close_round(state)
                round_step = 0
                cached_roots = [None, None]
                continue

            turn = Turn.from_state(state)
            actor = turn.acting_player
            evaluate = evaluates[actor]

            # Level 1: raw network output, no search - a direct call on the
            # TRUE state (not a Turn-derived reconstruction), so the sink
            # lookup immediately below is guaranteed to hit under the exact
            # same key we read it back with.
            priors, _value = evaluate(state)
            raw_prior_favorite = max(priors, key=priors.get)
            raw_rank_probs = rank_sinks[actor][state].copy()
            raw_points_pred = points_sinks[actor][state].copy()

            encoding = encode_state(state)

            # Level 2: MCTS search, reusing this seat's own tree across its
            # own turns exactly as evaluator._play_one_eval_game does.
            cached_root = cached_roots[actor]
            if cached_root is not None and cached_root.state != gamestate_from_turn(turn):
                cached_root = None
            already_visited = cached_root.visit_count if cached_root is not None else 0
            sims_this_call = max(0, num_simulations - already_visited)
            on_root_ready, on_simulation, initial_action_values = _value_snapshot_hooks(sims_this_call, actor)
            root = run_mcts(
                turn,
                evaluate,
                num_simulations=sims_this_call,
                c_puct=c_puct,
                add_root_noise=False,
                rng=rng,
                reuse_root=cached_root,
                on_root_ready=on_root_ready,
                on_simulation=on_simulation,
            )
            chosen_action = greedy_action(root, rng)
            cached_roots[actor] = root

            mcts_visit_counts = {a: e.visit_count for a, e in root.edges.items()}
            final_action_values = {a: float(e.mean_value()[actor]) for a, e in root.edges.items()}
            if not initial_action_values:
                initial_action_values = dict(final_action_values)
            heuristic_action = heuristic_bots[actor].choose_action(turn)
            heuristic_action_representative = group_representatives(turn)[heuristic_action]

            record.decisions.append(
                DecisionRecord(
                    step=len(record.decisions),
                    actor_seat=actor,
                    actor_name=names[actor],
                    phase=state.phase,
                    total_scores=state.total_scores,
                    state=state,
                    encoding=encoding,
                    raw_policy_priors=dict(priors),
                    raw_prior_favorite=raw_prior_favorite,
                    raw_rank_probs=raw_rank_probs,
                    raw_points_pred=raw_points_pred,
                    mcts_num_simulations_requested=num_simulations,
                    mcts_visit_counts=mcts_visit_counts,
                    mcts_root_value=None if root.value is None else root.value.copy(),
                    reused_tree_visits=already_visited,
                    initial_action_values=initial_action_values,
                    final_action_values=final_action_values,
                    chosen_action=chosen_action,
                    search_overrode_prior=_search_overrode_prior(dict(priors), mcts_visit_counts, raw_prior_favorite),
                    heuristic_action=heuristic_action,
                    heuristic_action_representative=heuristic_action_representative,
                )
            )

            state = apply_action(state, chosen_action)
            round_step += 1
            for i in range(2):
                if state.phase in ("round_over", "game_over"):
                    cached_roots[i] = None
                else:
                    cached_roots[i] = advance_cached_root(cached_roots[i], turn, chosen_action, Turn.from_state(state))
        else:
            raise RuntimeError(f"play_and_record: seed={seed} did not finish within {max_steps} steps")

        record.final_total_scores = state.total_scores
        record.final_ranks = final_ranks(state.total_scores)
        record.winner_name = names[record.final_ranks.index(0)]
        record.rounds_played = round_count + (1 if state.phase == "game_over" else 0)
        return record
    finally:
        torch.set_num_threads(prior_num_threads)


def record_training_selfplay_game(
    checkpoint_path: str,
    *,
    seed: int,
    network_kwargs: dict[str, Any] | None = None,
    tau: float | None = None,
    deterministic_torch: bool = True,
) -> GameRecord:
    """Plays one self-play game with the exact mechanics
    `rl.selfplay.generate_episode` uses to generate real training data -
    Dirichlet root noise, tau-tempered sampling, tied-action widening - all
    read straight from `checkpoint_path`'s own saved `extra.config` (see
    `rl.loop.run_training_loop`'s checkpoint writes) rather than requiring
    the caller to retype a config that must already match. Both seats are
    the SAME network (a true self-play mirror, not a contest between two
    checkpoints) - `seat_names` reflects that as "<checkpoint file> seat0"/
    "seat1".

    `tau`, if given, overrides the checkpoint's own saved tau - e.g. to
    compare a run's actual (often sharp, near-greedy) tau against tau=1.0
    (pi exactly proportional to raw visit counts, no sharpening at all) and
    see whether that alone changes which lines self-play explores. Every
    other config value still comes from the checkpoint.

    Unlike `play_and_record`, there is no opponent seat to alternate turns
    with and no tree reuse across turns - `generate_episode` itself never
    reuses a root either, since Dirichlet noise makes every decision's root
    a fresh, independently-noised draw.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["extra"]["config"]
    net_kwargs = network_kwargs if network_kwargs is not None else cfg["network_kwargs"]
    tau_used = tau if tau is not None else cfg["tau"]

    prior_num_threads = torch.get_num_threads()
    if deterministic_torch:
        torch.set_num_threads(1)
    try:
        net = load_net(checkpoint_path, net_kwargs)
        rank_sink: dict[GameState, np.ndarray] = {}
        points_sink: dict[GameState, np.ndarray] = {}
        evaluate = make_network_evaluator(net, rank_probs_sink=rank_sink, points_pred_sink=points_sink)

        checkpoint_label = Path(checkpoint_path).name
        seat_names = (f"{checkpoint_label} seat0", f"{checkpoint_label} seat1")
        record = GameRecord(
            seat_names=seat_names,
            checkpoint_paths=(checkpoint_path, checkpoint_path),
            seed=seed,
            num_simulations=cfg["num_simulations"],
            c_puct=cfg["c_puct"],
        )

        rng = np.random.default_rng(seed)
        state = new_match(player_count=2, seed=seed)
        round_step = 0
        round_count = 0

        for _ in range(cfg["max_steps_per_episode"]):
            if state.phase == "round_over":
                round_count += 1
                if round_count >= cfg["max_rounds"]:
                    break
                state = start_next_round(state)
                round_step = 0
                continue
            if state.phase == "game_over":
                break
            if round_step >= cfg["round_max_steps"]:
                state = force_close_round(state)
                round_step = 0
                continue

            turn = Turn.from_state(state)
            actor = turn.acting_player

            # Direct call on the TRUE state, purely so the sink lookup below
            # hits under the exact key read back - see play_and_record's
            # identical comment. encode_state never encodes true hidden
            # values, so this doesn't give the read a different answer than
            # run_mcts's own root expansion gets below; it's a read, not a
            # second opinion.
            evaluate(state)
            raw_rank_probs = rank_sink[state].copy()
            raw_points_pred = points_sink[state].copy()

            encoding = encode_state(state)

            on_root_ready, on_simulation, initial_action_values = _value_snapshot_hooks(cfg["num_simulations"], actor)
            root = run_mcts(
                turn,
                evaluate,
                num_simulations=cfg["num_simulations"],
                c_puct=cfg["c_puct"],
                dirichlet_alpha=cfg["dirichlet_alpha"],
                dirichlet_epsilon=cfg["dirichlet_epsilon"],
                add_root_noise=True,
                rng=rng,
                on_root_ready=on_root_ready,
                on_simulation=on_simulation,
            )

            raw_policy_priors = {a: e.prior_before_noise for a, e in root.edges.items()}
            dirichlet_noised_priors = {a: e.prior for a, e in root.edges.items()}
            raw_prior_favorite = max(raw_policy_priors, key=raw_policy_priors.get)
            mcts_visit_counts = {a: e.visit_count for a, e in root.edges.items()}
            final_action_values = {a: float(e.mean_value()[actor]) for a, e in root.edges.items()}
            if not initial_action_values:
                initial_action_values = dict(final_action_values)

            pi = visit_distribution(root, tau=tau_used)
            representative = sample_action(pi, rng)
            group = tied_actions(turn, representative)
            chosen_action = group[rng.integers(len(group))] if len(group) > 1 else representative

            record.decisions.append(
                DecisionRecord(
                    step=len(record.decisions),
                    actor_seat=actor,
                    actor_name=seat_names[actor],
                    phase=state.phase,
                    total_scores=state.total_scores,
                    state=state,
                    encoding=encoding,
                    raw_policy_priors=raw_policy_priors,
                    raw_prior_favorite=raw_prior_favorite,
                    raw_rank_probs=raw_rank_probs,
                    raw_points_pred=raw_points_pred,
                    mcts_num_simulations_requested=cfg["num_simulations"],
                    mcts_visit_counts=mcts_visit_counts,
                    mcts_root_value=None if root.value is None else root.value.copy(),
                    reused_tree_visits=0,
                    initial_action_values=initial_action_values,
                    final_action_values=final_action_values,
                    chosen_action=chosen_action,
                    search_overrode_prior=_search_overrode_prior(raw_policy_priors, mcts_visit_counts, raw_prior_favorite),
                    dirichlet_noised_priors=dirichlet_noised_priors,
                    pi_target=dict(pi),
                    tau=tau_used,
                    tied_group_size=len(group),
                )
            )

            state = apply_action(state, chosen_action)
            round_step += 1
        else:
            raise RuntimeError(
                f"record_training_selfplay_game: seed={seed} did not finish within "
                f"{cfg['max_steps_per_episode']} steps"
            )

        record.final_total_scores = state.total_scores
        record.final_ranks = final_ranks(state.total_scores)
        record.winner_name = seat_names[record.final_ranks.index(0)]
        record.rounds_played = round_count + (1 if state.phase == "game_over" else 0)
        return record
    finally:
        torch.set_num_threads(prior_num_threads)
