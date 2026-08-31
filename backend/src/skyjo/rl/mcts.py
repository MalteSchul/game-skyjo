"""Vector-valued MCTS: PUCT selection keyed to the active player's own
utility component, multi-agent backpropagation, Dirichlet root noise.

Rooted at a `Turn`, not a `GameState` - the search only ever sees what the
`Turn`/`Observation` layer already treats as public (see `domain.observation`
and `rl.hidden_info`). Every `GameState` this module holds internally is a
redacted one (`hidden_info.gamestate_from_turn`/`rescrub`): whichever hidden
card the *true* game holds at any position, this module never reads it -
stock draws, initial flips, and discard-reveals are resolved as genuine
chance events, sampled from the pool of cards that are still unknown given
only what's public, never by peeking at the answer.

Two kinds of node:
- `MCTSNode`: a decision point, exactly as before - PUCT over `MCTSEdge`s
  keyed by `Action`.
- `ChanceNode`: sits between a reveal-triggering edge and the decision node
  it leads to, with `ChanceEdge`s keyed by the revealed *value* instead of
  an action, weighted by how many of that value are still unknown. Visits
  resample every time (`sample_reveal`), so repeat visits to the same value
  land on the same cached child (ordinary MCTS structure-sharing) while
  different sampled values grow sibling branches - the tree's own Q-average
  at the parent edge becomes a genuine Monte Carlo estimate over the hidden
  distribution, not one frozen guess at the future.

One case doesn't get a `ChanceNode`: the action that closes a round forces
*every* remaining hidden card on *every* board face-up at once (Skyjo scores
a round by revealing everyone's hand). Enumerating that jointly is
intractable and, since a fresh round is dealt from an entirely independent
shuffle right after, there's nothing meaningfully cacheable about a specific
resolution anyway - so it's resampled and evaluated fresh on every visit,
never cached on the edge. That's still an average over many independently
sampled continuations across the search, just without a persistent subtree
below it (see `_is_round_closing`/`_advance_round_closing`).
"""

from __future__ import annotations

import inspect
import math
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from skyjo.domain.action_equivalence import group_representatives
from skyjo.domain.engine import Action, ActionType, GameState, apply_action, start_next_round
from skyjo.domain.observation import Turn
from skyjo.rl.hidden_info import (
    gamestate_from_turn,
    is_reveal,
    rescrub,
    resolve_drawn_stock_card,
    resolve_reveal,
    resolve_round_close,
    sample_reveal,
    unknown_card_counts,
    will_close_round,
)

# priors: prob per legal (post-collapsing) action, normalized over legal actions only.
# value: utility vector of length n_act, aligned to player index.
LeafResult = tuple[dict[Action, float], np.ndarray]
# Minimum contract: one GameState in, one LeafResult out. An implementation
# may *additionally* accept a keyword-only `turn: Turn | None = None` (see
# `make_network_evaluator`) - this module already has that leaf's `Turn` on
# hand (it needs one anyway, for `_expand`'s equivalence-collapsing) and, when
# the evaluator accepts it, passes it along so `encode_state` doesn't have to
# re-derive `legal_actions(state)` from scratch to build its mask. Detected
# once per search via `_accepts_keyword`, not assumed - a plain single-arg
# evaluator (a test stub, a hand-rolled one) keeps working exactly as before.
EvaluateFn = Callable[[GameState], LeafResult]
# Same contract as EvaluateFn, but over many states at once (batch order in,
# same order out) - one evaluator call instead of one per state. See
# `run_mcts_batch`. May likewise accept an optional keyword-only `turns`.
BatchEvaluateFn = Callable[[Sequence[GameState]], "list[LeafResult]"]


def _accepts_keyword(fn: Callable, name: str) -> bool:
    """Whether `fn(..., name=...)` is safe to call: `fn` declares that
    keyword-only param explicitly, or accepts arbitrary `**kwargs`. Checked
    once per `run_mcts`/`run_mcts_batch` call (not per leaf/simulation), so
    the cost of introspecting `fn`'s signature is negligible next to the
    hundreds of simulations it gates."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

DEFAULT_C_PUCT = 1.5
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_DIRICHLET_EPSILON = 0.25


@dataclass
class MCTSEdge:
    action: Action
    prior: float
    n_act: int
    # `prior` as `_expand` first set it, before `_apply_root_noise` (root
    # edges only) mixes in Dirichlet noise and overwrites `prior` in place -
    # kept around purely so the raw evaluator output stays inspectable
    # (e.g. via `tree_export`) after the mix. Equal to `prior` on every
    # non-root edge, since noise is only ever applied at the root. Defaults
    # to whatever `prior` was constructed with, for callers (tests, mostly)
    # that never touch noise at all.
    prior_before_noise: float | None = None
    visit_count: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    child: MCTSNode | ChanceNode | None = None

    def __post_init__(self) -> None:
        if self.value_sum.shape != (self.n_act,):
            self.value_sum = np.zeros(self.n_act)
        if self.prior_before_noise is None:
            self.prior_before_noise = self.prior

    def mean_value(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.n_act)
        return self.value_sum / self.visit_count


@dataclass
class ChanceEdge:
    value: int
    prior: float
    n_act: int
    visit_count: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    child: MCTSNode | None = None

    def __post_init__(self) -> None:
        if self.value_sum.shape != (self.n_act,):
            self.value_sum = np.zeros(self.n_act)


@dataclass
class ChanceNode:
    n_act: int
    counts: Counter[int]
    edges: dict[int, ChanceEdge] = field(default_factory=dict)


@dataclass
class MCTSNode:
    state: GameState
    n_act: int
    is_terminal: bool
    expanded: bool = False
    edges: dict[Action, MCTSEdge] = field(default_factory=dict)
    # The utility vector this node's own state was assigned when it was
    # expanded: `evaluate(state)`'s value for a decision node, or
    # `_terminal_utility` for a terminal one. Purely a record of that one
    # evaluation - never read back during search (backprop already folded
    # it into the *parent* edge's `value_sum` at expansion time, see
    # `_expand_child`) - kept only so a caller (e.g. `tree_export`) can see
    # what the network itself predicted for a position, not just the
    # Monte-Carlo average visits produced afterwards. None until expanded.
    value: np.ndarray | None = None

    @property
    def visit_count(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())


def _is_terminal(state: GameState) -> bool:
    return state.phase == "game_over"


def _terminal_utility(total_scores: Sequence[int]) -> np.ndarray:
    """Utility vector from the true final standings (lower score = better rank).

    Ties share the average of their rank positions (standard "fractional"
    competition ranking), so the same linear payoff mapping used by the
    network's rank head (`w[r] = 1 - 2r/(n-1)`) stays well-defined for tied
    finishes rather than needing a separate tie-break rule.
    """
    n = len(total_scores)
    order = sorted(range(n), key=lambda i: total_scores[i])
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and total_scores[order[j + 1]] == total_scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    denom = max(n - 1, 1)
    return 1.0 - 2.0 * ranks / denom


def _expand(node: MCTSNode, turn: Turn, priors: dict[Action, float]) -> None:
    """Builds `node.edges` from the network's priors, collapsing provably-
    equivalent actions (see `domain.action_equivalence`) onto one shared
    representative edge first, so the search doesn't spend visits telling
    apart options that can't differ given what's currently known.
    """
    representative_of = group_representatives(turn)
    collapsed: dict[Action, float] = {}
    for action, prior in priors.items():
        representative = representative_of.get(action, action)
        collapsed[representative] = collapsed.get(representative, 0.0) + prior

    node.edges = {
        action: MCTSEdge(action=action, prior=prior, n_act=node.n_act)
        for action, prior in collapsed.items()
    }
    node.expanded = True


def _apply_root_noise(
    node: MCTSNode, alpha: float, epsilon: float, rng: np.random.Generator
) -> None:
    actions = list(node.edges.keys())
    if not actions:
        return
    noise = rng.dirichlet([alpha] * len(actions))
    for action, eta in zip(actions, noise, strict=True):
        edge = node.edges[action]
        edge.prior = (1 - epsilon) * edge.prior + epsilon * eta


def _best_by_puct(edges: Iterable[MCTSEdge], k: int, sqrt_total: float, c_puct: float) -> MCTSEdge:
    """Core PUCT scoring loop, factored out of `_select_edge` so
    `_select_root_edge_capped` can reuse the identical `q + u` formula over a
    *filtered* subset of a node's edges instead of duplicating it.
    """
    best_edge: MCTSEdge | None = None
    best_score = -math.inf
    for edge in edges:
        q = edge.mean_value()[k]
        u = c_puct * edge.prior * sqrt_total / (1 + edge.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_edge = edge

    assert best_edge is not None, "_best_by_puct: edges is empty - should never be called with none"
    return best_edge


def _select_edge(node: MCTSNode, c_puct: float) -> MCTSEdge:
    return _best_by_puct(
        node.edges.values(), node.state.current_player, math.sqrt(node.visit_count), c_puct
    )


def _select_root_edge_capped(node: MCTSNode, c_puct: float) -> MCTSEdge:
    """Root-only alternative to `_select_edge`, for live/eval decision-making
    rather than self-play: caps how far the most-visited root edge can pull
    ahead of the second-most-visited one. Once a single edge is a clear
    leader (2+ visits ahead), a further visit into it can't change the
    eventual argmax-visits decision - it's spent instead checking whether the
    runner-up could still catch up. Below the root, `_select_edge` stays
    unchanged: only the root's final choice is a best-arm problem, deeper
    nodes still need their own Q built the normal way.

    A tie for the lead (two or more edges sharing the max visit count) never
    triggers exclusion - only a single undisputed leader does, so an
    undecided race is left entirely to ordinary PUCT.
    """
    edges = list(node.edges.values())
    if len(edges) == 1:
        return edges[0]

    k = node.state.current_player
    sqrt_total = math.sqrt(node.visit_count)
    visit_counts = sorted((edge.visit_count for edge in edges), reverse=True)
    max_visits, second_max_visits = visit_counts[0], visit_counts[1]

    if max_visits - 1 > second_max_visits:
        leader = next(edge for edge in edges if edge.visit_count == max_visits)
        edges = [edge for edge in edges if edge is not leader]

    return _best_by_puct(edges, k, sqrt_total, c_puct)


def _leaf_node(
    state: GameState, n_act: int
) -> Generator[tuple[GameState, Turn], LeafResult, tuple[MCTSNode, np.ndarray]]:
    """Builds the `MCTSNode` for a freshly-reached leaf `state`.

    A terminal state resolves immediately from the true final standings - no
    evaluation needed, so this never yields for one. Otherwise it computes
    this leaf's `Turn` once - needed regardless, for `_expand`'s equivalence
    collapsing - and yields `(state, turn)` exactly once, expecting the
    evaluator's `(priors, value)` result sent back (via `.send`) to finish
    building the node. That single `yield` is the seam that lets a caller
    batch this evaluation together with other trees' pending leaves
    (`run_mcts_batch`) instead of every leaf calling an evaluator on its own -
    see `_drive_gen` for the unbatched side of that same seam. Handing the
    same `turn` object to the driver too (rather than it recomputing one) is
    what lets an evaluator that accepts `turn`/`turns` skip re-deriving
    `legal_actions(state)` a second time for this same leaf.
    """
    if _is_terminal(state):
        child = MCTSNode(state=state, n_act=n_act, is_terminal=True)
        child.value = _terminal_utility(state.total_scores)
        return child, child.value

    turn = Turn.from_state(state)
    priors, value = yield state, turn
    child = MCTSNode(state=state, n_act=n_act, is_terminal=False)
    _expand(child, turn, priors)
    child.value = value
    return child, value


def _drive_gen(
    gen: Generator[tuple[GameState, Turn], LeafResult, None], evaluate: EvaluateFn, *, supports_turn: bool
) -> None:
    """Eagerly resolves a `_simulate_once_gen` generator with a synchronous,
    one-state-at-a-time `evaluate` call - this is what makes unbatched search
    (`_simulate_once`) and batched search (`run_mcts_batch`) run the exact
    same tree-walk code, just driven differently: one leaf at a time here,
    many trees' leaves together there. `supports_turn` - from `_accepts_keyword`,
    checked once per search rather than once per leaf - decides whether the
    leaf's already-computed `Turn` is worth passing to `evaluate` at all.
    """
    try:
        leaf_state, turn = next(gen)
    except StopIteration:
        return  # resolved without an evaluation (a cached-terminal leaf)
    result = evaluate(leaf_state, turn=turn) if supports_turn else evaluate(leaf_state)
    try:
        gen.send(result)
    except StopIteration:
        return
    raise AssertionError("_drive_gen: generator yielded more than once")


def _is_round_closing(state: GameState, action: Action) -> bool:
    return will_close_round(state, action)


# All three `_advance_*` helpers below pass `validate=False` to `apply_action`:
# every `action` they're given is `edge.action` from `_select_edge`, i.e. it
# came out of this exact `state`'s own `node.edges` (built from
# `Turn.from_state(state).legal_actions` in `_expand`) - re-checking
# `action in legal_actions(state)` here would just recompute a list this
# module already knows `action` is a member of, on every single simulated
# transition (profiling: legal_actions was ~3x more often than leaves
# expanded before this).
def _advance_deterministic(state: GameState, action: Action) -> GameState:
    """Neither a reveal nor round-closing: nothing hidden is involved."""
    return rescrub(apply_action(state, action, validate=False))


def _advance_resolved_reveal(state: GameState, action: Action, value: int) -> GameState:
    """A single-card reveal, already resolved to `value` by a ChanceEdge."""
    if action.type is ActionType.DRAW_STOCK:
        return rescrub(resolve_drawn_stock_card(apply_action(state, action, validate=False), value))
    primed = resolve_reveal(state, action, value)
    return rescrub(apply_action(primed, action, validate=False))


def _advance_round_closing(state: GameState, action: Action, rng: np.random.Generator) -> GameState:
    """The last-awaiting player's closing action: resolves this action's own
    reveal (if any) plus every other player's remaining hidden cards at
    once, since round-close scoring reveals everyone's hand simultaneously.
    Folds any resulting `round_over` into the freshly-dealt next round, same
    as the deterministic path used to do for every transition.
    """
    patched = state
    already_resolved: tuple[int, int] | None = None
    if is_reveal(state, action):
        value = sample_reveal(unknown_card_counts(Turn.from_state(state)), rng)
        patched = resolve_reveal(patched, action, value)
        already_resolved = (state.current_player, action.position)

    patched = resolve_round_close(patched, rng, already_resolved=already_resolved)
    next_state = rescrub(apply_action(patched, action, validate=False))
    while next_state.phase == "round_over":
        next_state = rescrub(start_next_round(next_state))
    return next_state


def _simulate_once_gen(
    root: MCTSNode, c_puct: float, rng: np.random.Generator, *, cap_root_lead: bool = False
) -> Generator[GameState, LeafResult, None]:
    """Generator form of one simulation: walks the tree exactly as before,
    pausing at `yield` only where the walk reaches a leaf that genuinely
    needs an evaluation (`_leaf_node` resolves a cached-terminal leaf without
    yielding at all). `_simulate_once` (via `_drive_gen`) and `run_mcts_batch`
    both drive this same walk - one leaf at a time, or many trees' leaves
    batched together - so the tree-walk logic lives in exactly one place
    regardless of which driver is used.

    `cap_root_lead`, if set, swaps in `_select_root_edge_capped` for the
    root's own edge choice only (see that function) - every other node in
    the walk still selects via plain `_select_edge`.
    """
    path: list[MCTSEdge | ChanceEdge] = []
    node = root

    while True:
        if node is root and cap_root_lead:
            edge = _select_root_edge_capped(node, c_puct)
        else:
            edge = _select_edge(node, c_puct)
        path.append(edge)
        state = node.state

        if _is_round_closing(state, edge.action):
            # Never cached on edge.child: every visit resamples the reveal
            # and the round-close together, so this edge's own Q is an
            # average over many independently sampled continuations rather
            # than one frozen future (see module docstring).
            child_state = _advance_round_closing(state, edge.action, rng)
            _, value = yield from _leaf_node(child_state, node.n_act)
            break

        if is_reveal(state, edge.action):
            chance = edge.child
            if not isinstance(chance, ChanceNode):
                chance = ChanceNode(
                    n_act=node.n_act, counts=unknown_card_counts(Turn.from_state(state))
                )
                edge.child = chance

            sampled_value = sample_reveal(chance.counts, rng)
            chance_edge = chance.edges.get(sampled_value)
            if chance_edge is None:
                total = sum(chance.counts.values())
                chance_edge = ChanceEdge(
                    value=sampled_value,
                    prior=chance.counts[sampled_value] / total,
                    n_act=node.n_act,
                )
                chance.edges[sampled_value] = chance_edge
            path.append(chance_edge)

            if chance_edge.child is None:
                child_state = _advance_resolved_reveal(state, edge.action, sampled_value)
                child, value = yield from _leaf_node(child_state, node.n_act)
                chance_edge.child = child
                break
            if chance_edge.child.is_terminal:
                value = _terminal_utility(chance_edge.child.state.total_scores)
                break
            node = chance_edge.child
            continue

        # Deterministic: neither a reveal nor round-closing.
        if not isinstance(edge.child, MCTSNode):
            child_state = _advance_deterministic(state, edge.action)
            child, value = yield from _leaf_node(child_state, node.n_act)
            edge.child = child
            break
        if edge.child.is_terminal:
            value = _terminal_utility(edge.child.state.total_scores)
            break
        node = edge.child

    for visited in path:
        visited.visit_count += 1
        visited.value_sum += value


def _simulate_once(
    root: MCTSNode,
    evaluate: EvaluateFn,
    c_puct: float,
    rng: np.random.Generator,
    *,
    supports_turn: bool,
    cap_root_lead: bool = False,
) -> None:
    _drive_gen(
        _simulate_once_gen(root, c_puct, rng, cap_root_lead=cap_root_lead),
        evaluate,
        supports_turn=supports_turn,
    )


def run_mcts(
    root_turn: Turn,
    evaluate: EvaluateFn,
    *,
    num_simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
    add_root_noise: bool = True,
    rng: np.random.Generator | None = None,
    on_simulation: Callable[[int], None] | None = None,
    on_root_ready: Callable[[MCTSNode], None] | None = None,
    reuse_root: MCTSNode | None = None,
    cap_root_lead: bool = False,
) -> MCTSNode:
    """`on_root_ready`, if given, fires exactly once - after the root is
    expanded and root noise applied, before the first simulation - with the
    same `MCTSNode` object this call will keep mutating and eventually
    return. It exists so a caller can snapshot the tree mid-search (e.g. via
    `on_simulation`) without waiting for `run_mcts` to return: the object it
    receives *is* the live root, not a copy.

    `cap_root_lead`, default off, swaps the root's own edge selection to
    `_select_root_edge_capped` (see that function) for live/eval
    decision-making instead of plain PUCT - intended for callers that read
    the final answer off `greedy_action` rather than self-play's tau-sampled
    training targets. Every node below the root is unaffected either way.

    `reuse_root`, if given, must already be an expanded `MCTSNode` whose own
    `.state` is exactly `gamestate_from_turn(root_turn)` - e.g. a child pulled
    out of a previous `run_mcts` call's tree once the game has genuinely
    advanced to that position (see `bots.mcts_bot.MctsBot`, the only current
    caller). Search resumes from its existing edges/visit-counts instead of
    expanding a fresh root, and `num_simulations` more are layered on top of
    whatever it already has - the standard "carry the subtree forward" reuse
    an AlphaZero-style search can do once the real game reaches a position it
    already explored. Raises ValueError if the state doesn't match: this is a
    last-resort consistency guard, not the primary safety mechanism - a
    caller is expected to have already verified the position genuinely
    advanced from `reuse_root`'s own state (`MctsBot` does, via its own
    tree-walk) before ever passing one in.
    """
    if num_simulations < 0:
        raise ValueError("run_mcts: num_simulations must be >= 0")
    rng = rng if rng is not None else np.random.default_rng()

    n_act = len(root_turn.boards)
    root_state = gamestate_from_turn(root_turn)
    supports_turn = _accepts_keyword(evaluate, "turn")

    if reuse_root is not None:
        if reuse_root.state != root_state:
            raise ValueError("run_mcts: reuse_root.state does not match gamestate_from_turn(root_turn)")
        root = reuse_root
        if add_root_noise:
            _apply_root_noise(root, dirichlet_alpha, dirichlet_epsilon, rng)
    else:
        root = MCTSNode(state=root_state, n_act=n_act, is_terminal=False)
        priors, value = evaluate(root_state, turn=root_turn) if supports_turn else evaluate(root_state)
        _expand(root, root_turn, priors)
        root.value = value
        if add_root_noise:
            _apply_root_noise(root, dirichlet_alpha, dirichlet_epsilon, rng)

    if on_root_ready is not None:
        on_root_ready(root)

    for i in range(num_simulations):
        _simulate_once(
            root, evaluate, c_puct, rng, supports_turn=supports_turn, cap_root_lead=cap_root_lead
        )
        if on_simulation is not None:
            on_simulation(i + 1)

    return root


def run_mcts_batch(
    root_turns: Sequence[Turn],
    evaluate_batch: BatchEvaluateFn,
    *,
    num_simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
    add_root_noise: bool = True,
    rngs: Sequence[np.random.Generator] | None = None,
    cap_root_lead: bool = False,
) -> list[MCTSNode]:
    """Runs `len(root_turns)` independent searches side by side - one root
    per turn - batching every round's leaf evaluations into a single
    `evaluate_batch` call instead of evaluating each tree's leaf on its own.

    The trees never interact: this produces exactly what running `run_mcts`
    once per turn would (same tree-walk code, see `_simulate_once_gen`), just
    with every round's `len(root_turns)` pending leaves - one per still-active
    tree - handed to the evaluator together. It exists purely for network-call
    throughput (batch-of-N is far cheaper per-state than N calls of batch-1),
    not search quality or depth.

    Every root advances by exactly `num_simulations` simulations. A root that
    resolves a simulation without needing an evaluation (a cached-terminal
    leaf, same as the unbatched path) simply contributes nothing to that
    round's batch, which can shrink accordingly.
    """
    if num_simulations < 0:
        raise ValueError("run_mcts_batch: num_simulations must be >= 0")
    if not root_turns:
        return []
    rngs = list(rngs) if rngs is not None else [np.random.default_rng() for _ in root_turns]
    if len(rngs) != len(root_turns):
        raise ValueError("run_mcts_batch: rngs must be the same length as root_turns")

    roots = [
        MCTSNode(state=gamestate_from_turn(turn), n_act=len(turn.boards), is_terminal=False)
        for turn in root_turns
    ]
    supports_turns = _accepts_keyword(evaluate_batch, "turns")

    root_states = [root.state for root in roots]
    root_results = (
        evaluate_batch(root_states, turns=root_turns) if supports_turns else evaluate_batch(root_states)
    )
    for root, turn, (priors, value), rng in zip(roots, root_turns, root_results, rngs, strict=True):
        _expand(root, turn, priors)
        root.value = value
        if add_root_noise:
            _apply_root_noise(root, dirichlet_alpha, dirichlet_epsilon, rng)

    for _ in range(num_simulations):
        gens = [
            _simulate_once_gen(root, c_puct, rng, cap_root_lead=cap_root_lead)
            for root, rng in zip(roots, rngs, strict=True)
        ]
        pending_indices: list[int] = []
        pending_states: list[GameState] = []
        pending_turns: list[Turn] = []
        for i, gen in enumerate(gens):
            try:
                state, turn = next(gen)
                pending_states.append(state)
                pending_turns.append(turn)
                pending_indices.append(i)
            except StopIteration:
                pass  # resolved without an evaluation (a cached-terminal leaf)

        if not pending_states:
            continue
        results = (
            evaluate_batch(pending_states, turns=pending_turns)
            if supports_turns
            else evaluate_batch(pending_states)
        )
        for i, result in zip(pending_indices, results, strict=True):
            try:
                gens[i].send(result)
            except StopIteration:
                continue
            raise AssertionError("run_mcts_batch: a simulation yielded more than once")

    return roots


def visit_distribution(root: MCTSNode, tau: float = 1.0) -> dict[Action, float]:
    if not root.edges:
        raise ValueError("visit_distribution: root has no legal actions (terminal or unexpanded)")

    visits = {action: edge.visit_count for action, edge in root.edges.items()}

    if tau <= 1e-3:
        best_action = max(visits, key=lambda a: visits[a])
        return {action: (1.0 if action == best_action else 0.0) for action in visits}

    powered = {action: count ** (1.0 / tau) for action, count in visits.items()}
    total = sum(powered.values())
    if total == 0:
        return {action: 1.0 / len(powered) for action in powered}
    return {action: value / total for action, value in powered.items()}


def sample_action(pi: dict[Action, float], rng: np.random.Generator) -> Action:
    actions = list(pi.keys())
    probs = np.asarray([pi[action] for action in actions], dtype=np.float64)
    probs = probs / probs.sum()
    index = rng.choice(len(actions), p=probs)
    return actions[index]


def greedy_action(
    root: MCTSNode, rng: np.random.Generator, *, tie_break: Literal["random", "value"] = "random"
) -> Action:
    """Most-visited root action - the search's actual best line, for a live
    bot/eval opponent rather than self-play's tau-sampled training targets.

    Ties (common against an uninformative/untrained evaluator, e.g. every
    edge visited once) are broken randomly by default rather than always
    taking the same "first" edge: a deterministic tie-break can lock two such
    players into repeating the same tied pair of actions forever (e.g.
    draw-then-discard, never placing) since nothing about the state that
    matters to the tie ever changes turn to turn.

    `tie_break="value"` instead breaks ties by highest `mean_value()` for the
    acting player - the hedge for `run_mcts(..., cap_root_lead=True)`, whose
    whole point is deliberately compressing visit-count gaps at the root, so
    ties there are more likely and, unlike the uninformative-evaluator case
    above, an actual value estimate is worth trusting over a coin flip.
    """
    if not root.edges:
        raise ValueError("greedy_action: root has no legal actions (terminal or unexpanded)")
    visits = {action: edge.visit_count for action, edge in root.edges.items()}
    max_visits = max(visits.values())
    best_actions = [action for action, count in visits.items() if count == max_visits]
    if len(best_actions) == 1:
        return best_actions[0]
    if tie_break == "value":
        k = root.state.current_player
        return max(best_actions, key=lambda action: root.edges[action].mean_value()[k])
    return best_actions[rng.integers(len(best_actions))]


def infer_revealed_value(turn_before: Turn, action: Action, turn_after: Turn) -> int | None:
    """Best-effort recovery of the real value `action` revealed, purely from
    public `Turn` fields `turn_after` already exposes - never anything a
    cache-reusing caller wasn't already entitled to see. Returns None
    whenever it can't be recovered with confidence (e.g. an immediate
    column-clear already wiped the very position that would have shown it)
    - a safe answer `advance_cached_root` treats exactly like any other
    cache miss.
    """
    if action.type is ActionType.DRAW_STOCK:
        # Still the same acting player's turn (awaiting_placement next), and
        # a drawn card sits in `drawn_card` until it's placed or discarded.
        return turn_after.drawn_card
    if action.position is None:
        return None
    card = turn_after.boards[turn_before.acting_player].cards[action.position]
    return card.value if card is not None and card.face_up else None


def advance_cached_root(
    cached_root: MCTSNode | None, turn_before: Turn, action: Action, turn_after: Turn
) -> MCTSNode | None:
    """Advances a previously-searched tree by one real transition - `action`,
    taken by whichever seat `turn_before.acting_player` was - to whatever
    subtree already covers the resulting position, or None if nothing
    safely does.

    Shared by `bots.mcts_bot.MctsBot.observe_transition` (a live bot
    tracking its own cache across a real match) and
    `evaluator._play_one_eval_game` (the same idea for a `HeuristicBot`
    opponent's moves during evaluation) - both need the identical safe
    tree-walk, just from different callers, neither of which should
    duplicate it.

    Every branch either finds the exact already-visited child that `action`
    (and, for a reveal, its real resolved value) leads to, or returns None -
    there's no partial/uncertain middle ground, since an incorrect reuse
    would corrupt the search silently while a missed one only costs some
    redundant computation next time.
    """
    if cached_root is None:
        return None
    state = cached_root.state
    if gamestate_from_turn(turn_before) != state:
        return None

    representative = group_representatives(turn_before).get(action, action)
    edge = cached_root.edges.get(representative)
    if edge is None or edge.child is None or will_close_round(state, action):
        # Round-closing is never cached on its edge in the first place (see
        # this module's docstring) - nothing to carry forward.
        return None

    if is_reveal(state, action):
        chance = edge.child
        revealed_value = infer_revealed_value(turn_before, action, turn_after)
        chance_edge = (
            chance.edges.get(revealed_value)
            if isinstance(chance, ChanceNode) and revealed_value is not None
            else None
        )
        child = chance_edge.child if chance_edge is not None else None
    else:
        child = edge.child if isinstance(edge.child, MCTSNode) else None

    return child if isinstance(child, MCTSNode) and not child.is_terminal else None
