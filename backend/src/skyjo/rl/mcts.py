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

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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
EvaluateFn = Callable[[GameState], "tuple[dict[Action, float], np.ndarray]"]

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


def _select_edge(node: MCTSNode, c_puct: float) -> MCTSEdge:
    k = node.state.current_player
    sqrt_total = math.sqrt(node.visit_count)

    best_edge: MCTSEdge | None = None
    best_score = -math.inf
    for edge in node.edges.values():
        q = edge.mean_value()[k]
        u = c_puct * edge.prior * sqrt_total / (1 + edge.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_edge = edge

    assert best_edge is not None, (
        "_select_edge: node has no edges - should never be called on a leaf"
    )
    return best_edge


def _expand_child(
    state: GameState, evaluate: EvaluateFn, n_act: int
) -> tuple[MCTSNode, np.ndarray]:
    terminal = _is_terminal(state)
    child = MCTSNode(state=state, n_act=n_act, is_terminal=terminal)
    if terminal:
        child.value = _terminal_utility(state.total_scores)
        return child, child.value

    priors, value = evaluate(state)
    _expand(child, Turn.from_state(state), priors)
    child.value = value
    return child, value


def _is_round_closing(state: GameState, action: Action) -> bool:
    return will_close_round(state, action)


def _advance_deterministic(state: GameState, action: Action) -> GameState:
    """Neither a reveal nor round-closing: nothing hidden is involved."""
    return rescrub(apply_action(state, action))


def _advance_resolved_reveal(state: GameState, action: Action, value: int) -> GameState:
    """A single-card reveal, already resolved to `value` by a ChanceEdge."""
    if action.type is ActionType.DRAW_STOCK:
        return rescrub(resolve_drawn_stock_card(apply_action(state, action), value))
    primed = resolve_reveal(state, action, value)
    return rescrub(apply_action(primed, action))


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
    next_state = rescrub(apply_action(patched, action))
    while next_state.phase == "round_over":
        next_state = rescrub(start_next_round(next_state))
    return next_state


def _simulate_once(
    root: MCTSNode, evaluate: EvaluateFn, c_puct: float, rng: np.random.Generator
) -> None:
    path: list[MCTSEdge | ChanceEdge] = []
    node = root

    while True:
        edge = _select_edge(node, c_puct)
        path.append(edge)
        state = node.state

        if _is_round_closing(state, edge.action):
            # Never cached on edge.child: every visit resamples the reveal
            # and the round-close together, so this edge's own Q is an
            # average over many independently sampled continuations rather
            # than one frozen future (see module docstring).
            child_state = _advance_round_closing(state, edge.action, rng)
            child, value = _expand_child(child_state, evaluate, node.n_act)
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
                child, value = _expand_child(child_state, evaluate, node.n_act)
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
            child, value = _expand_child(child_state, evaluate, node.n_act)
            edge.child = child
            break
        if edge.child.is_terminal:
            value = _terminal_utility(edge.child.state.total_scores)
            break
        node = edge.child

    for visited in path:
        visited.visit_count += 1
        visited.value_sum += value


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
) -> MCTSNode:
    """`on_root_ready`, if given, fires exactly once - after the root is
    expanded and root noise applied, before the first simulation - with the
    same `MCTSNode` object this call will keep mutating and eventually
    return. It exists so a caller can snapshot the tree mid-search (e.g. via
    `on_simulation`) without waiting for `run_mcts` to return: the object it
    receives *is* the live root, not a copy.
    """
    if num_simulations < 0:
        raise ValueError("run_mcts: num_simulations must be >= 0")
    rng = rng if rng is not None else np.random.default_rng()

    n_act = len(root_turn.boards)
    root_state = gamestate_from_turn(root_turn)
    root = MCTSNode(state=root_state, n_act=n_act, is_terminal=False)

    priors, value = evaluate(root_state)
    _expand(root, root_turn, priors)
    root.value = value
    if add_root_noise:
        _apply_root_noise(root, dirichlet_alpha, dirichlet_epsilon, rng)

    if on_root_ready is not None:
        on_root_ready(root)

    for i in range(num_simulations):
        _simulate_once(root, evaluate, c_puct, rng)
        if on_simulation is not None:
            on_simulation(i + 1)

    return root


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
