"""Vector-valued MCTS: PUCT selection keyed to the active player's own
utility component, multi-agent backpropagation, Dirichlet root noise.

A `round_over` GameState is not a decision point (nobody chooses to start
the next round - `start_next_round` is a deterministic pass-through, mirrored
here from the same pattern the bot autoplay loop and match API use), so it's
never stored as a tree node: `_advance_state` folds it into the transition
that produced it. `game_over` states end the search: their value comes from
the actual final ranking, not a network estimate.

Player count is fixed for a whole match, so `n_act` - and therefore the
shape of every edge's value vector - is constant across an entire tree.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from skyjo.domain.engine import Action, GameState, apply_action, start_next_round

# priors: prob per legal action, normalized over legal actions only.
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
    visit_count: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    child: MCTSNode | None = None

    def __post_init__(self) -> None:
        if self.value_sum.shape != (self.n_act,):
            self.value_sum = np.zeros(self.n_act)

    def mean_value(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(self.n_act)
        return self.value_sum / self.visit_count


@dataclass
class MCTSNode:
    state: GameState
    n_act: int
    is_terminal: bool
    expanded: bool = False
    edges: dict[Action, MCTSEdge] = field(default_factory=dict)

    @property
    def visit_count(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())


def _is_terminal(state: GameState) -> bool:
    return state.phase == "game_over"


def _advance_state(state: GameState, action: Action) -> GameState:
    next_state = apply_action(state, action)
    while next_state.phase == "round_over":
        next_state = start_next_round(next_state)
    return next_state


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


def _expand(node: MCTSNode, priors: dict[Action, float]) -> None:
    node.edges = {
        action: MCTSEdge(action=action, prior=prior, n_act=node.n_act) for action, prior in priors.items()
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

    assert best_edge is not None, "_select_edge: node has no edges - should never be called on a leaf"
    return best_edge


def _simulate_once(root: MCTSNode, evaluate: EvaluateFn, c_puct: float) -> None:
    path: list[MCTSEdge] = []
    node = root

    while True:
        edge = _select_edge(node, c_puct)
        path.append(edge)

        if edge.child is None:
            child_state = _advance_state(node.state, edge.action)
            is_terminal = _is_terminal(child_state)
            child = MCTSNode(state=child_state, n_act=node.n_act, is_terminal=is_terminal)
            edge.child = child
            if is_terminal:
                value = _terminal_utility(child_state.total_scores)
            else:
                priors, value = evaluate(child_state)
                _expand(child, priors)
            break

        if edge.child.is_terminal:
            value = _terminal_utility(edge.child.state.total_scores)
            break

        node = edge.child

    for edge in path:
        edge.visit_count += 1
        edge.value_sum += value


def run_mcts(
    root_state: GameState,
    evaluate: EvaluateFn,
    *,
    num_simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
    add_root_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> MCTSNode:
    if num_simulations < 0:
        raise ValueError("run_mcts: num_simulations must be >= 0")
    rng = rng if rng is not None else np.random.default_rng()

    n_act = len(root_state.boards)
    root = MCTSNode(state=root_state, n_act=n_act, is_terminal=_is_terminal(root_state))
    if root.is_terminal:
        return root

    priors, _ = evaluate(root_state)
    _expand(root, priors)
    if add_root_noise:
        _apply_root_noise(root, dirichlet_alpha, dirichlet_epsilon, rng)

    for _ in range(num_simulations):
        _simulate_once(root, evaluate, c_puct)

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
