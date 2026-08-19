"""Collapses provably-equivalent legal actions so anything that scores or
searches candidates - a bot, or MCTS - doesn't waste work on options that
can't differ.

Two actions targeting the same board are equivalent when nothing currently
known distinguishes their outcomes: the same action type, the same value at
the target slot (or both hidden), and the same set of already-revealed
values among the rest of that slot's column. That's everything a policy
could possibly condition its evaluation on right now - anything else about
the target (its exact position, which row it's in) has no rules meaning by
itself and can't affect the outcome.

This does NOT belong in legal_actions itself: the engine has to expose every
real board coordinate, since positions matter for the actual game mechanics
even when they're temporarily interchangeable from the current decision's
point of view. Collapsing is a policy-side optimization, not a rules concept.
Lives in `domain/` rather than under `bots/` because both bots and the RL
search need it, and neither should depend on the other.
"""

from __future__ import annotations

from skyjo.domain.engine import COLUMNS, Action
from skyjo.domain.observation import BoardView, Turn


def equivalence_key(action: Action, board: BoardView) -> tuple[object, ...]:
    position = action.position
    assert position is not None
    col = position % COLUMNS
    column_positions = (col, col + COLUMNS, col + 2 * COLUMNS)

    own_value = board.cards[position].value
    other_revealed_values = sorted(
        board.cards[p].value
        for p in column_positions
        if p != position and board.cards[p].value is not None
    )
    return (action.type, own_value, tuple(other_revealed_values))


def group_representatives(turn: Turn) -> dict[Action, Action]:
    """Maps every legal action to the one `distinct_actions` kept for its class.

    Iterates `turn.legal_actions` in order and, within each equivalence
    class, keeps the first action seen as the representative - matching
    `distinct_actions`'s own tie-break exactly, since it's built from this
    same mapping.
    """
    board = turn.boards[turn.acting_player]
    representative_by_key: dict[tuple[object, ...], Action] = {}
    mapping: dict[Action, Action] = {}

    for action in turn.legal_actions:
        if action.position is None:
            mapping[action] = action
            continue

        key = equivalence_key(action, board)
        representative = representative_by_key.setdefault(key, action)
        mapping[action] = representative

    return mapping


def distinct_actions(turn: Turn) -> tuple[Action, ...]:
    mapping = group_representatives(turn)
    seen: set[Action] = set()
    result: list[Action] = []

    for action in turn.legal_actions:
        representative = mapping[action]
        if representative in seen:
            continue
        seen.add(representative)
        result.append(representative)

    return tuple(result)
