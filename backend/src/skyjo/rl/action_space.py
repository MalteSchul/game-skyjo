"""Fixed-size discrete action space for the policy head.

`Action` from the engine is (type, position | None) with position possibly
None. The network needs a dense integer index per distinct action so a
single Linear layer can emit one logit per action, independent of which
state produced the legal-action list.
"""

from __future__ import annotations

import numpy as np

from skyjo.domain.engine import BOARD_SIZE, Action, ActionType, GameState, legal_actions

_POSITIONAL_TYPES = (ActionType.FLIP_INITIAL, ActionType.PLACE, ActionType.DISCARD_AND_REVEAL)
_NON_POSITIONAL_TYPES = (ActionType.DRAW_STOCK, ActionType.DRAW_DISCARD)

# Index layout: for each positional type, BOARD_SIZE consecutive slots (one
# per board position), then one slot per non-positional type. Order is
# arbitrary but must be stable, since it's baked into every trained network.
_TYPE_OFFSETS: dict[ActionType, int] = {}
_offset = 0
for _type in _POSITIONAL_TYPES:
    _TYPE_OFFSETS[_type] = _offset
    _offset += BOARD_SIZE
for _type in _NON_POSITIONAL_TYPES:
    _TYPE_OFFSETS[_type] = _offset
    _offset += 1

ACTION_SPACE_SIZE = _offset


def action_to_index(action: Action) -> int:
    offset = _TYPE_OFFSETS[action.type]
    if action.type in _POSITIONAL_TYPES:
        if action.position is None:
            raise ValueError(f"action_to_index: {action} of positional type must carry a position")
        return offset + action.position
    return offset


def index_to_action(index: int) -> Action:
    if not (0 <= index < ACTION_SPACE_SIZE):
        raise ValueError(f"index_to_action: {index} is out of range [0, {ACTION_SPACE_SIZE})")
    for action_type in _POSITIONAL_TYPES:
        offset = _TYPE_OFFSETS[action_type]
        if offset <= index < offset + BOARD_SIZE:
            return Action(type=action_type, position=index - offset)
    for action_type in _NON_POSITIONAL_TYPES:
        if _TYPE_OFFSETS[action_type] == index:
            return Action(type=action_type)
    raise AssertionError(f"index_to_action: {index} not covered by any action type - offsets out of sync")


def legal_action_mask(state: GameState) -> np.ndarray:
    """Boolean mask of shape (ACTION_SPACE_SIZE,), True at legal indices."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for action in legal_actions(state):
        mask[action_to_index(action)] = True
    return mask


def pi_to_vector(pi: dict[Action, float]) -> np.ndarray:
    """Dense (ACTION_SPACE_SIZE,) vector from a {action: prob} visit distribution."""
    vector = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action, prob in pi.items():
        vector[action_to_index(action)] = prob
    return vector
