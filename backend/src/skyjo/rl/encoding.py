"""Encodes a `GameState` into the network's fixed-size input vector.

The vector's length does not depend on the match's player count: boards
beyond the active player count `N_act` are zero-padded up to `N_MAX_PLAYERS`.
That's what lets one network handle every match size from 2 to 8 players -
`N_act` itself is included as a one-hot feature so the network can tell a
padded slot from a genuinely empty one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skyjo.domain.deck import DECK_SIZE
from skyjo.domain.engine import (
    BOARD_SIZE,
    DEFAULT_TARGET_SCORE,
    MAX_PLAYERS,
    MIN_PLAYERS,
    GameState,
)
from skyjo.rl.action_space import ACTION_SPACE_SIZE, legal_action_mask

N_MAX_PLAYERS = MAX_PLAYERS  # 8, matches the rank head's 8x8 shape

_CARD_FEATURES = 3  # [present, face_up, normalized_value]
_BOARD_FEATURES = BOARD_SIZE * _CARD_FEATURES
_MIN_CARD_VALUE = -2
_MAX_CARD_VALUE = 12
_ACTIVE_COUNT_CLASSES = MAX_PLAYERS - MIN_PLAYERS + 1  # N_act in [2, 8] -> 7 classes

# _normalize_value always lands in [0, 1], so -1 is never a value a real card
# can take on - unlike 0, which collides with the lowest real card (-2). Used
# everywhere a "value" feature sits next to its own presence/face_up bit, so
# the value itself is unambiguous without needing the network to learn an
# AND-gate between the two.
_ABSENT_VALUE = -1.0

_PHASES = ("initial_flip", "awaiting_draw", "awaiting_placement", "round_over", "game_over")

GLOBAL_FEATURES = (
    2  # discard top: present, normalized value
    + 1  # discard count, normalized
    + 1  # stock count, normalized
    + len(_PHASES)  # phase one-hot
    + 2  # drawn card: present, normalized value
    + 1  # finisher present
    + N_MAX_PLAYERS  # finisher one-hot (all zero if no finisher)
    + N_MAX_PLAYERS  # players_awaiting_final_turn multi-hot
    + N_MAX_PLAYERS  # total_scores, normalized
    + 1  # target_score, normalized
    + N_MAX_PLAYERS  # active player k, one-hot
    + _ACTIVE_COUNT_CLASSES  # N_act, one-hot
)

INPUT_DIM = N_MAX_PLAYERS * _BOARD_FEATURES + GLOBAL_FEATURES + ACTION_SPACE_SIZE


def _normalize_value(value: int) -> float:
    return (value - _MIN_CARD_VALUE) / (_MAX_CARD_VALUE - _MIN_CARD_VALUE)


@dataclass(frozen=True)
class StateEncoding:
    features: np.ndarray  # shape (INPUT_DIM,), float32
    legal_action_mask: np.ndarray  # shape (ACTION_SPACE_SIZE,), bool
    active_player: int  # k
    active_count: int  # N_act


def encode_state(state: GameState) -> StateEncoding:
    n_act = len(state.boards)
    if not (MIN_PLAYERS <= n_act <= MAX_PLAYERS):
        raise ValueError(f"encode_state: player_count {n_act} outside [{MIN_PLAYERS}, {MAX_PLAYERS}]")

    board_block = np.zeros((N_MAX_PLAYERS, _BOARD_FEATURES), dtype=np.float32)
    for player, board in enumerate(state.boards):
        slot = np.zeros((BOARD_SIZE, _CARD_FEATURES), dtype=np.float32)
        for position, card in enumerate(board.cards):
            if card is None:
                continue
            slot[position, 0] = 1.0
            slot[position, 1] = 1.0 if card.face_up else 0.0
            slot[position, 2] = _normalize_value(card.value) if card.face_up else _ABSENT_VALUE
        board_block[player] = slot.reshape(-1)

    mask = legal_action_mask(state)

    global_features: list[float] = []
    discard_top = state.discard[-1] if state.discard else None
    global_features += [1.0 if discard_top is not None else 0.0]
    global_features += [_normalize_value(discard_top) if discard_top is not None else _ABSENT_VALUE]
    global_features += [len(state.discard) / DECK_SIZE]
    global_features += [len(state.stock) / DECK_SIZE]
    global_features += [1.0 if state.phase == phase else 0.0 for phase in _PHASES]
    global_features += [1.0 if state.drawn_card is not None else 0.0]
    global_features += [_normalize_value(state.drawn_card) if state.drawn_card is not None else _ABSENT_VALUE]
    global_features += [1.0 if state.finisher is not None else 0.0]

    finisher_onehot = [0.0] * N_MAX_PLAYERS
    if state.finisher is not None:
        finisher_onehot[state.finisher] = 1.0
    global_features += finisher_onehot

    awaiting_multihot = [0.0] * N_MAX_PLAYERS
    for player in state.players_awaiting_final_turn:
        awaiting_multihot[player] = 1.0
    global_features += awaiting_multihot

    total_scores_norm = [0.0] * N_MAX_PLAYERS
    for player in range(n_act):
        total_scores_norm[player] = state.total_scores[player] / max(state.target_score, 1)
    global_features += total_scores_norm

    global_features += [state.target_score / DEFAULT_TARGET_SCORE]

    active_player_onehot = [0.0] * N_MAX_PLAYERS
    active_player_onehot[state.current_player] = 1.0
    global_features += active_player_onehot

    active_count_onehot = [0.0] * _ACTIVE_COUNT_CLASSES
    active_count_onehot[n_act - MIN_PLAYERS] = 1.0
    global_features += active_count_onehot

    features = np.concatenate(
        [board_block.reshape(-1), np.asarray(global_features, dtype=np.float32), mask.astype(np.float32)]
    )
    assert features.shape == (INPUT_DIM,), f"encode_state: built {features.shape}, expected ({INPUT_DIM},)"

    return StateEncoding(
        features=features,
        legal_action_mask=mask,
        active_player=state.current_player,
        active_count=n_act,
    )
