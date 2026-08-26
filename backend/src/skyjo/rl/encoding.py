"""Encodes a `GameState` into the network's fixed-size input vector.

The vector's length does not depend on the match's player count: boards
beyond the active player count `N_act` are zero-padded up to `N_MAX_PLAYERS`.
That's what lets one network handle every match size from 2 to 8 players -
`N_act` itself is included as a one-hot feature so the network can tell a
padded slot from a genuinely empty one.

Players are encoded in *canonical* (rotated) order, not absolute seat order:
slot 0 always holds the state's `current_player`, slot 1 the next player in
turn order, and so on (see `rotation_perm`). This is the AlphaZero
board-flipping trick - it makes "whose turn it is" a structural property of
the input (always slot 0) instead of something the network has to learn from
a separate one-hot feature, and lets the same weights generalize across
seats. `StateEncoding.perm` records the permutation used, so callers can map
the network's per-player output (also in canonical order) back to absolute
seat indices - see `evaluator.py`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from skyjo.domain.action_equivalence import distinct_actions
from skyjo.domain.deck import CARD_COUNTS, DECK_SIZE
from skyjo.domain.engine import (
    BOARD_SIZE,
    DEFAULT_TARGET_SCORE,
    MAX_PLAYERS,
    MIN_PLAYERS,
    Action,
    GameState,
)
from skyjo.domain.observation import Turn
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

# Every discarded card and every face-up card (on any board) is public
# knowledge of exactly which values have been removed from play - Skyjo is a
# card-counting game, so how many of each value are still unseen (in the
# stock or face-down somewhere) is directly relevant to whether drawing from
# stock is worth the risk. `encode_state` only ever computes this from public
# state (discard/drawn_card/face-up board cards), never from stock's actual
# contents, so it can't leak hidden information.
_N_DECK_VALUES = len(CARD_COUNTS)

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
    + _ACTIVE_COUNT_CLASSES  # N_act, one-hot
    + _N_DECK_VALUES  # remaining-deck composition, one fraction per card value
)

INPUT_DIM = N_MAX_PLAYERS * _BOARD_FEATURES + GLOBAL_FEATURES + ACTION_SPACE_SIZE


def _normalize_value(value: int) -> float:
    return (value - _MIN_CARD_VALUE) / (_MAX_CARD_VALUE - _MIN_CARD_VALUE)


def _remaining_deck_histogram(state: GameState) -> list[float]:
    """One fraction per card value in `CARD_COUNTS` order: how much of that
    value's total count is still unseen (not in the discard pile, not the
    drawn card, not face-up on any board) - 1.0 means none of it has been
    seen yet, 0.0 means every copy is already accounted for. Face-down board
    cards and stock contents are genuinely unknown (to every player, not just
    the network), so they're correctly excluded from `visible` rather than
    treated as unseen-with-known-identity.
    """
    visible: Counter[int] = Counter(state.discard)
    if state.drawn_card is not None:
        visible[state.drawn_card] += 1
    for board in state.boards:
        for card in board.cards:
            if card is not None and card.face_up:
                visible[card.value] += 1

    return [max(0, total - visible[value]) / total for value, total in CARD_COUNTS]


def rotation_perm(current_player: int, n_act: int) -> np.ndarray:
    """`perm[slot]` is the absolute player id occupying canonical `slot`.
    `perm[0] == current_player` always; slots `n_act..N_MAX_PLAYERS` are
    unused padding, left as identity so the array is always fully defined.

    To rotate an absolute-order array `a` (len `n_act`) into canonical order:
    `a[perm[:n_act]]`. To un-rotate a canonical-order array `v` (len `n_act`)
    back to absolute order: `out = np.empty_like(v); out[perm[:n_act]] = v`.
    """
    perm = np.arange(N_MAX_PLAYERS)
    perm[:n_act] = (current_player + np.arange(n_act)) % n_act
    return perm


@dataclass(frozen=True)
class StateEncoding:
    features: np.ndarray  # shape (INPUT_DIM,), float32
    legal_action_mask: np.ndarray  # shape (ACTION_SPACE_SIZE,), bool
    perm: np.ndarray  # shape (N_MAX_PLAYERS,), see `rotation_perm`
    active_count: int  # N_act


def encode_state(state: GameState, *, legal_actions: Sequence[Action] | None = None) -> StateEncoding:
    """`legal_actions`, if given, is exactly the set of actions the mask
    should mark True and the policy head should normalize over - callers
    that already have it (an MCTS leaf's collapsed `distinct_actions`, see
    `evaluator.make_network_evaluator`) skip re-deriving it here. Omitted,
    it defaults to this state's own `distinct_actions(Turn.from_state(state))`
    - the *collapsed* representative set, not the full raw `legal_actions`
    list - so the network's policy head is never asked to spread probability
    mass (or learn a target of zero) across actions that are provably
    identical right now: see `domain.action_equivalence`. Every caller keeps
    working unchanged, since this was already true of `Turn.from_state`
    (raises IllegalActionError with no legal actions, i.e. round_over/
    game_over - never encoded anyway).
    """
    n_act = len(state.boards)
    if not (MIN_PLAYERS <= n_act <= MAX_PLAYERS):
        raise ValueError(f"encode_state: player_count {n_act} outside [{MIN_PLAYERS}, {MAX_PLAYERS}]")
    if legal_actions is None:
        legal_actions = distinct_actions(Turn.from_state(state))

    perm = rotation_perm(state.current_player, n_act)

    board_block = np.zeros((N_MAX_PLAYERS, _BOARD_FEATURES), dtype=np.float32)
    for canonical_slot in range(n_act):
        board = state.boards[perm[canonical_slot]]
        slot = np.zeros((BOARD_SIZE, _CARD_FEATURES), dtype=np.float32)
        for position, card in enumerate(board.cards):
            if card is None:
                continue
            slot[position, 0] = 1.0
            slot[position, 1] = 1.0 if card.face_up else 0.0
            slot[position, 2] = _normalize_value(card.value) if card.face_up else _ABSENT_VALUE
        board_block[canonical_slot] = slot.reshape(-1)

    mask = legal_action_mask(state, actions=legal_actions)

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
        finisher_onehot[(state.finisher - state.current_player) % n_act] = 1.0
    global_features += finisher_onehot

    awaiting_multihot = [0.0] * N_MAX_PLAYERS
    for player in state.players_awaiting_final_turn:
        awaiting_multihot[(player - state.current_player) % n_act] = 1.0
    global_features += awaiting_multihot

    total_scores_norm = [0.0] * N_MAX_PLAYERS
    for canonical_slot in range(n_act):
        total_scores_norm[canonical_slot] = state.total_scores[perm[canonical_slot]] / max(state.target_score, 1)
    global_features += total_scores_norm

    global_features += [state.target_score / DEFAULT_TARGET_SCORE]

    active_count_onehot = [0.0] * _ACTIVE_COUNT_CLASSES
    active_count_onehot[n_act - MIN_PLAYERS] = 1.0
    global_features += active_count_onehot

    global_features += _remaining_deck_histogram(state)

    features = np.concatenate(
        [board_block.reshape(-1), np.asarray(global_features, dtype=np.float32), mask.astype(np.float32)]
    )
    assert features.shape == (INPUT_DIM,), f"encode_state: built {features.shape}, expected ({INPUT_DIM},)"

    return StateEncoding(
        features=features,
        legal_action_mask=mask,
        perm=perm,
        active_count=n_act,
    )
