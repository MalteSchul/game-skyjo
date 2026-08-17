"""Skyjo rules engine: pure game logic, no I/O and no RL-specific concerns.

State transitions are functional (`apply_action(state, action) -> new GameState`)
so state is cheap to branch/clone and the whole module can later sit behind a
Gym-style `step()` for RL training. `legal_actions(state)` is the source of
truth for what's playable and is what a policy masks against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Literal

from skyjo.deck import build_deck

BOARD_SIZE = 12
COLUMNS = 4
MIN_PLAYERS = 2
MAX_PLAYERS = 8
INITIAL_FLIPS = 2
DEFAULT_TARGET_SCORE = 100

Phase = Literal["initial_flip", "awaiting_draw", "awaiting_placement", "round_over", "game_over"]


class IllegalActionError(Exception):
    """Raised when apply_action is given an action not in legal_actions(state)."""


class ActionType(Enum):
    FLIP_INITIAL = auto()
    DRAW_STOCK = auto()
    DRAW_DISCARD = auto()
    PLACE = auto()
    DISCARD_AND_REVEAL = auto()


@dataclass(frozen=True)
class Action:
    type: ActionType
    position: int | None = None


@dataclass(frozen=True)
class Card:
    value: int
    face_up: bool


@dataclass(frozen=True)
class PlayerBoard:
    # Length BOARD_SIZE, row-major over 3 rows x COLUMNS columns. None = a cleared column slot.
    cards: tuple[Card | None, ...]


@dataclass(frozen=True)
class GameState:
    boards: tuple[PlayerBoard, ...]
    stock: tuple[int, ...]  # next draw = stock[-1]
    discard: tuple[int, ...]  # top = discard[-1]
    current_player: int
    drawn_card: int | None
    finisher: int | None
    players_awaiting_final_turn: frozenset[int]
    round_scores: tuple[int, ...] | None
    total_scores: tuple[int, ...]
    phase: Phase
    # Running seed for any randomness after the initial deal (round re-deals,
    # mid-round discard-into-stock reshuffles). None means "unseeded match".
    reshuffle_seed: int | None
    target_score: int = DEFAULT_TARGET_SCORE


def new_match(player_count: int, seed: int | None = None) -> GameState:
    if seed is not None and not isinstance(seed, int):
        raise TypeError("new_match: seed must be an int if provided")
    if not isinstance(player_count, int) or not (MIN_PLAYERS <= player_count <= MAX_PLAYERS):
        raise ValueError(
            f"new_match: player_count must be an int between {MIN_PLAYERS} and {MAX_PLAYERS}"
        )

    total_scores = tuple(0 for _ in range(player_count))
    return _deal_round(total_scores, DEFAULT_TARGET_SCORE, seed)


def start_next_round(state: GameState) -> GameState:
    if state.phase != "round_over":
        raise IllegalActionError("start_next_round is only valid when phase == 'round_over'")
    return _deal_round(state.total_scores, state.target_score, state.reshuffle_seed)


def legal_actions(state: GameState) -> list[Action]:
    if state.phase in ("round_over", "game_over"):
        return []

    board = state.boards[state.current_player]

    if state.phase == "initial_flip":
        return [
            Action(ActionType.FLIP_INITIAL, position=i)
            for i, card in enumerate(board.cards)
            if card is not None and not card.face_up
        ]

    if state.phase == "awaiting_draw":
        actions = []
        if state.stock or len(state.discard) > 1:
            actions.append(Action(ActionType.DRAW_STOCK))
        if state.discard:
            actions.append(Action(ActionType.DRAW_DISCARD))
        return actions

    # awaiting_placement
    actions = []
    for i, card in enumerate(board.cards):
        if card is None:
            continue
        actions.append(Action(ActionType.PLACE, position=i))
        if not card.face_up:
            actions.append(Action(ActionType.DISCARD_AND_REVEAL, position=i))
    return actions


def apply_action(state: GameState, action: Action) -> GameState:
    if action not in legal_actions(state):
        raise IllegalActionError(f"{action} is not legal in phase {state.phase!r}")

    if action.type is ActionType.FLIP_INITIAL:
        return _apply_flip_initial(state, action.position)
    if action.type is ActionType.DRAW_STOCK:
        return _apply_draw_stock(state)
    if action.type is ActionType.DRAW_DISCARD:
        return _apply_draw_discard(state)
    if action.type is ActionType.PLACE:
        return _apply_place(state, action.position)
    return _apply_discard_and_reveal(state, action.position)


def _deal_round(
    total_scores: tuple[int, ...],
    target_score: int,
    reshuffle_seed: int | None,
) -> GameState:
    player_count = len(total_scores)
    deal_seed, reshuffle_seed = _advance_seed(reshuffle_seed)
    deck = list(build_deck(seed=deal_seed))

    boards = []
    for _ in range(player_count):
        hand, deck = deck[:BOARD_SIZE], deck[BOARD_SIZE:]
        boards.append(PlayerBoard(cards=tuple(Card(value=v, face_up=False) for v in hand)))

    top_discard, deck = deck[-1], deck[:-1]

    return GameState(
        boards=tuple(boards),
        stock=tuple(deck),
        discard=(top_discard,),
        current_player=0,
        drawn_card=None,
        finisher=None,
        players_awaiting_final_turn=frozenset(),
        round_scores=None,
        total_scores=total_scores,
        phase="initial_flip",
        reshuffle_seed=reshuffle_seed,
        target_score=target_score,
    )


def _advance_seed(seed: int | None) -> tuple[int | None, int | None]:
    if seed is None:
        return None, None
    return seed, seed + 1


def _apply_flip_initial(state: GameState, position: int) -> GameState:
    board = state.boards[state.current_player]
    revealed = replace(board.cards[position], face_up=True)
    new_board = replace(board, cards=_with_position(board.cards, position, revealed))
    boards = _with_index(state.boards, state.current_player, new_board)

    if all(_face_up_count(b) >= INITIAL_FLIPS for b in boards):
        return replace(state, boards=boards, current_player=_determine_starter(boards), phase="awaiting_draw")

    next_player = (state.current_player + 1) % len(state.boards)
    return replace(state, boards=boards, current_player=next_player)


def _face_up_count(board: PlayerBoard) -> int:
    return sum(1 for card in board.cards if card is not None and card.face_up)


def _determine_starter(boards: tuple[PlayerBoard, ...]) -> int:
    sums = [sum(c.value for c in b.cards if c is not None and c.face_up) for b in boards]
    # max() returns the first (lowest-index) match on ties.
    return max(range(len(boards)), key=lambda i: sums[i])


def _apply_draw_stock(state: GameState) -> GameState:
    stock, discard, reshuffle_seed = state.stock, state.discard, state.reshuffle_seed
    if not stock:
        stock, discard, reshuffle_seed = _reshuffle(discard, reshuffle_seed)

    drawn, stock = stock[-1], stock[:-1]
    return replace(
        state,
        stock=stock,
        discard=discard,
        drawn_card=drawn,
        phase="awaiting_placement",
        reshuffle_seed=reshuffle_seed,
    )


def _reshuffle(
    discard: tuple[int, ...], reshuffle_seed: int | None
) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    top, rest = discard[-1], list(discard[:-1])
    seed, next_seed = _advance_seed(reshuffle_seed)
    random.Random(seed).shuffle(rest)
    return tuple(rest), (top,), next_seed


def _apply_draw_discard(state: GameState) -> GameState:
    drawn, discard = state.discard[-1], state.discard[:-1]
    return replace(state, discard=discard, drawn_card=drawn, phase="awaiting_placement")


def _apply_place(state: GameState, position: int) -> GameState:
    board = state.boards[state.current_player]
    old_card = board.cards[position]
    new_cards = _with_position(board.cards, position, Card(value=state.drawn_card, face_up=True))
    new_board = _clear_completed_columns(replace(board, cards=new_cards), position)
    boards = _with_index(state.boards, state.current_player, new_board)
    discard = state.discard + (old_card.value,)
    return _finish_turn(replace(state, boards=boards, discard=discard, drawn_card=None))


def _apply_discard_and_reveal(state: GameState, position: int) -> GameState:
    board = state.boards[state.current_player]
    revealed = replace(board.cards[position], face_up=True)
    new_board = _clear_completed_columns(
        replace(board, cards=_with_position(board.cards, position, revealed)), position
    )
    boards = _with_index(state.boards, state.current_player, new_board)
    discard = state.discard + (state.drawn_card,)
    return _finish_turn(replace(state, boards=boards, discard=discard, drawn_card=None))


def _clear_completed_columns(board: PlayerBoard, position: int) -> PlayerBoard:
    col = position % COLUMNS
    positions = (col, col + COLUMNS, col + 2 * COLUMNS)
    cards = [board.cards[p] for p in positions]
    if all(c is not None and c.face_up for c in cards) and len({c.value for c in cards}) == 1:
        updated = list(board.cards)
        for p in positions:
            updated[p] = None
        return replace(board, cards=tuple(updated))
    return board


def _finish_turn(state: GameState) -> GameState:
    player = state.current_player
    finisher = state.finisher
    awaiting = state.players_awaiting_final_turn

    if finisher is None and _is_board_complete(state.boards[player]):
        finisher = player
        awaiting = frozenset(i for i in range(len(state.boards)) if i != player)
    elif player in awaiting:
        awaiting = awaiting - {player}

    if finisher is not None and not awaiting:
        return _score_and_close_round(state, finisher, awaiting)

    next_player = (player + 1) % len(state.boards)
    return replace(
        state,
        current_player=next_player,
        finisher=finisher,
        players_awaiting_final_turn=awaiting,
        phase="awaiting_draw",
    )


def _is_board_complete(board: PlayerBoard) -> bool:
    return all(card is None or card.face_up for card in board.cards)


def _score_and_close_round(state: GameState, finisher: int, awaiting: frozenset[int]) -> GameState:
    round_scores = tuple(
        sum(c.value for c in board.cards if c is not None) for board in state.boards
    )
    lowest = min(round_scores)
    finisher_is_sole_lowest = round_scores[finisher] == lowest and round_scores.count(lowest) == 1
    adjusted = tuple(
        score * 2 if i == finisher and not finisher_is_sole_lowest else score
        for i, score in enumerate(round_scores)
    )
    total_scores = tuple(t + a for t, a in zip(state.total_scores, adjusted, strict=True))
    phase: Phase = "game_over" if any(t >= state.target_score for t in total_scores) else "round_over"

    return replace(
        state,
        phase=phase,
        finisher=finisher,
        players_awaiting_final_turn=awaiting,
        round_scores=round_scores,
        total_scores=total_scores,
    )


def _with_position(
    cards: tuple[Card | None, ...], position: int, new_card: Card | None
) -> tuple[Card | None, ...]:
    return cards[:position] + (new_card,) + cards[position + 1 :]


def _with_index(
    boards: tuple[PlayerBoard, ...], index: int, new_board: PlayerBoard
) -> tuple[PlayerBoard, ...]:
    return boards[:index] + (new_board,) + boards[index + 1 :]
