"""Round-trips a `GameState` through plain JSON-ready dicts, so a specific
position (including hidden card values, not just what a player could see)
can be written to a file and handed back in - e.g. as `--state-file` input
to `scripts/dump_mcts_tree.py`, instead of only ever starting from a fresh
`new_match()` deal.
"""

from __future__ import annotations

from typing import Any

from skyjo.domain.engine import Card, GameState, PlayerBoard


def _card_to_dict(card: Card | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return {"value": card.value, "face_up": card.face_up}


def _card_from_dict(data: dict[str, Any] | None) -> Card | None:
    if data is None:
        return None
    return Card(value=data["value"], face_up=data["face_up"])


def _board_to_dict(board: PlayerBoard) -> dict[str, Any]:
    return {"cards": [_card_to_dict(card) for card in board.cards]}


def _board_from_dict(data: dict[str, Any]) -> PlayerBoard:
    return PlayerBoard(cards=tuple(_card_from_dict(card) for card in data["cards"]))


def game_state_to_dict(state: GameState) -> dict[str, Any]:
    return {
        "boards": [_board_to_dict(board) for board in state.boards],
        "stock": list(state.stock),
        "discard": list(state.discard),
        "current_player": state.current_player,
        "drawn_card": state.drawn_card,
        "finisher": state.finisher,
        "players_awaiting_final_turn": sorted(state.players_awaiting_final_turn),
        "round_scores": list(state.round_scores) if state.round_scores is not None else None,
        "total_scores": list(state.total_scores),
        "phase": state.phase,
        "reshuffle_seed": state.reshuffle_seed,
        "target_score": state.target_score,
    }


def game_state_from_dict(data: dict[str, Any]) -> GameState:
    return GameState(
        boards=tuple(_board_from_dict(board) for board in data["boards"]),
        stock=tuple(data["stock"]),
        discard=tuple(data["discard"]),
        current_player=data["current_player"],
        drawn_card=data["drawn_card"],
        finisher=data["finisher"],
        players_awaiting_final_turn=frozenset(data["players_awaiting_final_turn"]),
        round_scores=tuple(data["round_scores"]) if data["round_scores"] is not None else None,
        total_scores=tuple(data["total_scores"]),
        phase=data["phase"],
        reshuffle_seed=data["reshuffle_seed"],
        target_score=data["target_score"],
    )
