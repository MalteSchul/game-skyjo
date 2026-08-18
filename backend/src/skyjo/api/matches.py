from fastapi import APIRouter, HTTPException

from skyjo.api.schemas import (
    ActionRequest,
    MatchHistoryOut,
    MatchStateOut,
    NewMatchRequest,
    action_type_from_name,
)
from skyjo.api.store import MatchNode, MatchNotFoundError, MatchStore, NodeNotFoundError
from skyjo.domain.engine import (
    Action,
    GameState,
    IllegalActionError,
    apply_action,
    new_match,
    start_next_round,
)

router = APIRouter(prefix="/matches", tags=["matches"])
store = MatchStore()


@router.post("", response_model=MatchStateOut, status_code=201)
def create_match(request: NewMatchRequest) -> MatchStateOut:
    try:
        state = new_match(player_count=request.player_count, seed=request.seed)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    player_names = _resolve_player_names(request.player_names, request.player_count)
    match_id = store.create(state, player_names)
    return MatchStateOut.from_state(match_id, state, player_names)


@router.get("/{match_id}", response_model=MatchStateOut)
def get_match(match_id: str) -> MatchStateOut:
    node, player_names = _get_head_or_404(match_id)
    return MatchStateOut.from_state(match_id, node.state, player_names)


@router.post("/{match_id}/actions", response_model=MatchStateOut)
def apply_match_action(match_id: str, request: ActionRequest) -> MatchStateOut:
    _get_head_or_404(match_id)
    action = Action(type=action_type_from_name(request.type), position=request.position)

    def compute(state: GameState) -> GameState:
        return apply_action(state, action)

    try:
        node, player_names = store.apply_action(match_id, action, compute)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return MatchStateOut.from_state(match_id, node.state, player_names)


@router.post("/{match_id}/next-round", response_model=MatchStateOut)
def start_match_next_round(match_id: str) -> MatchStateOut:
    _get_head_or_404(match_id)

    try:
        node, player_names = store.start_next_round(match_id, start_next_round)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return MatchStateOut.from_state(match_id, node.state, player_names)


@router.get("/{match_id}/history", response_model=MatchHistoryOut)
def get_match_history(match_id: str) -> MatchHistoryOut:
    try:
        nodes, head_id, _ = store.get_history(match_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc
    return MatchHistoryOut.from_nodes(nodes, head_id)


@router.post("/{match_id}/history/{node_id}/goto", response_model=MatchStateOut)
def goto_match_history_node(match_id: str, node_id: str) -> MatchStateOut:
    try:
        node, player_names = store.goto(match_id, node_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"history node {node_id!r} not found") from exc

    return MatchStateOut.from_state(match_id, node.state, player_names)


def _get_head_or_404(match_id: str) -> tuple[MatchNode, tuple[str, ...]]:
    try:
        return store.get_head(match_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc


def _resolve_player_names(names: list[str] | None, player_count: int) -> tuple[str, ...]:
    defaults = tuple(f"Player {i + 1}" for i in range(player_count))
    if names is None:
        return defaults
    if len(names) != player_count:
        raise HTTPException(
            status_code=400,
            detail=f"player_names must have exactly player_count ({player_count}) entries",
        )
    return tuple(name.strip() or defaults[i] for i, name in enumerate(names))
