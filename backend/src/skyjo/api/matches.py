from fastapi import APIRouter, HTTPException

from skyjo.api.schemas import (
    ActionRequest,
    MatchStateOut,
    NewMatchRequest,
    action_type_from_name,
)
from skyjo.api.store import MatchNotFoundError, MatchStore
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

    match_id = store.create(state)
    return MatchStateOut.from_state(match_id, state)


@router.get("/{match_id}", response_model=MatchStateOut)
def get_match(match_id: str) -> MatchStateOut:
    state = _get_state_or_404(match_id)
    return MatchStateOut.from_state(match_id, state)


@router.post("/{match_id}/actions", response_model=MatchStateOut)
def apply_match_action(match_id: str, request: ActionRequest) -> MatchStateOut:
    state = _get_state_or_404(match_id)
    action = Action(type=action_type_from_name(request.type), position=request.position)

    try:
        new_state = apply_action(state, action)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    store.update(match_id, new_state)
    return MatchStateOut.from_state(match_id, new_state)


@router.post("/{match_id}/next-round", response_model=MatchStateOut)
def start_match_next_round(match_id: str) -> MatchStateOut:
    state = _get_state_or_404(match_id)

    try:
        new_state = start_next_round(state)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    store.update(match_id, new_state)
    return MatchStateOut.from_state(match_id, new_state)


def _get_state_or_404(match_id: str) -> GameState:
    try:
        return store.get(match_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc
