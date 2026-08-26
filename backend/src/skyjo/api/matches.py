from fastapi import APIRouter, File, HTTPException, UploadFile

from skyjo.api.schemas import (
    ActionRequest,
    MatchHistoryOut,
    MatchStateOut,
    NewMatchRequest,
    action_type_from_name,
)
from skyjo.api.store import (
    AutoplayStatus,
    MatchBusyError,
    MatchNode,
    MatchNotFoundError,
    MatchStore,
    NodeNotFoundError,
)
from skyjo.bots.base import ExposesSearchTree
from skyjo.bots.factory import create_bot
from skyjo.bots.mcts_bot import list_available_models, save_uploaded_model
from skyjo.domain.engine import (
    Action,
    GameState,
    IllegalActionError,
    apply_action,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn

router = APIRouter(prefix="/matches", tags=["matches"])
store = MatchStore()

# A bot's move is a single `apply_action` call, so an entire round of mixed
# human/bot play (initial_flip through round_over) is well under this - it's
# only a backstop against a pathological future bot never converging.
MAX_BOT_AUTOPLAY_STEPS = 2000

# How long a request blocks hoping the bot(s) it triggers finish inline. An
# instant bot (e.g. RandomBot) always resolves well within this, so the
# caller gets the fully-played-out state in one round trip, same as before
# autoplay moved to a background thread. A slow bot (e.g. MCTS) times this
# out; the caller gets a "thinking" status back and polls GET for the rest.
AUTOPLAY_GRACE_SECONDS = 0.2


@router.get("/mcts-models", response_model=list[str])
def get_mcts_models() -> list[str]:
    """Names selectable as an mcts_bot seat's `player_mcts_models` entry - see
    `bots.mcts_bot.list_available_models`. Registered ahead of the
    "/{match_id}" GET route so "mcts-models" isn't parsed as a match id."""
    return list_available_models()


@router.post("/mcts-models", response_model=str)
async def upload_mcts_model(file: UploadFile = File(...)) -> str:  # noqa: B008 - FastAPI's own dependency-injection idiom, evaluated by FastAPI itself, not on each call
    """Uploads a checkpoint file (as saved by `rl.checkpoint.save_checkpoint`)
    and makes it selectable as an mcts_bot seat's `mcts_model` under a name
    derived from the uploaded filename. Returns that name."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    content = await file.read()
    try:
        return save_uploaded_model(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("", response_model=MatchStateOut, status_code=201)
def create_match(request: NewMatchRequest) -> MatchStateOut:
    try:
        state = new_match(player_count=request.player_count, seed=request.seed)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    player_names = _resolve_player_names(request.player_names, request.player_count)
    player_types = _resolve_player_types(request.player_types, request.player_count)
    mcts_models = _resolve_mcts_models(request.player_mcts_models, request.player_count)
    mcts_num_simulations = _resolve_mcts_num_simulations(request.player_mcts_num_simulations, request.player_count)
    try:
        bots = tuple(
            create_bot(
                player_type,
                seed=None if request.seed is None else request.seed + i,
                mcts_model=mcts_models[i],
                num_simulations=mcts_num_simulations[i],
            )
            for i, player_type in enumerate(player_types)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    match_id = store.create(state, player_names, player_types, bots)
    node, player_names, player_types, status = _trigger_and_await(match_id)
    return MatchStateOut.from_state(match_id, node.state, player_names, player_types, status)


@router.get("/{match_id}", response_model=MatchStateOut)
def get_match(match_id: str) -> MatchStateOut:
    node, player_names, player_types = _get_head_or_404(match_id)
    status = store.get_status(match_id)
    return MatchStateOut.from_state(match_id, node.state, player_names, player_types, status)


@router.post("/{match_id}/actions", response_model=MatchStateOut)
def apply_match_action(match_id: str, request: ActionRequest) -> MatchStateOut:
    _get_head_or_404(match_id)
    action = Action(type=action_type_from_name(request.type), position=request.position)

    def compute(state: GameState) -> GameState:
        return apply_action(state, action)

    try:
        store.apply_action(match_id, action, compute)
        node, player_names, player_types, status = _trigger_and_await(match_id)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MatchBusyError as exc:
        raise HTTPException(status_code=409, detail="match is busy: a bot is still deciding") from exc

    return MatchStateOut.from_state(match_id, node.state, player_names, player_types, status)


@router.post("/{match_id}/next-round", response_model=MatchStateOut)
def start_match_next_round(match_id: str) -> MatchStateOut:
    _get_head_or_404(match_id)

    try:
        store.start_next_round(match_id, start_next_round)
        node, player_names, player_types, status = _trigger_and_await(match_id)
    except IllegalActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MatchBusyError as exc:
        raise HTTPException(status_code=409, detail="match is busy: a bot is still deciding") from exc

    return MatchStateOut.from_state(match_id, node.state, player_names, player_types, status)


@router.get("/{match_id}/history/{node_id}/mcts-tree")
def get_history_node_mcts_tree(match_id: str, node_id: str) -> dict:
    """The search tree behind `node_id`'s move (see `MatchNode.mcts_tree`),
    captured back when that node was created - so this reflects whichever
    move that was at the time, not whatever the bot happens to be searching
    right now. Lets the history panel show the tree for any past bot move,
    not just the current head. 404s for an unknown match/node, a human move,
    or a bot with no real search (e.g. HeuristicBot)."""
    try:
        node = store.get_node(match_id, node_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"history node {node_id!r} not found") from exc
    if node.mcts_tree is None:
        raise HTTPException(status_code=404, detail=f"history node {node_id!r} has no search tree")
    return node.mcts_tree


@router.get("/{match_id}/history", response_model=MatchHistoryOut)
def get_match_history(match_id: str) -> MatchHistoryOut:
    try:
        nodes, head_id, _ = store.get_history(match_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc
    return MatchHistoryOut.from_nodes(nodes, head_id)


@router.post("/{match_id}/history/{node_id}/goto", response_model=MatchStateOut)
def goto_match_history_node(match_id: str, node_id: str) -> MatchStateOut:
    # Deliberately does not auto-play bots: goto is pure navigation to a state
    # that already exists in the tree, not a new decision point.
    try:
        node, player_names, player_types = store.goto(match_id, node_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"history node {node_id!r} not found") from exc
    except MatchBusyError as exc:
        raise HTTPException(status_code=409, detail="match is busy: a bot is still deciding") from exc

    status = store.get_status(match_id)
    return MatchStateOut.from_state(match_id, node.state, player_names, player_types, status)


def _get_head_or_404(match_id: str) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
    try:
        return store.get_head(match_id)
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"match {match_id!r} not found") from exc


def _trigger_and_await(match_id: str) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...], AutoplayStatus]:
    """Kicks off (or joins, if one is already running) the match's autoplay
    thread and waits up to AUTOPLAY_GRACE_SECONDS for it to finish, then
    returns the head as it stands at that point: fully resolved if the bot(s)
    finished in time, still "thinking" otherwise."""
    event = store.trigger_autoplay(match_id, _run_autoplay_loop)
    event.wait(timeout=AUTOPLAY_GRACE_SECONDS)
    node, player_names, player_types = store.get_head(match_id)
    return node, player_names, player_types, store.get_status(match_id)


def _run_autoplay_loop(match_id: str) -> None:
    """Plays out consecutive bot turns from the current head until it's a
    human's turn or the round/match ends. A single human action can hand
    control to several bot seats in a row, and initial_flip needs more than
    one action per seat, hence the loop rather than a single bot move. Runs
    on the background thread started by MatchStore.trigger_autoplay - use
    `apply_autoplay_action`, not `apply_action`, since this *is* the thread
    that's allowed to act while status is "thinking"."""
    node, _player_names, _player_types = store.get_head(match_id)
    for _ in range(MAX_BOT_AUTOPLAY_STEPS):
        if node.state.phase in ("round_over", "game_over"):
            return

        bot = store.get_bot(match_id, node.state.current_player)
        if bot is None:
            return

        store.set_thinking(match_id, node.state.current_player)
        turn = Turn.from_state(node.state)
        action = bot.choose_action(turn, report_progress=lambda p: store.set_thinking_progress(match_id, p))
        # Captured here, right as this decision is made - not after
        # apply_autoplay_action returns, since observe_transition (called for
        # every seat's bot as part of that) walks a cached MCTS root forward
        # past this exact search the moment the move is applied (see
        # MctsBot.last_tree_snapshots's docstring).
        mcts_tree = bot.last_tree_snapshots() if isinstance(bot, ExposesSearchTree) else None

        def compute(state: GameState, action: Action = action) -> GameState:
            return apply_action(state, action)

        node, _player_names, _player_types = store.apply_autoplay_action(match_id, action, compute, mcts_tree=mcts_tree)

    raise RuntimeError(f"bot auto-play exceeded {MAX_BOT_AUTOPLAY_STEPS} steps for match {match_id!r}")


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


def _resolve_player_types(types: list[str] | None, player_count: int) -> tuple[str, ...]:
    if types is None:
        return tuple("human" for _ in range(player_count))
    if len(types) != player_count:
        raise HTTPException(
            status_code=400,
            detail=f"player_types must have exactly player_count ({player_count}) entries",
        )
    return tuple(types)


def _resolve_mcts_models(models: list[str | None] | None, player_count: int) -> tuple[str | None, ...]:
    if models is None:
        return tuple(None for _ in range(player_count))
    if len(models) != player_count:
        raise HTTPException(
            status_code=400,
            detail=f"player_mcts_models must have exactly player_count ({player_count}) entries",
        )
    return tuple(models)


# Keeps a misconfigured seat from making a request block indefinitely or
# pegging the CPU - not a claim that either bound is a strong/fast search, just
# a sane outer fence around what the UI's number input allows.
MIN_MCTS_NUM_SIMULATIONS = 1
MAX_MCTS_NUM_SIMULATIONS = 1_000_000


def _resolve_mcts_num_simulations(values: list[int | None] | None, player_count: int) -> tuple[int | None, ...]:
    if values is None:
        return tuple(None for _ in range(player_count))
    if len(values) != player_count:
        raise HTTPException(
            status_code=400,
            detail=f"player_mcts_num_simulations must have exactly player_count ({player_count}) entries",
        )
    for value in values:
        if value is not None and not (MIN_MCTS_NUM_SIMULATIONS <= value <= MAX_MCTS_NUM_SIMULATIONS):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"player_mcts_num_simulations entries must be between {MIN_MCTS_NUM_SIMULATIONS} "
                    f"and {MAX_MCTS_NUM_SIMULATIONS}, got {value}"
                ),
            )
    return tuple(values)
