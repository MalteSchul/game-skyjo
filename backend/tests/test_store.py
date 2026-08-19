import threading

import pytest

from skyjo.api.store import AutoplayStatus, MatchBusyError, MatchStore
from skyjo.domain.engine import Action, ActionType, apply_action, new_match

# test_an_exception_in_work_still_resets_status_to_idle deliberately raises
# inside a background thread; pytest's unhandled-thread-exception capture
# fires once that thread's excepthook actually runs, which can race past the
# end of that specific test and land on whichever test happens to be running
# next - so the filter is applied file-wide rather than on that one test.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")

# --- fixture helpers ---------------------------------------------------------


def _new_match_id(store: MatchStore) -> str:
    state = new_match(player_count=2, seed=1)
    return store.create(state, ("Ada", "Grace"), ("human", "human"), (None, None))


def _flip_action(position: int) -> Action:
    return Action(type=ActionType.FLIP_INITIAL, position=position)


def _compute_flip(position: int):
    def compute(state):
        return apply_action(state, _flip_action(position))

    return compute


# --- trigger_autoplay ---------------------------------------------------------


def test_trigger_autoplay_runs_work_on_a_background_thread_and_reports_idle_when_done():
    store = MatchStore()
    match_id = _new_match_id(store)
    ran_on_thread = threading.Event()

    def work(mid: str) -> None:
        ran_on_thread.set()

    event = store.trigger_autoplay(match_id, work)

    assert event.wait(timeout=2.0)
    assert ran_on_thread.is_set()
    assert store.get_status(match_id) == AutoplayStatus(status="idle", player=None, progress=None)


def test_thinking_status_and_progress_are_observable_while_work_is_still_running():
    store = MatchStore()
    match_id = _new_match_id(store)
    started = threading.Event()
    release = threading.Event()

    def work(mid: str) -> None:
        store.set_thinking(mid, player=1)
        store.set_thinking_progress(mid, 0.5)
        started.set()
        release.wait(timeout=2.0)

    event = store.trigger_autoplay(match_id, work)
    assert started.wait(timeout=2.0)

    status = store.get_status(match_id)
    assert status == AutoplayStatus(status="thinking", player=1, progress=0.5)

    release.set()
    assert event.wait(timeout=2.0)
    assert store.get_status(match_id).status == "idle"


def test_a_second_trigger_while_one_is_running_reuses_the_same_event():
    store = MatchStore()
    match_id = _new_match_id(store)
    release = threading.Event()

    def work(mid: str) -> None:
        release.wait(timeout=2.0)

    first_event = store.trigger_autoplay(match_id, work)
    second_event = store.trigger_autoplay(match_id, lambda mid: None)

    assert second_event is first_event
    release.set()
    assert first_event.wait(timeout=2.0)


def test_an_exception_in_work_still_resets_status_to_idle():
    store = MatchStore()
    match_id = _new_match_id(store)

    def work(mid: str) -> None:
        store.set_thinking(mid, player=0)
        raise RuntimeError("boom")

    # The exception is expected to escape the background thread - there's
    # nothing there to catch it. See the module-level filterwarnings above.
    event = store.trigger_autoplay(match_id, work)

    assert event.wait(timeout=2.0)
    assert store.get_status(match_id) == AutoplayStatus(status="idle", player=None, progress=None)


def test_set_thinking_progress_is_a_no_op_once_status_is_idle_again():
    store = MatchStore()
    match_id = _new_match_id(store)

    store.set_thinking_progress(match_id, 0.9)

    assert store.get_status(match_id) == AutoplayStatus(status="idle", player=None, progress=None)


# --- busy guard ----------------------------------------------------------------


def test_apply_action_raises_while_a_bot_is_thinking():
    store = MatchStore()
    match_id = _new_match_id(store)
    store.set_thinking(match_id, player=0)

    with pytest.raises(MatchBusyError):
        store.apply_action(match_id, _flip_action(0), _compute_flip(0))


def test_start_next_round_raises_while_a_bot_is_thinking():
    store = MatchStore()
    match_id = _new_match_id(store)
    store.set_thinking(match_id, player=0)

    with pytest.raises(MatchBusyError):
        store.start_next_round(match_id, lambda state: state)


def test_goto_raises_while_a_bot_is_thinking():
    store = MatchStore()
    match_id = _new_match_id(store)
    root_id = store.get_head(match_id)[0].node_id
    store.set_thinking(match_id, player=0)

    with pytest.raises(MatchBusyError):
        store.goto(match_id, root_id)


def test_apply_autoplay_action_is_allowed_while_thinking():
    store = MatchStore()
    match_id = _new_match_id(store)
    store.set_thinking(match_id, player=0)

    node, _, _ = store.apply_autoplay_action(match_id, _flip_action(0), _compute_flip(0))

    assert node.state.boards[0].cards[0].face_up is True


def test_apply_action_is_allowed_again_once_a_pending_autoplay_finishes():
    store = MatchStore()
    match_id = _new_match_id(store)
    event = store.trigger_autoplay(match_id, lambda mid: None)
    assert event.wait(timeout=2.0)

    node, _, _ = store.apply_action(match_id, _flip_action(0), _compute_flip(0))

    assert node.state.boards[0].cards[0].face_up is True
