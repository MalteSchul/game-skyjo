import threading

import pytest

from skyjo.api.store import AutoplayStatus, MatchBusyError, MatchStore
from skyjo.domain.engine import Action, ActionType, apply_action, new_match
from skyjo.domain.observation import Turn

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


class _RecordingObserverBot:
    """Implements `ObservesActions` but never actually acts - these tests
    only care about whether/when `observe_transition` gets called."""

    def __init__(self) -> None:
        self.calls: list[tuple[Turn, Action, Turn]] = []

    def choose_action(self, turn, *, report_progress=None):
        raise NotImplementedError("not exercised by these tests")

    def observe_transition(self, turn_before, action, turn_after) -> None:
        self.calls.append((turn_before, action, turn_after))


# --- observe_transition wiring: every seat's bot hears every real action -----


def test_apply_action_notifies_every_seats_observer_bot_not_just_the_actor():
    store = MatchStore()
    state = new_match(player_count=2, seed=1)
    observer_0, observer_1 = _RecordingObserverBot(), _RecordingObserverBot()
    match_id = store.create(state, ("Ada", "Grace"), ("human", "human"), (observer_0, observer_1))

    store.apply_action(match_id, _flip_action(0), _compute_flip(0))

    assert len(observer_0.calls) == 1
    assert len(observer_1.calls) == 1
    assert observer_0.calls[0] == observer_1.calls[0]
    turn_before, action, turn_after = observer_0.calls[0]
    assert action == _flip_action(0)
    assert turn_before.acting_player == 0
    assert turn_after.boards[0].cards[0] is not None and turn_after.boards[0].cards[0].face_up


def test_apply_autoplay_action_also_notifies_observer_bots():
    store = MatchStore()
    state = new_match(player_count=2, seed=1)
    observer = _RecordingObserverBot()
    match_id = store.create(state, ("Ada", "Grace"), ("human", "human"), (observer, None))
    store.set_thinking(match_id, player=0)

    store.apply_autoplay_action(match_id, _flip_action(0), _compute_flip(0))

    assert len(observer.calls) == 1


def test_start_next_round_does_not_notify_observer_bots():
    # A next_round edge carries no action - nothing for observe_transition to
    # report, and a freshly-dealt round is an independent shuffle anyway.
    store = MatchStore()
    state = new_match(player_count=2, seed=1)
    observer = _RecordingObserverBot()
    match_id = store.create(state, ("Ada", "Grace"), ("human", "human"), (observer, None))

    store.start_next_round(match_id, lambda state: state)

    assert observer.calls == []


def test_a_bot_that_does_not_implement_observes_actions_is_left_alone():
    store = MatchStore()
    state = new_match(player_count=2, seed=1)

    class _PlainBot:
        def choose_action(self, turn, *, report_progress=None):
            raise NotImplementedError

    match_id = store.create(state, ("Ada", "Grace"), ("human", "human"), (_PlainBot(), None))

    # Must not raise just because _PlainBot has no observe_transition.
    store.apply_action(match_id, _flip_action(0), _compute_flip(0))


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


# --- per-node mcts_tree storage ------------------------------------------------


def test_apply_autoplay_action_attaches_the_given_tree_to_the_new_node():
    store = MatchStore()
    match_id = _new_match_id(store)
    tree = {"kind": "decision", "edges": []}

    node, _, _ = store.apply_autoplay_action(match_id, _flip_action(0), _compute_flip(0), mcts_tree=tree)

    assert node.mcts_tree == tree
    assert store.get_node(match_id, node.node_id).mcts_tree == tree


def test_apply_action_leaves_mcts_tree_as_none():
    store = MatchStore()
    match_id = _new_match_id(store)

    node, _, _ = store.apply_action(match_id, _flip_action(0), _compute_flip(0))

    assert node.mcts_tree is None


def test_replaying_an_existing_edge_keeps_its_original_tree_rather_than_the_new_one():
    store = MatchStore()
    match_id = _new_match_id(store)
    root_id = store.get_head(match_id)[0].node_id
    original_tree = {"kind": "decision", "edges": ["original"]}
    node, _, _ = store.apply_autoplay_action(match_id, _flip_action(0), _compute_flip(0), mcts_tree=original_tree)
    store.goto(match_id, root_id)

    replayed, _, _ = store.apply_autoplay_action(
        match_id, _flip_action(0), _compute_flip(0), mcts_tree={"kind": "decision", "edges": ["different"]}
    )

    assert replayed.node_id == node.node_id
    assert replayed.mcts_tree == original_tree


def test_get_node_does_not_move_the_head():
    store = MatchStore()
    match_id = _new_match_id(store)
    root_id = store.get_head(match_id)[0].node_id
    store.apply_action(match_id, _flip_action(0), _compute_flip(0))
    head_before = store.get_head(match_id)[0].node_id

    fetched = store.get_node(match_id, root_id)

    assert fetched.node_id == root_id
    assert store.get_head(match_id)[0].node_id == head_before
