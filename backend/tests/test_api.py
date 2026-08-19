import threading
import time

from fastapi.testclient import TestClient

from skyjo.api import app, matches

client = TestClient(app)


def _wait_for_idle(client: TestClient, match_id: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/matches/{match_id}").json()
        if body["status"] == "idle":
            return body
        time.sleep(0.01)
    raise AssertionError(f"match {match_id!r} did not reach idle status within {timeout}s")


class _SlowBot:
    """A bot whose choose_action blocks on a caller-controlled event, so tests
    can deterministically observe the API mid-decision instead of racing a
    real bot's (near-instant) resolution time."""

    def __init__(self, release: threading.Event) -> None:
        self._release = release

    def choose_action(self, turn, *, report_progress=None):
        if report_progress is not None:
            report_progress(0.5)
        self._release.wait(timeout=2.0)
        return turn.legal_actions[0]


def test_health_endpoint_responds_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- POST /matches ------------------------------------------------------------


def test_create_match_deals_a_round_with_hidden_cards_and_flip_actions():
    response = client.post("/matches", json={"player_count": 3, "seed": 42})

    assert response.status_code == 201
    body = response.json()
    assert body["phase"] == "initial_flip"
    assert len(body["boards"]) == 3
    assert all(len(board["cards"]) == 12 for board in body["boards"])
    # Cards are dealt face down, so their values must not leak over the wire.
    assert all(
        card["value"] is None and card["face_up"] is False
        for board in body["boards"]
        for card in board["cards"]
    )
    assert body["legal_actions"] == [
        {"type": "flip_initial", "position": i} for i in range(12)
    ]


def test_create_match_rejects_out_of_range_player_count():
    response = client.post("/matches", json={"player_count": 1})

    assert response.status_code == 400


def test_create_match_rejects_malformed_body():
    response = client.post("/matches", json={"player_count": "two"})

    assert response.status_code == 422


def test_create_match_without_names_defaults_to_player_n():
    response = client.post("/matches", json={"player_count": 3})

    assert response.json()["player_names"] == ["Player 1", "Player 2", "Player 3"]


def test_create_match_uses_the_given_player_names():
    response = client.post(
        "/matches", json={"player_count": 2, "player_names": ["Ada", "Grace"]}
    )

    assert response.json()["player_names"] == ["Ada", "Grace"]


def test_create_match_falls_back_to_default_for_a_blank_name():
    response = client.post("/matches", json={"player_count": 2, "player_names": ["Ada", "  "]})

    assert response.json()["player_names"] == ["Ada", "Player 2"]


def test_create_match_rejects_a_player_names_length_mismatch():
    response = client.post(
        "/matches", json={"player_count": 3, "player_names": ["Ada", "Grace"]}
    )

    assert response.status_code == 400


# --- GET /matches/{id} ----------------------------------------------------------


def test_get_match_returns_the_same_state_as_creation():
    created = client.post("/matches", json={"player_count": 2, "seed": 1}).json()

    response = client.get(f"/matches/{created['match_id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_match_with_unknown_id_returns_404():
    response = client.get("/matches/does-not-exist")

    assert response.status_code == 404


# --- POST /matches/{id}/actions --------------------------------------------------


def test_action_response_keeps_the_player_names_from_creation():
    created = client.post(
        "/matches", json={"player_count": 2, "seed": 7, "player_names": ["Ada", "Grace"]}
    ).json()

    response = client.post(
        f"/matches/{created['match_id']}/actions", json={"type": "flip_initial", "position": 0}
    )

    assert response.json()["player_names"] == ["Ada", "Grace"]


def test_flip_initial_action_reveals_a_card_and_advances_the_turn():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})

    assert response.status_code == 200
    body = response.json()
    assert body["boards"][0]["cards"][0]["face_up"] is True
    assert body["boards"][0]["cards"][0]["value"] is not None
    assert body["current_player"] == 1


def test_action_on_unknown_match_returns_404():
    response = client.post("/matches/does-not-exist/actions", json={"type": "flip_initial", "position": 0})

    assert response.status_code == 404


def test_illegal_action_returns_409():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.post(f"/matches/{match_id}/actions", json={"type": "draw_stock"})

    assert response.status_code == 409


def test_unknown_action_type_returns_422():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.post(f"/matches/{match_id}/actions", json={"type": "teleport"})

    assert response.status_code == 422


# --- POST /matches/{id}/next-round -----------------------------------------------


def test_next_round_before_round_is_over_returns_409():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.post(f"/matches/{match_id}/next-round")

    assert response.status_code == 409


def test_next_round_on_unknown_match_returns_404():
    response = client.post("/matches/does-not-exist/next-round")

    assert response.status_code == 404


# --- GET /matches/{id}/history -----------------------------------------------


def test_history_starts_with_just_the_root_node():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.get(f"/matches/{match_id}/history")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 1
    root = body["nodes"][0]
    assert root["parent_id"] is None
    assert root["edge"] == {"kind": "root", "action_type": None, "position": None}
    assert body["head_id"] == root["node_id"]


def test_history_records_a_node_per_action_and_advances_the_head():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})
    response = client.get(f"/matches/{match_id}/history")

    body = response.json()
    assert len(body["nodes"]) == 2
    leaf = next(n for n in body["nodes"] if n["node_id"] == body["head_id"])
    assert leaf["edge"] == {"kind": "action", "action_type": "flip_initial", "position": 0}
    assert leaf["actor"] == 0
    root_id = next(n for n in body["nodes"] if n["parent_id"] is None)["node_id"]
    assert leaf["parent_id"] == root_id


def test_history_on_unknown_match_returns_404():
    response = client.get("/matches/does-not-exist/history")

    assert response.status_code == 404


# --- POST /matches/{id}/history/{node_id}/goto --------------------------------


def test_goto_moves_the_head_back_to_an_earlier_node():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]
    root_id = client.get(f"/matches/{match_id}/history").json()["nodes"][0]["node_id"]
    client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})

    response = client.post(f"/matches/{match_id}/history/{root_id}/goto")

    assert response.status_code == 200
    body = response.json()
    assert body["boards"][0]["cards"][0]["face_up"] is False
    assert client.get(f"/matches/{match_id}").json() == body
    assert client.get(f"/matches/{match_id}/history").json()["head_id"] == root_id


def test_replaying_the_same_action_from_a_past_node_reuses_the_existing_branch():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]
    root_id = client.get(f"/matches/{match_id}/history").json()["nodes"][0]["node_id"]
    first = client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0}).json()

    client.post(f"/matches/{match_id}/history/{root_id}/goto")
    replayed = client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0}).json()

    assert replayed == first
    assert len(client.get(f"/matches/{match_id}/history").json()["nodes"]) == 2


def test_diverging_from_a_past_node_grows_a_new_branch_instead_of_overwriting():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]
    root_id = client.get(f"/matches/{match_id}/history").json()["nodes"][0]["node_id"]
    client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})

    client.post(f"/matches/{match_id}/history/{root_id}/goto")
    client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 1})

    history = client.get(f"/matches/{match_id}/history").json()
    assert len(history["nodes"]) == 3
    children_of_root = [n for n in history["nodes"] if n["parent_id"] == root_id]
    assert {n["edge"]["position"] for n in children_of_root} == {0, 1}


def test_goto_unknown_node_returns_404():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.post(f"/matches/{match_id}/history/does-not-exist/goto")

    assert response.status_code == 404


def test_goto_on_unknown_match_returns_404():
    response = client.post("/matches/does-not-exist/history/does-not-exist/goto")

    assert response.status_code == 404


# --- bot seats ------------------------------------------------------------------


def test_create_match_defaults_every_seat_to_human():
    response = client.post("/matches", json={"player_count": 3})

    assert response.json()["player_types"] == ["human", "human", "human"]


def test_create_match_uses_the_given_player_types():
    response = client.post(
        "/matches", json={"player_count": 2, "player_types": ["random_bot", "human"]}
    )

    assert response.json()["player_types"] == ["random_bot", "human"]


def test_create_match_rejects_a_player_types_length_mismatch():
    response = client.post(
        "/matches", json={"player_count": 3, "player_types": ["human", "random_bot"]}
    )

    assert response.status_code == 400


def test_create_match_rejects_an_unknown_player_type():
    response = client.post(
        "/matches", json={"player_count": 2, "player_types": ["human", "grandmaster"]}
    )

    assert response.status_code == 422


def test_a_bot_seat_auto_plays_its_initial_flip_before_the_response_is_returned():
    response = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["random_bot", "human"]},
    )

    body = response.json()
    assert body["phase"] == "initial_flip"
    assert body["current_player"] == 1
    face_up = sum(1 for card in body["boards"][0]["cards"] if card["face_up"])
    assert face_up == 1


def test_bot_seat_finishes_its_initial_flips_after_the_human_takes_a_turn():
    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["random_bot", "human"]},
    ).json()["match_id"]

    response = client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})

    body = response.json()
    face_up = sum(1 for card in body["boards"][0]["cards"] if card["face_up"])
    assert face_up == 2
    assert body["current_player"] == 1
    assert body["phase"] == "initial_flip"


def test_two_bot_seats_play_an_entire_round_automatically():
    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 3, "player_types": ["random_bot", "random_bot"]},
    ).json()["match_id"]

    # RandomBot resolves within the grace period nearly always, but polling
    # to idle (rather than asserting on the creation response directly) keeps
    # this robust regardless of how long that takes on a given machine.
    body = _wait_for_idle(client, match_id)
    assert body["phase"] in ("round_over", "game_over")
    assert body["legal_actions"] == []


def test_goto_does_not_trigger_additional_bot_auto_play():
    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["random_bot", "human"]},
    ).json()["match_id"]
    root_id = client.get(f"/matches/{match_id}/history").json()["nodes"][0]["node_id"]

    response = client.post(f"/matches/{match_id}/history/{root_id}/goto")

    body = response.json()
    face_up = sum(1 for card in body["boards"][0]["cards"] if card["face_up"])
    assert face_up == 0
    assert body["current_player"] == 0


# --- bot thinking status ---------------------------------------------------------


def test_a_fully_human_match_reports_idle_status():
    response = client.post("/matches", json={"player_count": 2, "seed": 7})

    body = response.json()
    assert body["status"] == "idle"
    assert body["thinking_player"] is None
    assert body["thinking_progress"] is None


def test_response_reports_thinking_status_while_a_slow_bot_is_still_deciding(monkeypatch):
    release = threading.Event()

    def fake_create_bot(player_type, seed=None):
        return None if player_type == "human" else _SlowBot(release)

    monkeypatch.setattr(matches, "create_bot", fake_create_bot)
    monkeypatch.setattr(matches, "AUTOPLAY_GRACE_SECONDS", 0.05)

    response = client.post(
        "/matches", json={"player_count": 2, "seed": 1, "player_types": ["random_bot", "human"]}
    )

    body = response.json()
    assert body["status"] == "thinking"
    assert body["thinking_player"] == 0
    assert body["thinking_progress"] == 0.5

    release.set()
    match_id = body["match_id"]
    _wait_for_idle(client, match_id)


def test_polling_get_match_reaches_idle_once_a_slow_bot_finishes_deciding(monkeypatch):
    release = threading.Event()

    def fake_create_bot(player_type, seed=None):
        return None if player_type == "human" else _SlowBot(release)

    monkeypatch.setattr(matches, "create_bot", fake_create_bot)
    monkeypatch.setattr(matches, "AUTOPLAY_GRACE_SECONDS", 0.05)

    match_id = client.post(
        "/matches", json={"player_count": 2, "seed": 1, "player_types": ["random_bot", "human"]}
    ).json()["match_id"]
    assert client.get(f"/matches/{match_id}").json()["status"] == "thinking"

    release.set()
    body = _wait_for_idle(client, match_id)
    face_up = sum(1 for card in body["boards"][0]["cards"] if card["face_up"])
    assert face_up == 1


def test_actions_endpoint_returns_409_while_a_slow_bot_is_still_deciding(monkeypatch):
    release = threading.Event()

    def fake_create_bot(player_type, seed=None):
        return None if player_type == "human" else _SlowBot(release)

    monkeypatch.setattr(matches, "create_bot", fake_create_bot)
    monkeypatch.setattr(matches, "AUTOPLAY_GRACE_SECONDS", 0.05)

    match_id = client.post(
        "/matches", json={"player_count": 2, "seed": 1, "player_types": ["random_bot", "human"]}
    ).json()["match_id"]

    response = client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})

    assert response.status_code == 409
    release.set()
    _wait_for_idle(client, match_id)


def test_goto_returns_409_while_a_slow_bot_is_still_deciding(monkeypatch):
    release = threading.Event()

    def fake_create_bot(player_type, seed=None):
        return None if player_type == "human" else _SlowBot(release)

    monkeypatch.setattr(matches, "create_bot", fake_create_bot)
    monkeypatch.setattr(matches, "AUTOPLAY_GRACE_SECONDS", 0.05)

    match_id = client.post(
        "/matches", json={"player_count": 2, "seed": 1, "player_types": ["random_bot", "human"]}
    ).json()["match_id"]
    root_id = client.get(f"/matches/{match_id}/history").json()["nodes"][0]["node_id"]

    response = client.post(f"/matches/{match_id}/history/{root_id}/goto")

    assert response.status_code == 409
    release.set()
    _wait_for_idle(client, match_id)
