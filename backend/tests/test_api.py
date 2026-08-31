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


def test_get_mcts_models_lists_checkpoints_in_the_models_dir(tmp_path, monkeypatch):
    from skyjo.rl.checkpoint import save_checkpoint
    from skyjo.rl.network import AlphaZeroNet

    save_checkpoint(tmp_path / "strong.pt", AlphaZeroNet(), None, iteration=1, total_train_steps=1)
    monkeypatch.setenv("SKYJO_MCTS_MODELS_DIR", str(tmp_path))

    response = client.get("/matches/mcts-models")

    assert response.status_code == 200
    assert response.json() == ["strong"]


def test_upload_mcts_model_adds_it_to_the_available_list(tmp_path, monkeypatch):
    from skyjo.rl.checkpoint import save_checkpoint
    from skyjo.rl.network import AlphaZeroNet

    monkeypatch.setenv("SKYJO_MCTS_MODELS_DIR", str(tmp_path / "models"))
    source = tmp_path / "source.pt"
    save_checkpoint(source, AlphaZeroNet(), None, iteration=1, total_train_steps=1)

    response = client.post(
        "/matches/mcts-models",
        files={"file": ("uploaded.pt", source.read_bytes(), "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json() == "uploaded"
    assert client.get("/matches/mcts-models").json() == ["uploaded"]


def test_upload_mcts_model_rejects_a_non_checkpoint_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYJO_MCTS_MODELS_DIR", str(tmp_path))

    response = client.post(
        "/matches/mcts-models",
        files={"file": ("bad.pt", b"not a checkpoint", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert client.get("/matches/mcts-models").json() == []


def test_create_match_with_an_unknown_mcts_model_is_rejected():
    response = client.post(
        "/matches",
        json={
            "player_count": 2,
            "player_types": ["mcts_bot", "human"],
            "player_mcts_models": ["does-not-exist", None],
        },
    )

    assert response.status_code == 400


def test_create_match_rejects_a_player_mcts_models_length_mismatch():
    response = client.post(
        "/matches",
        json={"player_count": 2, "player_mcts_models": ["does-not-exist"]},
    )

    assert response.status_code == 400


def test_create_match_uses_the_given_mcts_num_simulations():
    response = client.post(
        "/matches",
        json={
            "player_count": 2,
            "player_types": ["mcts_bot", "human"],
            "player_mcts_num_simulations": [2, None],
        },
    )

    assert response.status_code == 201


def test_create_match_rejects_an_out_of_range_mcts_num_simulations():
    response = client.post(
        "/matches",
        json={
            "player_count": 2,
            "player_types": ["mcts_bot", "human"],
            "player_mcts_num_simulations": [0, None],
        },
    )

    assert response.status_code == 400


def test_create_match_rejects_a_player_mcts_num_simulations_length_mismatch():
    response = client.post(
        "/matches",
        json={"player_count": 2, "player_mcts_num_simulations": [2]},
    )

    assert response.status_code == 400


def test_create_match_uses_the_given_mcts_cap_root_lead():
    response = client.post(
        "/matches",
        json={
            "player_count": 2,
            "player_types": ["mcts_bot", "human"],
            "player_mcts_num_simulations": [2, None],
            "player_mcts_cap_root_lead": [True, False],
        },
    )

    assert response.status_code == 201


def test_create_match_rejects_a_player_mcts_cap_root_lead_length_mismatch():
    response = client.post(
        "/matches",
        json={"player_count": 2, "player_mcts_cap_root_lead": [True]},
    )

    assert response.status_code == 400


def test_an_mcts_bot_seat_auto_plays_its_initial_flip(monkeypatch):
    # Keeps the real AlphaZeroNet -> MCTS -> factory -> API path intact, just
    # with few enough simulations per move to stay fast in CI.
    monkeypatch.setenv("SKYJO_MCTS_NUM_SIMULATIONS", "2")

    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["mcts_bot", "human"]},
    ).json()["match_id"]

    body = _wait_for_idle(client, match_id, timeout=10.0)
    assert body["phase"] == "initial_flip"
    assert body["current_player"] == 1
    face_up = sum(1 for card in body["boards"][0]["cards"] if card["face_up"])
    assert face_up == 1


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

    def fake_create_bot(
        player_type, seed=None, mcts_model=None, num_simulations=None, cap_root_lead=False
    ):
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

    def fake_create_bot(
        player_type, seed=None, mcts_model=None, num_simulations=None, cap_root_lead=False
    ):
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

    def fake_create_bot(
        player_type, seed=None, mcts_model=None, num_simulations=None, cap_root_lead=False
    ):
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

    def fake_create_bot(
        player_type, seed=None, mcts_model=None, num_simulations=None, cap_root_lead=False
    ):
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


# --- GET /matches/{id}/history/{node_id}/mcts-tree -----------------------------


def _node_for_actor(match_id: str, actor: int) -> str:
    nodes = client.get(f"/matches/{match_id}/history").json()["nodes"]
    return next(n["node_id"] for n in nodes if n["actor"] == actor)


def test_mcts_tree_endpoint_returns_the_search_tree_behind_that_nodes_move(monkeypatch):
    monkeypatch.setenv("SKYJO_MCTS_NUM_SIMULATIONS", "2")

    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["mcts_bot", "human"]},
    ).json()["match_id"]
    _wait_for_idle(client, match_id, timeout=10.0)
    node_id = _node_for_actor(match_id, 0)

    response = client.get(f"/matches/{match_id}/history/{node_id}/mcts-tree")

    assert response.status_code == 200
    body = response.json()
    # A dict of progress-checkpoint snapshots (see MctsBot.last_tree_snapshots),
    # keyed by visit count as a string - not a single bare tree.
    assert body
    for tree in body.values():
        assert tree["kind"] == "decision"
        assert tree["edges"]
    final = body[str(max(int(k) for k in body))]
    assert final["visit_count"] > 0


def test_history_node_reports_has_mcts_tree_only_for_the_bots_move(monkeypatch):
    monkeypatch.setenv("SKYJO_MCTS_NUM_SIMULATIONS", "2")

    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["mcts_bot", "human"]},
    ).json()["match_id"]
    _wait_for_idle(client, match_id, timeout=10.0)

    nodes = client.get(f"/matches/{match_id}/history").json()["nodes"]
    root = next(n for n in nodes if n["actor"] is None)
    bot_move = next(n for n in nodes if n["actor"] == 0)
    assert root["has_mcts_tree"] is False
    assert bot_move["has_mcts_tree"] is True


def test_mcts_tree_endpoint_reflects_the_move_at_that_node_even_after_further_search(monkeypatch):
    # A later decision (this bot's own next move, from further along the same
    # match) must not retroactively change what an earlier node's tree looks
    # like - see MatchNode.mcts_tree's docstring.
    monkeypatch.setenv("SKYJO_MCTS_NUM_SIMULATIONS", "2")

    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["mcts_bot", "human"]},
    ).json()["match_id"]
    _wait_for_idle(client, match_id, timeout=10.0)
    first_node_id = _node_for_actor(match_id, 0)
    first_response = client.get(f"/matches/{match_id}/history/{first_node_id}/mcts-tree")
    assert first_response.status_code == 200
    first_tree = first_response.json()

    # The human's own flip_initial hands control back to seat 0's bot for its
    # *second* initial-flip decision (see test_bot_seat_finishes_its_initial_
    # flips_after_the_human_takes_a_turn), which searches - and would advance
    # the bot's own cached tree - all over again.
    client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})
    _wait_for_idle(client, match_id, timeout=10.0)

    second_response = client.get(f"/matches/{match_id}/history/{first_node_id}/mcts-tree")
    assert second_response.status_code == 200
    assert second_response.json() == first_tree


def test_mcts_tree_endpoint_404s_for_a_human_move():
    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["human", "human"]},
    ).json()["match_id"]
    client.post(f"/matches/{match_id}/actions", json={"type": "flip_initial", "position": 0})
    human_node_id = _node_for_actor(match_id, 0)

    response = client.get(f"/matches/{match_id}/history/{human_node_id}/mcts-tree")

    assert response.status_code == 404


def test_mcts_tree_endpoint_404s_for_a_bot_without_a_search_tree():
    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["random_bot", "human"]},
    ).json()["match_id"]
    node_id = _node_for_actor(match_id, 0)

    response = client.get(f"/matches/{match_id}/history/{node_id}/mcts-tree")

    assert response.status_code == 404


def test_mcts_tree_endpoint_404s_for_the_root_node():
    match_id = client.post(
        "/matches",
        json={"player_count": 2, "seed": 7, "player_types": ["mcts_bot", "human"]},
    ).json()["match_id"]
    root_id = client.get(f"/matches/{match_id}/history").json()["nodes"][0]["node_id"]

    response = client.get(f"/matches/{match_id}/history/{root_id}/mcts-tree")

    assert response.status_code == 404


def test_mcts_tree_endpoint_404s_for_an_unknown_node():
    match_id = client.post("/matches", json={"player_count": 2, "seed": 7}).json()["match_id"]

    response = client.get(f"/matches/{match_id}/history/does-not-exist/mcts-tree")

    assert response.status_code == 404


def test_mcts_tree_endpoint_on_unknown_match_returns_404():
    response = client.get("/matches/does-not-exist/history/does-not-exist/mcts-tree")

    assert response.status_code == 404
