from fastapi.testclient import TestClient

from skyjo.api import app

client = TestClient(app)


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
