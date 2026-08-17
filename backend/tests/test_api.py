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
