from fastapi.testclient import TestClient

from skyjo_rl.api import app

client = TestClient(app)


def test_health_endpoint_responds_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
