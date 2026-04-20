from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_api_smoke_health():
    response = client.get("/health")
    assert response.status_code == 200