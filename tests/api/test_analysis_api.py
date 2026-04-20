from fastapi.testclient import TestClient

from app.api.main import app
from app.api.dependencies import get_orchestrator


class FakeOrchestrator:
    def run(self, product_id: str, query: str, top_k: int = 3):
        return {
            "plan": {
                "use_data": True,
                "use_report": True,
            },
            "final_output": {
                "product_id": product_id,
                "query": query,
                "report": "Test report",
            },
        }


def override_orchestrator():
    return FakeOrchestrator()


app.dependency_overrides[get_orchestrator] = override_orchestrator
client = TestClient(app)


def test_analyze_endpoint():
    payload = {
        "product_id": "B09SPZPDJK",
        "query": "sound quality",
        "top_k": 3,
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "plan" in data
    assert "final_output" in data
    assert data["final_output"]["product_id"] == "B09SPZPDJK"