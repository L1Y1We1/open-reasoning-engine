from fastapi.testclient import TestClient

from reasoning_engine.api import app


def test_health_endpoint_does_not_load_models() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["vector_store"] in {"qdrant", "milvus"}

