from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_response_shape():
    payload = {"exposure_route": "ingestion", "income_index": 1.0}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    expected_keys = {
        "polymer_type",
        "confidence",
        "particle_count",
        "mpri",
        "pspii",
        "bridge_score",
        "risk_tier",
        "details",
    }
    assert expected_keys.issubset(data.keys())
    assert 0 <= data["bridge_score"] <= 100
