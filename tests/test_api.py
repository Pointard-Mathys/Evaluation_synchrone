import numpy as np
from fastapi.testclient import TestClient

import app as api_app


class DummyModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]])


client = TestClient(api_app.app)
AUTH_HEADERS = {"X-API-Token": "test-token"}


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(api_app, "feature_columns", ["tenure_months"])

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_returns_503_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(api_app, "model", None)
    monkeypatch.setattr(api_app, "feature_columns", ["tenure_months"])

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unhealthy"
    assert response.json()["detail"]["model_loaded"] is False


def test_health_returns_503_when_feature_columns_are_missing(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(api_app, "feature_columns", [])

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unhealthy"
    assert response.json()["detail"]["feature_columns_loaded"] is False


def test_predict_valid_input(monkeypatch):
    monkeypatch.setattr(api_app, "API_TOKEN", "test-token")
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(
        api_app,
        "feature_columns",
        [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_One year",
            "contract_Two year",
        ],
    )

    payload = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_rejects_missing_token(monkeypatch):
    monkeypatch.setattr(api_app, "API_TOKEN", "test-token")

    response = client.post("/predict", json={"tenure_months": 12})

    assert response.status_code == 401


def test_predict_rejects_missing_field(monkeypatch):
    monkeypatch.setattr(api_app, "API_TOKEN", "test-token")

    response = client.post("/predict", json={"tenure_months": 12}, headers=AUTH_HEADERS)

    assert response.status_code == 422


def test_metrics_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(api_app, "API_TOKEN", "test-token")

    response = client.get("/metrics", headers={"X-API-Token": "wrong-token"})

    assert response.status_code == 401
