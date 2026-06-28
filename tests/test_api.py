"""FastAPI TestClient coverage — endpoints, golden prediction, validation 422s."""
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EXAMPLE_WINE

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["sklearn_version"]


def test_features_endpoint():
    r = client.get("/features")
    assert r.status_code == 200
    assert r.json()["feature_order"][-1] == "wine_type"


def test_model_info_reports_real_metrics():
    r = client.get("/model/info")
    body = r.json()
    assert body["regression"]["r2"] > 0.4
    assert body["classification"]["roc_auc"] > 0.7
    assert body["dataset_rows"] == 6497


def test_predict_score_golden():
    r = client.post("/predict/score", json=EXAMPLE_WINE)
    assert r.status_code == 200
    assert 0 <= r.json()["quality"] <= 10


def test_predict_grade_probabilities_sum_to_one():
    r = client.post("/predict/grade", json=EXAMPLE_WINE)
    assert r.status_code == 200
    body = r.json()
    assert body["grade"] in {"high", "low"}
    assert abs(body["proba_high"] + body["proba_low"] - 1.0) < 1e-3


def test_predict_both():
    r = client.post("/predict", json=EXAMPLE_WINE)
    assert r.status_code == 200
    body = r.json()
    assert "score" in body
    assert body["grade"]["grade"] in {"high", "low"}


def test_validation_422_on_bad_wine_type():
    bad = {**EXAMPLE_WINE, "wine_type": "rosé"}
    assert client.post("/predict", json=bad).status_code == 422


def test_validation_422_on_negative_value():
    bad = {**EXAMPLE_WINE, "alcohol": -5}
    assert client.post("/predict", json=bad).status_code == 422
