"""FastAPI TestClient coverage — endpoints, golden prediction, validation 422s."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EXAMPLE_WINE
from ml.contract import load_assessment_contract

client = TestClient(app)
CONTRACT = load_assessment_contract()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["sklearn_version"]
    assert body["model_contract"] == CONTRACT["contract_version"]
    assert body["source_commit"] == CONTRACT["source_commit"]


def test_features_endpoint():
    r = client.get("/features")
    assert r.status_code == 200
    assert r.json()["feature_order"][-1] == "wine_type"


def test_model_info_reports_real_metrics():
    r = client.get("/model/info")
    body = r.json()
    assert body["regression"]["r2"] > 0.4
    assert body["classification"]["roc_auc"] > 0.7
    assert body["classification"]["sensitivity_low"] > 0.7
    assert body["classification"]["specificity_high"] > 0.7
    assert body["dataset_rows"] == 5320
    assert body["provenance"]["model_contract"] == CONTRACT["contract_version"]
    assert body["provenance"]["source_commit"] == CONTRACT["source_commit"]
    assert (
        body["provenance"]["source_selection_sha256"]
        == CONTRACT["source_selection_sha256"]
    )


def test_predict_score_golden():
    r = client.post("/predict/score", json=EXAMPLE_WINE)
    assert r.status_code == 200
    assert r.json()["quality"] == pytest.approx(5.065, abs=1e-12)


def test_predict_grade_probabilities_sum_to_one():
    r = client.post("/predict/grade", json=EXAMPLE_WINE)
    assert r.status_code == 200
    body = r.json()
    assert body["grade"] == "low"
    assert body["label"] == 1
    assert body["proba_high"] == pytest.approx(0.1681, abs=1e-4)
    assert body["proba_low"] == pytest.approx(0.8319, abs=1e-4)
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
