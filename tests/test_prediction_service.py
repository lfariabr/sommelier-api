"""PredictionService — local path + api success + api-cold fallback."""
import httpx
from fastapi.testclient import TestClient

from api.main import app
from ui.config import EXAMPLE
from ui.services.prediction_service import PredictionService

client = TestClient(app)


def _feats():
    return dict(EXAMPLE)


def test_compute_local():
    out = PredictionService.compute(_feats(), "local", "")
    assert out["served_via"] == "local model"
    assert 0 <= out["score"] <= 10
    assert out["grade"]["grade"] in {"high", "low"}


def test_local_inference_matches_fastapi_for_example_wine(monkeypatch):
    local = PredictionService.compute(_feats(), "local", "")

    def _post(url, json, timeout):
        return client.post("/predict", json=json)

    monkeypatch.setattr(httpx, "post", _post)
    via_api = PredictionService.compute(_feats(), "api", "http://example.test")

    assert via_api["served_via"] == "live API"
    assert {
        "score": local["score"],
        "grade": local["grade"],
    } == {
        "score": via_api["score"],
        "grade": via_api["grade"],
    }


def test_compute_api_success(monkeypatch):
    fake = {"score": 5.5, "grade": {"grade": "high", "label": 0,
                                    "proba_high": 0.7, "proba_low": 0.3}}
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake

    def _post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _Resp()

    monkeypatch.setattr(httpx, "post", _post)
    out = PredictionService.compute(_feats(), "api", "http://example.test")
    assert out["served_via"] == "live API"
    assert out["score"] == 5.5
    assert captured["wine_type"] == "red"


def test_compute_api_fallback_on_cold(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("backend cold")

    monkeypatch.setattr(httpx, "post", _boom)
    out = PredictionService.compute(_feats(), "api", "http://localhost:9")
    assert "API cold" in out["served_via"]
    assert 0 <= out["score"] <= 10  # still works — fell back to local
