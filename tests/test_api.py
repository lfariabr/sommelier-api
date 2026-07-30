"""FastAPI TestClient coverage — endpoints, golden prediction, validation 422s."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EXAMPLE_WINE
from ml.contract import load_assessment_contract

client = TestClient(app)
CONTRACT = load_assessment_contract()

EXPECTED_PATH_METHODS = {
    "/health": {"get"},
    "/features": {"get"},
    "/model/info": {"get"},
    "/predict/score": {"post"},
    "/predict/grade": {"post"},
    "/predict": {"post"},
}

EXPECTED_COMPONENT_SCHEMAS = {
    "WineFeatures": {
        "properties": {
            "fixed_acidity": "number",
            "volatile_acidity": "number",
            "citric_acid": "number",
            "residual_sugar": "number",
            "chlorides": "number",
            "free_sulfur_dioxide": "number",
            "total_sulfur_dioxide": "number",
            "density": "number",
            "pH": "number",
            "sulphates": "number",
            "alcohol": "number",
            "wine_type": "string",
        },
        "required": {
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        },
    },
    "ScoreResponse": {
        "properties": {"quality": "number"},
        "required": {"quality"},
    },
    "GradeResponse": {
        "properties": {
            "grade": "string",
            "label": "integer",
            "proba_high": "number",
            "proba_low": "number",
        },
        "required": {"grade", "label", "proba_high", "proba_low"},
    },
    "PredictResponse": {
        "properties": {"score": "number", "grade": None},
        "required": {"score", "grade"},
    },
}


def _schema_ref(path: str, location: str) -> str:
    operation = client.get("/openapi.json").json()["paths"][path]["post"]
    if location == "request":
        return operation["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ]
    return operation["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ]


def test_openapi_locks_public_routes_and_methods():
    """The v0.2.0 model swap must not remove or rename public operations."""
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == set(EXPECTED_PATH_METHODS)
    for path, expected_methods in EXPECTED_PATH_METHODS.items():
        assert set(paths[path]) == expected_methods


def test_prediction_endpoints_keep_request_and_response_models():
    """Clients keep using the same Pydantic models after prediction values change."""
    assert _schema_ref("/predict/score", "request").endswith("/WineFeatures")
    assert _schema_ref("/predict/score", "response").endswith("/ScoreResponse")
    assert _schema_ref("/predict/grade", "request").endswith("/WineFeatures")
    assert _schema_ref("/predict/grade", "response").endswith("/GradeResponse")
    assert _schema_ref("/predict", "request").endswith("/WineFeatures")
    assert _schema_ref("/predict", "response").endswith("/PredictResponse")


def test_openapi_locks_prediction_field_names_and_types():
    """Request and response keys and primitive types remain backward compatible."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    for schema_name, expected in EXPECTED_COMPONENT_SCHEMAS.items():
        schema = schemas[schema_name]
        assert set(schema["properties"]) == set(expected["properties"])
        assert set(schema["required"]) == expected["required"]
        for field_name, expected_type in expected["properties"].items():
            if expected_type is not None:
                assert schema["properties"][field_name]["type"] == expected_type

    assert schemas["PredictResponse"]["properties"]["grade"]["$ref"].endswith(
        "/GradeResponse"
    )


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
