"""FastAPI service — thin adapter over ml/predict.py.

Run locally:  python -m api.main   (or `make api`)
Prod (Render): uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.schemas import (
    GradeResponse,
    PredictResponse,
    ScoreResponse,
    WineFeatures,
)
from ml.predict import load_artifacts, predict_both, predict_grade, predict_score


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()  # warm the joblib cache once at startup, not per request
    yield


app = FastAPI(
    title="sommelier-api",
    description=(
        "Two-lens wine quality predictor on the UCI Wine Quality dataset. "
        "POST 11 physicochemical readings + wine_type to get a predicted quality "
        "score (regression) and a high/low grade (classification)."
    ),
    version="0.0.1",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health():
    _, _, _, metrics = load_artifacts()
    return {"status": "ok", "sklearn_version": metrics["sklearn_version"]}


@app.get("/features", tags=["meta"])
def features():
    """Input schema: feature order, the wine_type map, and valid ranges per feature."""
    _, _, schema, _ = load_artifacts()
    return schema


@app.get("/model/info", tags=["meta"])
def model_info():
    """Both models: params, real training metrics, top features, sklearn version."""
    _, _, _, metrics = load_artifacts()
    return metrics


@app.post("/predict/score", response_model=ScoreResponse, tags=["predict"])
def predict_score_endpoint(wine: WineFeatures):
    return ScoreResponse(quality=predict_score(wine.to_features()))


@app.post("/predict/grade", response_model=GradeResponse, tags=["predict"])
def predict_grade_endpoint(wine: WineFeatures):
    return GradeResponse(**predict_grade(wine.to_features()))


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict_endpoint(wine: WineFeatures):
    both = predict_both(wine.to_features())
    return PredictResponse(score=both["score"], grade=GradeResponse(**both["grade"]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
