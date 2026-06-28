"""Framework-agnostic inference core. Shared by the FastAPI service and the Streamlit UI.

Label semantics (A2 classifier) — read carefully, the encoding is intentionally inverted:
    class 1 = LOW   (quality < 6)   ← positive / minority class
    class 0 = HIGH  (quality >= 6)
This matches ml/train.py. A test asserts a known high-quality wine maps to grade "high"
so the inversion can never silently flip a badge in the UI.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib

from ml import ARTIFACTS_DIR
from ml.features import single_row

# class label -> human-facing grade
GRADE_LABELS = {0: "high", 1: "low"}


@lru_cache(maxsize=1)
def load_artifacts():
    """Load (regressor, classifier, schema, metrics) once, cached for the process."""
    reg = joblib.load(ARTIFACTS_DIR / "regressor.joblib")
    clf = joblib.load(ARTIFACTS_DIR / "classifier.joblib")
    schema = json.loads((ARTIFACTS_DIR / "schema.json").read_text())
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    return reg, clf, schema, metrics


def predict_score(features: dict) -> float:
    """A1 regression → predicted quality (continuous)."""
    reg, _, _, _ = load_artifacts()
    return float(reg.predict(single_row(features))[0])


def predict_grade(features: dict) -> dict:
    """A2 classification → {grade, label, proba_high, proba_low}."""
    _, clf, _, _ = load_artifacts()
    X = single_row(features)
    label = int(clf.predict(X)[0])
    proba_low = float(clf.predict_proba(X)[0][1])  # class 1 = low
    return {
        "grade": GRADE_LABELS[label],
        "label": label,
        "proba_high": round(1.0 - proba_low, 4),
        "proba_low": round(proba_low, 4),
    }


def predict_both(features: dict) -> dict:
    """Both lenses in one call (what the API's /predict and the UI use)."""
    return {"score": predict_score(features), "grade": predict_grade(features)}
