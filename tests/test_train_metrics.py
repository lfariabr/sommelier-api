"""Metric-reproduction + label-semantics tests over the committed artifacts."""
import json

from ml import ARTIFACTS_DIR
from ml.features import load_raw
from ml.predict import GRADE_LABELS, load_artifacts, predict_grade, predict_score

TOL = 0.02  # tolerance vs the source notebook's reported metrics


def _metrics():
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text())


def test_regression_metrics_reproduce():
    m = _metrics()["regression"]
    assert abs(m["r2"] - 0.500) <= TOL
    assert abs(m["mae"] - 0.436) <= TOL
    assert abs(m["rmse"] - 0.608) <= TOL


def test_classification_metrics_reproduce():
    m = _metrics()["classification"]
    assert abs(m["accuracy"] - 0.737) <= TOL
    assert abs(m["roc_auc"] - 0.809) <= TOL


def test_metrics_metadata_present():
    m = _metrics()
    assert m["dataset_rows"] == 6497
    assert m["sklearn_version"]          # surfaced at /model/info
    assert m["random_state"] == 42


def test_artifacts_load_and_schema_shape():
    reg, clf, schema, metrics = load_artifacts()
    assert schema["feature_order"][-1] == "wine_type"
    assert schema["wine_type_map"] == {"red": 1, "white": 0}
    assert "example" in schema


def test_label_map_is_low_one_high_zero():
    # The intentional inversion: class 1 = low, class 0 = high.
    assert GRADE_LABELS == {0: "high", 1: "low"}


def test_grade_mapping_not_inverted():
    """Clearly high-quality wines (quality >= 7) must mostly map to grade 'high'.
    If the label encoding were flipped, this collapses to 'low' and fails."""
    df = load_raw()
    high = df[df["quality"] >= 7].head(300)
    grades = [predict_grade(row.to_dict())["grade"] for _, row in high.iterrows()]
    frac_high = sum(g == "high" for g in grades) / len(grades)
    assert frac_high > 0.6, f"only {frac_high:.0%} of high-quality wines graded high — label inverted?"


def test_score_returns_plausible_quality():
    score = predict_score(load_raw().iloc[0].to_dict())
    assert 0 <= score <= 10
