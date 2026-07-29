"""Metric-reproduction + label-semantics tests over the committed artifacts."""
import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from ml import ARTIFACTS_DIR
from ml.contract import load_assessment_contract, validate_dataset_files
from ml.estimators import build_classifier
from ml.features import FEATURE_ORDER, build_feature_matrix, load_raw
from ml.predict import GRADE_LABELS, load_artifacts, predict_grade, predict_score

REGRESSION_TOL = 0.02
CONTRACT = load_assessment_contract()


def _metrics():
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text())


def test_regression_metrics_reproduce():
    m = _metrics()["regression"]
    assert abs(m["r2"] - 0.415) <= REGRESSION_TOL
    assert abs(m["mae"] - 0.510) <= REGRESSION_TOL
    assert abs(m["rmse"] - 0.663) <= REGRESSION_TOL


def test_classification_metrics_reproduce():
    # Must exactly match the submitted MLN601 A2 v7 balanced Decision Tree.
    m = _metrics()["classification"]
    expected = CONTRACT["expected_test_metrics"]
    assert m["model"] == CONTRACT["estimator"]["type"]
    for metric_name in (
        "accuracy",
        "roc_auc",
        "sensitivity_low",
        "specificity_high",
        "f1_low",
    ):
        assert m[metric_name] == pytest.approx(expected[metric_name], abs=1e-12)
    assert m["confusion_matrix"] == expected["confusion_matrix"]


def test_classification_passes_screening_gates():
    # The A2 v7 operational gates the served model was approved against.
    m = _metrics()["classification"]
    assert m["roc_auc"] >= 0.75
    assert m["sensitivity_low"] >= 0.70
    assert m["specificity_high"] >= 0.70


def test_metrics_metadata_present():
    m = _metrics()
    dataset = CONTRACT["dataset"]
    assert m["raw_rows"] == dataset["raw_rows"]
    assert m["duplicates_removed"] == dataset["duplicates_removed"]
    assert m["dataset_rows"] == dataset["unique_rows"]
    assert m["sklearn_version"]          # surfaced at /model/info
    assert m["random_state"] == CONTRACT["split"]["random_state"]
    cm = m["classification"]["confusion_matrix"]
    assert set(cm) == {"tn", "fp", "fn", "tp"}
    assert sum(cm.values()) == 1064      # the 20% held-out test split

    provenance = m["provenance"]
    assert provenance["model_contract"] == CONTRACT["contract_version"]
    assert provenance["source_commit"] == CONTRACT["source_commit"]
    assert provenance["submission_sha256"] == CONTRACT["submission_sha256"]
    assert provenance["source_metrics_sha256"] == CONTRACT["source_metrics_sha256"]
    assert provenance["dataset_sha256"] == CONTRACT["dataset"]["files"]


def test_raw_dataset_matches_submission_hashes():
    validate_dataset_files()


def test_served_classifier_exactly_matches_fresh_a2_v7_retrain():
    df = load_raw().drop_duplicates().reset_index(drop=True)
    X = build_feature_matrix(df)
    y = (df[CONTRACT["target"]["source"]] < CONTRACT["target"]["quality_threshold"]).astype(int)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=CONTRACT["split"]["test_size"],
        random_state=CONTRACT["split"]["random_state"],
        stratify=y,
    )
    fresh = build_classifier(CONTRACT["estimator"])
    fresh.fit(X_train, y_train)
    _, served, schema, _ = load_artifacts()

    assert schema["feature_order"] == CONTRACT["feature_order"] == FEATURE_ORDER
    np.testing.assert_array_equal(served.predict(X_test), fresh.predict(X_test))
    np.testing.assert_array_equal(
        served.predict_proba(X_test), fresh.predict_proba(X_test)
    )
    for attribute in (
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "value",
        "n_node_samples",
        "weighted_n_node_samples",
    ):
        np.testing.assert_array_equal(
            getattr(served.tree_, attribute), getattr(fresh.tree_, attribute)
        )


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
