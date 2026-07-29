"""Metric-reproduction + label-semantics tests over the committed artifacts."""
import json
import math

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml import ARTIFACTS_DIR
from ml.contract import load_assessment_contract, validate_dataset_files
from ml.estimators import build_classifier
from ml.features import FEATURE_ORDER, build_feature_matrix, load_raw
from ml.predict import GRADE_LABELS, load_artifacts, predict_grade, predict_score

CONTRACT = load_assessment_contract()
EXPECTED_REGRESSION_METRICS = {
    "r2": 0.41459084153850323,
    "mae": 0.5096052631578948,
    "rmse": 0.6634399807689515,
}
# macOS arm64 and Linux x64 selected one neighboring RF split differently in CI.
# Keep the published artifact exact while bounding fresh-retrain platform variance.
CROSS_PLATFORM_METRIC_TOLERANCE = 0.005
MAX_CROSS_PLATFORM_LABEL_DISAGREEMENTS = 2


def _metrics():
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text())


def _classification_split():
    df = load_raw().drop_duplicates().reset_index(drop=True)
    X = build_feature_matrix(df)
    y = (
        df[CONTRACT["target"]["source"]]
        < CONTRACT["target"]["quality_threshold"]
    ).astype(int)
    return train_test_split(
        X,
        y,
        test_size=CONTRACT["split"]["test_size"],
        random_state=CONTRACT["split"]["random_state"],
        stratify=y,
    )


def _classification_metrics(y_true, predictions, proba_low):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    sensitivity_low = float(recall_score(y_true, predictions, pos_label=1))
    specificity_high = float(tn / (tn + fp))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_low": float(precision_score(y_true, predictions, pos_label=1)),
        "sensitivity_low": sensitivity_low,
        "specificity_high": specificity_high,
        "f1_low": float(f1_score(y_true, predictions, pos_label=1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "g_mean": math.sqrt(sensitivity_low * specificity_high),
        "roc_auc": float(roc_auc_score(y_true, proba_low)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def test_regression_metrics_reproduce():
    m = _metrics()["regression"]
    for metric_name, expected_value in EXPECTED_REGRESSION_METRICS.items():
        assert m[metric_name] == pytest.approx(expected_value, abs=1e-12)


def test_classification_metrics_reproduce():
    # Must exactly match the submitted MLN601 A2 v8 balanced Random Forest.
    m = _metrics()["classification"]
    expected = CONTRACT["expected_test_metrics"]
    assert m["model"] == CONTRACT["estimator"]["type"]
    for metric_name, expected_value in expected.items():
        if metric_name == "confusion_matrix":
            assert m[metric_name] == expected_value
        else:
            assert m[metric_name] == pytest.approx(expected_value, abs=1e-12)


def test_classification_passes_screening_gates():
    # The A2 v8 operational gates the served model was approved against.
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
    assert provenance["source_selection_sha256"] == CONTRACT["source_selection_sha256"]
    assert provenance["dataset_sha256"] == CONTRACT["dataset"]["files"]


def test_raw_dataset_matches_submission_hashes():
    validate_dataset_files()


def test_served_classifier_exactly_reproduces_a2_v8_contract():
    _, X_test, _, y_test = _classification_split()
    _, served, schema, _ = load_artifacts()

    assert isinstance(served, RandomForestClassifier)
    served_params = served.get_params()
    assert {
        name: served_params[name] for name in CONTRACT["estimator"]["params"]
    } == CONTRACT["estimator"]["params"]
    assert schema["feature_order"] == CONTRACT["feature_order"] == FEATURE_ORDER
    assert len(served.estimators_) == CONTRACT["estimator"]["params"]["n_estimators"]

    predictions = served.predict(X_test)
    actual = _classification_metrics(
        y_test, predictions, served.predict_proba(X_test)[:, 1]
    )
    expected = CONTRACT["expected_test_metrics"]
    assert actual["confusion_matrix"] == expected["confusion_matrix"]
    for metric_name, expected_value in expected.items():
        if metric_name != "confusion_matrix":
            assert actual[metric_name] == pytest.approx(expected_value, abs=1e-12)


def test_fresh_a2_v8_retrain_is_equivalent_across_platforms():
    X_train, X_test, y_train, y_test = _classification_split()
    fresh = build_classifier(CONTRACT["estimator"])
    fresh.fit(X_train, y_train)
    _, served, _, _ = load_artifacts()

    assert isinstance(fresh, RandomForestClassifier)
    assert len(fresh.estimators_) == len(served.estimators_) == 200
    np.testing.assert_array_equal(
        [tree.random_state for tree in fresh.estimators_],
        [tree.random_state for tree in served.estimators_],
    )

    fresh_predictions = fresh.predict(X_test)
    served_predictions = served.predict(X_test)
    disagreements = int(np.count_nonzero(fresh_predictions != served_predictions))
    assert disagreements <= MAX_CROSS_PLATFORM_LABEL_DISAGREEMENTS

    fresh_metrics = _classification_metrics(
        y_test, fresh_predictions, fresh.predict_proba(X_test)[:, 1]
    )
    expected = CONTRACT["expected_test_metrics"]
    for metric_name, expected_value in expected.items():
        if metric_name != "confusion_matrix":
            assert fresh_metrics[metric_name] == pytest.approx(
                expected_value, abs=CROSS_PLATFORM_METRIC_TOLERANCE
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
