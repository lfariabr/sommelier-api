"""Reproduce the two MLN601 models from the raw UCI CSVs and serialize artifacts.

A1 (regression):     RandomForestRegressor → predicts `quality` (continuous).
A2 (classification): contract-selected classifier → high (>=6) / low (<6).

Deterministic (random_state=42). The models are RE-TRAINED here from the raw CSVs
— never loaded from the graded notebooks — so the serving scikit-learn version is
the training version and the joblib artifacts can't drift.

Run:  python -m ml.train
"""
from __future__ import annotations

import json
import math
import platform
import time

import joblib
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from ml import ARTIFACTS_DIR
from ml.contract import load_assessment_contract, validate_dataset_files
from ml.estimators import build_classifier
from ml.features import FEATURE_ORDER, build_feature_matrix, load_raw

CONTRACT = load_assessment_contract()
RANDOM_STATE = CONTRACT["split"]["random_state"]
QUALITY_THRESHOLD = CONTRACT["target"]["quality_threshold"]
LABELS = {
    CONTRACT["target"]["negative_class"]: CONTRACT["target"]["negative_label"],
    CONTRACT["target"]["positive_class"]: CONTRACT["target"]["positive_label"],
}


def _validate_a2_metrics(actual: dict) -> None:
    """Stop artifact generation unless the submitted A2 result is reproduced."""
    expected = CONTRACT["expected_test_metrics"]
    contract_version = CONTRACT["contract_version"]
    for metric_name, expected_value in expected.items():
        if metric_name == "confusion_matrix":
            if actual[metric_name] != expected_value:
                raise RuntimeError(
                    f"{contract_version} parity failure for confusion_matrix: "
                    f"expected {expected_value}, got {actual[metric_name]}"
                )
            continue
        if not math.isclose(
            actual[metric_name], expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"{contract_version} parity failure for {metric_name}: "
                f"expected {expected_value}, got {actual[metric_name]}"
            )


def main() -> None:
    validate_dataset_files()
    classifier_type = CONTRACT["estimator"]["type"]
    clf = build_classifier(CONTRACT["estimator"])
    if FEATURE_ORDER != CONTRACT["feature_order"]:
        raise RuntimeError(
            f"{CONTRACT['contract_version']} parity failure: "
            "the serving feature order differs from the "
            "submitted notebook"
        )

    df = load_raw()
    # Exact source-row duplicates (1,177 in the UCI files) otherwise land on both
    # sides of the split and inflate every metric — found auditing MLN601 A2.
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = raw_rows - len(df)
    expected_dataset = CONTRACT["dataset"]
    observed_counts = (raw_rows, duplicates_removed, len(df))
    expected_counts = (
        expected_dataset["raw_rows"],
        expected_dataset["duplicates_removed"],
        expected_dataset["unique_rows"],
    )
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"{CONTRACT['contract_version']} dataset parity failure: "
            f"expected {expected_counts}, "
            f"got {observed_counts}"
        )
    X = build_feature_matrix(df)
    n_rows = len(df)
    print(f"Loaded {raw_rows} wines, removed {duplicates_removed} exact duplicates "
          f"→ {n_rows} unique "
          f"({int((df.wine_type == 1).sum())} red / {int((df.wine_type == 0).sum())} white)")

    # ---- A1: regression (predict the score) ----
    y_reg = df["quality"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_reg, test_size=0.20, random_state=RANDOM_STATE
    )
    reg = RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    reg_metrics = {
        "r2": float(r2_score(yte, pred)),
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(root_mean_squared_error(yte, pred)),
    }
    print(f"A1 RandomForestRegressor   "
          f"R2={reg_metrics['r2']:.4f}  MAE={reg_metrics['mae']:.4f}  RMSE={reg_metrics['rmse']:.4f}")

    # ---- A2: classification (grade high/low) ----
    y_clf = (df["quality"] < QUALITY_THRESHOLD).astype(int)  # 1 = low, 0 = high
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X,
        y_clf,
        test_size=CONTRACT["split"]["test_size"],
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )
    # Estimator and imbalance treatment mirror the approved assessment contract.
    clf.fit(Xtr2, ytr2)
    pred2 = clf.predict(Xte2)
    proba_low = clf.predict_proba(Xte2)[:, 1]  # class 1 = low
    tn, fp, fn, tp = confusion_matrix(yte2, pred2, labels=[0, 1]).ravel()
    sensitivity_low = float(recall_score(yte2, pred2, pos_label=1))
    specificity_high = float(tn / (tn + fp))
    clf_metrics = {
        "accuracy": float(accuracy_score(yte2, pred2)),
        "precision_low": float(precision_score(yte2, pred2, pos_label=1)),
        "sensitivity_low": sensitivity_low,
        "specificity_high": specificity_high,
        "f1_low": float(f1_score(yte2, pred2, pos_label=1)),
        "balanced_accuracy": float(balanced_accuracy_score(yte2, pred2)),
        "g_mean": math.sqrt(sensitivity_low * specificity_high),
        "roc_auc": float(roc_auc_score(yte2, proba_low)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    _validate_a2_metrics(clf_metrics)
    print(f"A2 {classifier_type}  "
          f"ACC={clf_metrics['accuracy']:.4f}  ROC_AUC={clf_metrics['roc_auc']:.4f}  "
          f"SENS={clf_metrics['sensitivity_low']:.4f}  SPEC={clf_metrics['specificity_high']:.4f}")

    # ---- feature importances (top 3 each) ----
    reg_imp = sorted(zip(FEATURE_ORDER, reg.feature_importances_), key=lambda t: -t[1])
    clf_imp = sorted(zip(FEATURE_ORDER, clf.feature_importances_), key=lambda t: -t[1])

    # ---- schema (ranges + a real example row for Swagger / sliders) ----
    ranges = {f: {"min": float(df[f].min()), "max": float(df[f].max())} for f in FEATURE_ORDER}
    example = {
        f: (int(df.iloc[0][f]) if f == "wine_type" else float(df.iloc[0][f]))
        for f in FEATURE_ORDER
    }
    schema = {
        "feature_order": FEATURE_ORDER,
        "wine_type_map": {"red": 1, "white": 0},
        "ranges": ranges,
        "example": example,
        "target": {
            "regression": "quality (continuous, 0-10 scale)",
            "classification": {"threshold": QUALITY_THRESHOLD, "labels": LABELS},
        },
    }

    provenance = {
        "model_contract": CONTRACT["contract_version"],
        "assessment": CONTRACT["assessment"],
        "submission_version": CONTRACT["submission_version"],
        "source_repository": CONTRACT["source_repository"],
        "source_commit": CONTRACT["source_commit"],
        "submission_sha256": CONTRACT["submission_sha256"],
        "source_metrics_sha256": CONTRACT["source_metrics_sha256"],
        "dataset_sha256": CONTRACT["dataset"]["files"],
    }
    if "source_selection_sha256" in CONTRACT:
        provenance["source_selection_sha256"] = CONTRACT["source_selection_sha256"]

    metrics = {
        "provenance": provenance,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at_unix": int(time.time()),
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates_removed,
        "dataset_rows": n_rows,
        "random_state": RANDOM_STATE,
        "regression": {
            "model": "RandomForestRegressor",
            "params": {"n_estimators": 400, "max_depth": None, "random_state": RANDOM_STATE},
            **reg_metrics,
            "top_features": [[f, round(v, 4)] for f, v in reg_imp[:3]],
        },
        "classification": {
            "model": classifier_type,
            "params": CONTRACT["estimator"]["params"],
            "threshold": QUALITY_THRESHOLD,
            "labels": LABELS,
            **clf_metrics,
            "top_features": [[f, round(v, 4)] for f, v in clf_imp[:3]],
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, ARTIFACTS_DIR / "regressor.joblib", compress=3)
    joblib.dump(clf, ARTIFACTS_DIR / "classifier.joblib", compress=3)
    (ARTIFACTS_DIR / "schema.json").write_text(json.dumps(schema, indent=2))
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
