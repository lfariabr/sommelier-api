"""Reproduce the two MLN601 models from the raw UCI CSVs and serialize artifacts.

A1 (regression):     RandomForestRegressor → predicts `quality` (continuous).
A2 (classification): DecisionTreeClassifier → high (>=6) / low (<6).

Deterministic (random_state=42). The models are RE-TRAINED here from the raw CSVs
— never loaded from the graded notebooks — so the serving scikit-learn version is
the training version and the joblib artifacts can't drift.

Run:  python -m ml.train
"""
from __future__ import annotations

import json
import platform
import time

import joblib
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml import ARTIFACTS_DIR
from ml.features import FEATURE_ORDER, build_feature_matrix, load_raw

RANDOM_STATE = 42
QUALITY_THRESHOLD = 6  # quality >= 6 → high (class 0); < 6 → low (class 1)
LABELS = {0: "high (>=6)", 1: "low (<6)"}


def main() -> None:
    df = load_raw()
    X = build_feature_matrix(df)
    n_rows = len(df)
    print(f"Loaded {n_rows} wines "
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
        X, y_clf, test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf
    )
    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=6, min_samples_leaf=20,
        class_weight=None, random_state=RANDOM_STATE,
    )
    clf.fit(Xtr2, ytr2)
    proba_low = clf.predict_proba(Xte2)[:, 1]  # class 1 = low
    clf_metrics = {
        "accuracy": float(accuracy_score(yte2, clf.predict(Xte2))),
        "roc_auc": float(roc_auc_score(yte2, proba_low)),
    }
    print(f"A2 DecisionTreeClassifier  "
          f"ACC={clf_metrics['accuracy']:.4f}  ROC_AUC={clf_metrics['roc_auc']:.4f}")

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

    metrics = {
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at_unix": int(time.time()),
        "dataset_rows": n_rows,
        "random_state": RANDOM_STATE,
        "regression": {
            "model": "RandomForestRegressor",
            "params": {"n_estimators": 400, "max_depth": None, "random_state": RANDOM_STATE},
            **reg_metrics,
            "top_features": [[f, round(v, 4)] for f, v in reg_imp[:3]],
        },
        "classification": {
            "model": "DecisionTreeClassifier",
            "params": {
                "criterion": "gini", "max_depth": 6, "min_samples_leaf": 20,
                "class_weight": None, "random_state": RANDOM_STATE,
            },
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
