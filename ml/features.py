"""Single source of truth for the wine feature contract.

Imported by train, the FastAPI service, and the Streamlit UI so the encoding and
column order can never drift between training and serving.
"""
from __future__ import annotations

import pandas as pd

from ml import DATA_DIR

# Exact column order the models were trained on. DO NOT REORDER — the joblib
# artifacts carry these names; a reorder silently corrupts predictions.
FEATURE_ORDER = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]
CHEMICAL_FEATURES = FEATURE_ORDER[:-1]   # the 11 physicochemical inputs
WINE_TYPE_MAP = {"red": 1, "white": 0}   # engineered flag (red=1, white=0)


def load_raw(data_dir=DATA_DIR) -> pd.DataFrame:
    """Load the red + white UCI CSVs (sep=';'), tag wine_type, concat → 6,497 rows."""
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    red["wine_type"] = WINE_TYPE_MAP["red"]
    white["wine_type"] = WINE_TYPE_MAP["white"]
    return pd.concat([red, white], ignore_index=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return X in the canonical 12-feature order."""
    return df[FEATURE_ORDER].copy()


def single_row(features: dict) -> pd.DataFrame:
    """Build a 1-row DataFrame in canonical order from a {feature_name: value} dict.

    `wine_type` may be given as "red"/"white" or as 1/0. Raises KeyError if any
    feature is missing.
    """
    feats = dict(features)
    wt = feats.get("wine_type")
    if isinstance(wt, str):
        feats["wine_type"] = WINE_TYPE_MAP[wt.lower()]
    missing = [f for f in FEATURE_ORDER if f not in feats]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    return pd.DataFrame([[feats[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
