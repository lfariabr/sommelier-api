"""Feature-contract tests — the single source of truth must not drift."""
import pytest

from ml.features import (
    CHEMICAL_FEATURES,
    FEATURE_ORDER,
    build_feature_matrix,
    load_raw,
    single_row,
)


def test_feature_order_is_twelve_with_wine_type_last():
    assert len(FEATURE_ORDER) == 12
    assert FEATURE_ORDER[-1] == "wine_type"
    assert len(CHEMICAL_FEATURES) == 11
    assert "wine_type" not in CHEMICAL_FEATURES


def test_load_raw_row_count_and_encoding():
    df = load_raw()
    assert len(df) == 6497
    assert set(df["wine_type"].unique()) == {0, 1}
    assert int((df["wine_type"] == 1).sum()) == 1599   # red
    assert int((df["wine_type"] == 0).sum()) == 4898   # white


def test_build_feature_matrix_keeps_order_drops_quality():
    X = build_feature_matrix(load_raw())
    assert list(X.columns) == FEATURE_ORDER
    assert "quality" not in X.columns


def test_single_row_accepts_string_or_int_wine_type():
    feats = {f: 0.0 for f in FEATURE_ORDER}
    feats["wine_type"] = "red"
    row = single_row(feats)
    assert list(row.columns) == FEATURE_ORDER
    assert int(row["wine_type"].iloc[0]) == 1


def test_single_row_missing_feature_raises():
    feats = {f: 0.0 for f in FEATURE_ORDER if f != "alcohol"}
    with pytest.raises(KeyError):
        single_row(feats)
