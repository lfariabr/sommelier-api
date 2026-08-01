"""Executable lineage checks for the A1-derived regression lens."""
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from ml.contract import (
    load_assessment_contract,
    load_regression_contract,
    validate_contract_compatibility,
)
from ml.features import FEATURE_ORDER, build_feature_matrix, load_raw

REGRESSION_CONTRACT = load_regression_contract()
CROSS_PLATFORM_METRIC_TOLERANCE = 0.005


def _reproduce(protocol: dict) -> tuple[int, dict[str, float]]:
    frame = load_raw()
    if protocol["row_policy"] == "drop_exact_duplicates_before_split":
        frame = frame.drop_duplicates().reset_index(drop=True)
    elif protocol["row_policy"] != "keep_exact_duplicates":
        raise AssertionError(f"Unknown row policy: {protocol['row_policy']}")

    X = build_feature_matrix(frame)
    y = frame["quality"]
    split = protocol["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split["test_size"],
        random_state=split["random_state"],
    )
    estimator = protocol["estimator"]
    assert estimator["type"] == "RandomForestRegressor"
    model = RandomForestRegressor(**estimator["params"]).fit(X_train, y_train)
    predictions = model.predict(X_test)
    return len(frame), {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
    }


def test_regression_contract_identifies_canonical_a1_evidence():
    assert REGRESSION_CONTRACT["contract_version"] == "mln601-a1-derived-v1"
    assert REGRESSION_CONTRACT["relationship"] == "assessment_derived"
    assert REGRESSION_CONTRACT["source_commit"] == (
        "93b39df59185126c5a40ae6e395a4cdc8d1d50aa"
    )
    assert REGRESSION_CONTRACT["submission_sha256"] == (
        "4db8def424459265b9283eb5d20b0f529a75aa6af3ab4f2530c47d876e46640a"
    )
    assert REGRESSION_CONTRACT["source_metrics_sha256"] == (
        "358f9e6b009a08e9f5eeb4294a36b7338a42648ff0fe29d8c5ae2a176d8bcca2"
    )


def test_regression_and_classification_contracts_share_inputs():
    validate_contract_compatibility()
    classification = load_assessment_contract()
    assert REGRESSION_CONTRACT["dataset"]["files"] == classification["dataset"][
        "files"
    ]
    assert REGRESSION_CONTRACT["feature_order"] == FEATURE_ORDER


def test_submitted_selection_matches_frozen_estimator():
    selected = REGRESSION_CONTRACT["selection"]["selected_params"]
    submitted = REGRESSION_CONTRACT["submitted_protocol"]["estimator"]["params"]
    assert {name: submitted[name] for name in selected} == selected


@pytest.mark.parametrize("protocol_name", ["submitted_protocol", "serving_adaptation"])
def test_regression_protocol_reproduces_declared_metrics(protocol_name):
    protocol = REGRESSION_CONTRACT[protocol_name]
    actual_rows, actual_metrics = _reproduce(protocol)

    assert actual_rows == protocol["model_rows"]
    for metric_name, expected_value in protocol["expected_test_metrics"].items():
        assert actual_metrics[metric_name] == pytest.approx(
            expected_value, abs=CROSS_PLATFORM_METRIC_TOLERANCE
        )


def test_serving_adaptation_is_explicitly_not_submission_parity():
    submitted = REGRESSION_CONTRACT["submitted_protocol"]
    serving = REGRESSION_CONTRACT["serving_adaptation"]

    assert submitted["model_rows"] == 6497
    assert serving["model_rows"] == 5320
    assert serving["duplicates_removed"] == 1177
    assert submitted["expected_test_metrics"] != serving["expected_test_metrics"]
