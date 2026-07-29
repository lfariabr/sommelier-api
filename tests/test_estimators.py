"""Unit tests for the contract-driven classifier allowlist."""
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.estimators import build_classifier


def _assert_contract_params(classifier, expected: dict) -> None:
    actual = classifier.get_params()
    assert {name: actual[name] for name in expected} == expected


def test_builds_current_decision_tree_contract():
    params = {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "random_state": 42,
    }

    classifier = build_classifier(
        {"type": "DecisionTreeClassifier", "params": params}
    )

    assert isinstance(classifier, DecisionTreeClassifier)
    _assert_contract_params(classifier, params)


def test_builds_planned_random_forest_contract():
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
    }

    classifier = build_classifier(
        {"type": "RandomForestClassifier", "params": params}
    )

    assert isinstance(classifier, RandomForestClassifier)
    _assert_contract_params(classifier, params)


def test_rejects_classifier_outside_allowlist():
    with pytest.raises(ValueError, match="Unsupported classifier type 'SVC'"):
        build_classifier({"type": "SVC", "params": {}})


def test_rejects_non_mapping_params():
    with pytest.raises(TypeError, match="field 'params' must be a mapping"):
        build_classifier({"type": "DecisionTreeClassifier", "params": None})
