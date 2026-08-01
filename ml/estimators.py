"""Reviewed estimator constructors allowed by the model contracts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

Classifier = DecisionTreeClassifier | RandomForestClassifier
Regressor = RandomForestRegressor

_CLASSIFIER_TYPES: dict[str, type[Classifier]] = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

_REGRESSOR_TYPES: dict[str, type[Regressor]] = {
    "RandomForestRegressor": RandomForestRegressor,
}


def build_classifier(estimator_config: Mapping[str, Any]) -> Classifier:
    """Build an explicitly supported classifier from its contract entry."""
    estimator_type = estimator_config.get("type")
    classifier_type = _CLASSIFIER_TYPES.get(estimator_type)
    if classifier_type is None:
        supported = ", ".join(sorted(_CLASSIFIER_TYPES))
        raise ValueError(
            f"Unsupported classifier type {estimator_type!r}. Supported types: {supported}"
        )

    params = estimator_config.get("params")
    if not isinstance(params, Mapping):
        raise TypeError("Classifier contract field 'params' must be a mapping")

    return classifier_type(**dict(params))


def build_regressor(estimator_config: Mapping[str, Any]) -> Regressor:
    """Build an explicitly supported regressor from a contract entry."""
    estimator_type = estimator_config.get("type")
    regressor_type = _REGRESSOR_TYPES.get(estimator_type)
    if regressor_type is None:
        supported = ", ".join(sorted(_REGRESSOR_TYPES))
        raise ValueError(
            f"Unsupported regressor type {estimator_type!r}. Supported types: "
            f"{supported}"
        )

    params = estimator_config.get("params")
    if not isinstance(params, Mapping):
        raise TypeError("Regressor contract field 'params' must be a mapping")

    return regressor_type(**dict(params))
