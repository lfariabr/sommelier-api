"""Reviewed classifier constructors allowed by the assessment contract."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

Classifier = DecisionTreeClassifier | RandomForestClassifier

_CLASSIFIER_TYPES: dict[str, type[Classifier]] = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
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
