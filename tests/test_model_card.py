"""Model-card copy stays tied to the committed A2 v8 metrics."""
from ml.predict import load_artifacts
from ui.model_card import (
    class_weight_rationale,
    classification_summary,
    classifier_caption,
    v8_vs_v7_tradeoff,
)


def _classification() -> dict:
    return load_artifacts()[3]["classification"]


def test_current_model_copy_comes_from_artifact_metadata():
    classification = _classification()

    assert classifier_caption(classification) == (
        "RandomForestClassifier | threshold quality >= 6 | class_weight=balanced"
    )
    assert classification_summary(classification) == (
        "a class-weighted `RandomForestClassifier` grades it **high (>=6)** or "
        "**low (<6)** (ROC-AUC 0.834, sensitivity 0.714, specificity 0.806, "
        "accuracy 0.772)"
    )


def test_class_weight_rationale_is_not_the_v7_comparison():
    copy = class_weight_rationale()

    assert "without creating synthetic rows" in copy
    assert "separate from the comparison" in copy


def test_v8_tradeoff_discloses_both_sides_in_lot_terms():
    copy = v8_vs_v7_tradeoff(_classification())

    assert "specificity increased by 8.1 percentage points" in copy
    assert "sensitivity decreased by 2.0 percentage points" in copy
    assert "8 additional low-quality lots out of 398" in copy
    assert "54 fewer false alarms out of 666" in copy
