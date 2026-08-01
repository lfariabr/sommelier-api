"""Model-card copy stays tied to the committed A2 v8 metrics."""
from ml.predict import load_artifacts
from ui.model_card import (
    class_weight_rationale,
    classification_summary,
    classifier_caption,
    regression_caption,
    regression_lineage_note,
    regression_lineage_table,
    v8_vs_v7_tradeoff,
)


def _classification() -> dict:
    return load_artifacts()[3]["classification"]


def _regression() -> dict:
    return load_artifacts()[3]["regression"]


def test_regression_copy_declares_derived_lineage():
    regression = _regression()

    assert regression_caption(regression) == (
        "RandomForestRegressor | A1-derived production retrain | "
        "contract mln601-a1-derived-v1"
    )
    assert regression_lineage_table(regression) == [
        {
            "Protocol": "A1 submitted",
            "Rows": "6,497",
            "R²": "0.500",
            "MAE": "0.436",
            "RMSE": "0.608",
        },
        {
            "Protocol": "Production serving",
            "Rows": "5,320",
            "R²": "0.415",
            "MAE": "0.510",
            "RMSE": "0.663",
        },
    ]


def test_regression_lineage_note_explains_the_metric_change():
    note = regression_lineage_note(_regression())

    assert "1,177 exact duplicates" in note
    assert "preventing duplicate leakage" in note
    assert "not submission-exact" in note


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
