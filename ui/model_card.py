"""Pure presentation helpers for the classification model card."""
from __future__ import annotations

V7_CONFUSION_MATRIX = {"tn": 483, "fp": 183, "fn": 106, "tp": 292}


def classifier_caption(classification: dict) -> str:
    """Describe the served classifier from artifact metadata."""
    params = classification["params"]
    return (
        f"{classification['model']} | threshold quality >= "
        f"{classification['threshold']} | class_weight={params['class_weight']}"
    )


def classification_summary(classification: dict) -> str:
    """Return the concise current-state summary used by the About view."""
    return (
        f"a class-weighted `{classification['model']}` grades it "
        f"**high (>={classification['threshold']})** or "
        f"**low (<{classification['threshold']})** "
        f"(ROC-AUC {classification['roc_auc']:.3f}, "
        f"sensitivity {classification['sensitivity_low']:.3f}, "
        f"specificity {classification['specificity_high']:.3f}, "
        f"accuracy {classification['accuracy']:.3f})"
    )


def class_weight_rationale() -> str:
    """Explain the v8 imbalance treatment without comparing it with v7."""
    return (
        "Within the v8 model matrix, class weighting was an imbalance treatment for "
        "eligible estimators. It shifts the operating point toward catching more "
        "low-quality wines without creating synthetic rows. This treatment rationale "
        "is separate from the comparison with the previously served v7 tree."
    )


def v8_vs_v7_tradeoff(classification: dict) -> str:
    """Quantify the held-out operating trade-off against the served v7 tree."""
    current = classification["confusion_matrix"]
    low_total = current["fn"] + current["tp"]
    high_total = current["tn"] + current["fp"]
    v7_sensitivity = V7_CONFUSION_MATRIX["tp"] / low_total
    v7_specificity = V7_CONFUSION_MATRIX["tn"] / high_total
    sensitivity_delta = classification["sensitivity_low"] - v7_sensitivity
    specificity_delta = classification["specificity_high"] - v7_specificity
    additional_misses = current["fn"] - V7_CONFUSION_MATRIX["fn"]
    fewer_false_alarms = V7_CONFUSION_MATRIX["fp"] - current["fp"]

    return (
        "Compared with the served v7 Decision Tree on the same held-out test, "
        f"specificity increased by {specificity_delta * 100:.1f} percentage points "
        f"while sensitivity decreased by {abs(sensitivity_delta) * 100:.1f} percentage "
        f"points. In lot-screening terms, the v8 forest missed {additional_misses} "
        f"additional low-quality lots out of {low_total}, while raising "
        f"{fewer_false_alarms} fewer false alarms out of {high_total}."
    )
