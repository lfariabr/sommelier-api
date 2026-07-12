"""Model card page — real training metrics, straight from the artifacts."""
from __future__ import annotations

import streamlit as st

from ml.predict import load_artifacts


def render() -> None:
    st.title("📊 Model card")
    _, _, _, metrics = load_artifacts()
    st.caption(
        f"scikit-learn {metrics['sklearn_version']} · "
        f"{metrics['raw_rows']:,} raw wines − {metrics['duplicates_removed']:,} exact "
        f"duplicates = {metrics['dataset_rows']:,} unique · random_state {metrics['random_state']}"
    )

    reg, clf = metrics["regression"], metrics["classification"]
    left, right = st.columns(2)
    with left:
        st.subheader("Score · regression")
        st.caption(reg["model"])
        st.metric("R²", f"{reg['r2']:.3f}")
        st.metric("MAE", f"{reg['mae']:.3f}")
        st.metric("RMSE", f"{reg['rmse']:.3f}")
        st.write("**Top features**")
        for name, imp in reg["top_features"]:
            st.write(f"- {name} ({imp:.2f})")
    with right:
        st.subheader("Grade · classification")
        st.caption(f"{clf['model']} · threshold quality ≥ {clf['threshold']} · class-weight balanced")
        st.metric("ROC-AUC", f"{clf['roc_auc']:.3f}")
        st.metric("Sensitivity (low caught)", f"{clf['sensitivity_low']:.3f}")
        st.metric("Specificity (high cleared)", f"{clf['specificity_high']:.3f}")
        st.metric("Accuracy", f"{clf['accuracy']:.3f}")
        st.write("**Top features**")
        for name, imp in clf["top_features"]:
            st.write(f"- {name} ({imp:.2f})")

    st.info(
        "These models predict **human taste-panel scores**, not an objective truth. "
        "Wine quality is subjective and the R² ceiling on this dataset is genuinely low. "
        "The grade model is class-weight balanced on purpose: it trades some false alarms "
        "for catching more genuinely low wines — the same operating choice approved in the "
        "source assessment."
    )
