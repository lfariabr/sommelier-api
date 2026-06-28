"""Model card page — real training metrics, straight from the artifacts."""
from __future__ import annotations

import streamlit as st

from ml.predict import load_artifacts


def render() -> None:
    st.title("📊 Model card")
    _, _, _, metrics = load_artifacts()
    st.caption(
        f"scikit-learn {metrics['sklearn_version']} · "
        f"{metrics['dataset_rows']:,} wines · random_state {metrics['random_state']}"
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
        st.caption(f"{clf['model']} · threshold quality ≥ {clf['threshold']}")
        st.metric("Accuracy", f"{clf['accuracy']:.3f}")
        st.metric("ROC-AUC", f"{clf['roc_auc']:.3f}")
        st.write("**Top features**")
        for name, imp in clf["top_features"]:
            st.write(f"- {name} ({imp:.2f})")

    st.info(
        "These models predict **human taste-panel scores**, not an objective truth. "
        "Wine quality is subjective and the R² ceiling on this dataset is genuinely low "
        "— the regressor explains about half the variance. Hence the name."
    )
