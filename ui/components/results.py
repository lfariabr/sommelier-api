"""Pure-display result widgets — two clearly separated lenses:
a quality-score gauge (regression) and a high/low grade badge (classification)."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_results(result: dict) -> None:
    score = result["score"]
    grade = result["grade"]
    is_high = grade["grade"] == "high"

    st.subheader("Verdict")
    left, right = st.columns(2, gap="large")

    # ---- Lens 1: the score (regression) ----
    with left:
        with st.container(border=True):
            st.markdown("#### 🎯 Quality score")
            st.caption("Regression · predicted on a 0–10 scale")
            st.plotly_chart(_gauge(score), width="stretch")
            st.metric("Predicted score", f"{score:.2f} / 10")

    # ---- Lens 2: the grade (classification) ----
    with right:
        with st.container(border=True):
            st.markdown("#### 🏷️ Grade")
            st.caption("Classification · threshold: quality ≥ 6")
            if is_high:
                st.success("## 🟢 HIGH")
            else:
                st.error("## 🔴 LOW")
            st.progress(
                float(grade["proba_high"]),
                text=f"P(high quality) = {grade['proba_high']:.0%}",
            )
            st.metric("Confidence", f"{max(grade['proba_high'], grade['proba_low']):.0%}")

    st.caption(f"served via: {result.get('served_via', 'local model')}")


def _gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score, 2),
            number={"font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#7b2d3b"},
            },
        )
    )
    fig.update_layout(height=240, margin=dict(t=20, b=10, l=20, r=20))
    return fig
