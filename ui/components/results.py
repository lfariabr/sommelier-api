"""Pure-display result widgets — quality gauge + high/low grade badge."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def render_results(result: dict) -> None:
    score = result["score"]
    grade = result["grade"]
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_gauge(score), use_container_width=True)
    with right:
        is_high = grade["grade"] == "high"
        st.metric("Grade", f"{'🟢' if is_high else '🔴'} {grade['grade'].upper()}")
        st.progress(
            float(grade["proba_high"]),
            text=f"P(high quality) = {grade['proba_high']:.0%}",
        )
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
            title={"text": "Predicted quality (0–10)"},
        )
    )
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
    return fig
