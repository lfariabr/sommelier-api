"""Main page — adjust a wine's chemistry, get both verdicts."""
from __future__ import annotations

import streamlit as st

from ui.components.inputs import wine_input_form
from ui.components.results import render_results
from ui.services.prediction_service import PredictionService


def render() -> None:
    st.title("🍷 Sommelier")
    st.write(
        "Two ML lenses on the same wine: a predicted **quality score** (regression) "
        "and a **high/low grade** (classification). Adjust the chemistry below."
    )
    feats = wine_input_form()
    if st.button("Taste it", type="primary", use_container_width=True):
        result = PredictionService.predict(tuple(sorted(feats.items())))
        render_results(result)
