"""Pure-display input widgets — render sliders from the schema, return a feature dict."""
from __future__ import annotations

import streamlit as st

from ml.features import CHEMICAL_FEATURES
from ui.config import EXAMPLE, FEATURE_RANGES


def wine_input_form() -> dict:
    """Render the 11 chemical sliders + a wine-type toggle. Returns spaced-key feature dict."""
    wine_type = st.radio("Wine type", ["red", "white"], horizontal=True)
    cols = st.columns(2)
    feats: dict = {}
    for i, feat in enumerate(CHEMICAL_FEATURES):
        rng = FEATURE_RANGES[feat]
        lo, hi = float(rng["min"]), float(rng["max"])
        with cols[i % 2]:
            feats[feat] = st.slider(
                feat.title(),
                min_value=lo,
                max_value=hi,
                value=float(EXAMPLE[feat]),
            )
    feats["wine_type"] = wine_type
    return feats
