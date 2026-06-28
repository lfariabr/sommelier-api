"""About page — the two-lenses narrative + provenance."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("ℹ️ About Sommelier")
    st.markdown(
        """
**Sommelier** asks the same 6,497 wines two different questions:

- **How good is this wine?** → a `RandomForestRegressor` predicts the quality score
  on a 0–10 scale (R² ≈ 0.50).
- **Is this wine good?** → a `DecisionTreeClassifier` grades it **high (≥6)** or
  **low (<6)** (ROC-AUC ≈ 0.81).

Both read the same 12 inputs: 11 physicochemical measurements (acidity, sugar,
sulphates, alcohol, …) plus a `wine_type` flag (red/white).

### How it's built
A framework-agnostic `ml/` core trains and serves both models. A **FastAPI** service
exposes them over a typed REST API (with Swagger docs), and this **Streamlit** app is a
thin client over the same core — running predictions locally by default, or calling the
live API with automatic fallback.

### Data & honesty
Dataset: **UCI Wine Quality** (Cortez et al., 2009). These models learn *human
taste-panel* scores — a subjective target with a real performance ceiling. The point
isn't a perfect oracle; it's a clean, reproducible, deployed two-lens ML service.

Code: [github.com/lfariabr/sommelier-api](https://github.com/lfariabr/sommelier-api)
        """
    )
