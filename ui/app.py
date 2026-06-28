"""Streamlit entry point. Run: streamlit run ui/app.py  (or `make ui`).

Inserts the repo root on sys.path so `ml` and `ui` resolve whether launched locally
or by Streamlit Community Cloud.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from ui.views import about, model_info, taste  # noqa: E402

st.set_page_config(page_title="Sommelier", page_icon="🍷", layout="centered")

nav = st.navigation(
    {
        "Sommelier": [
            st.Page(taste.render, title="Taste", icon="🍷", url_path="taste", default=True),
            st.Page(model_info.render, title="Model card", icon="📊", url_path="model-card"),
            st.Page(about.render, title="About", icon="ℹ️", url_path="about"),
        ]
    }
)
nav.run()
