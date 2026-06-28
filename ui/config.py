"""UI constants — single source of truth for the Streamlit app (mirrors 1stMillion/config.py).

Slider ranges, the example wine, and feature order are read from the committed
ml/artifacts/schema.json so the UI never hard-codes anything the model doesn't agree with.
"""
from __future__ import annotations

import json
import os

from ml import ARTIFACTS_DIR

APP_TITLE = "🍷 Sommelier"
CACHE_TTL = 3600  # predictions are deterministic — cache aggressively

_SCHEMA = json.loads((ARTIFACTS_DIR / "schema.json").read_text())
FEATURE_ORDER = _SCHEMA["feature_order"]
FEATURE_RANGES = _SCHEMA["ranges"]
EXAMPLE = _SCHEMA["example"]
QUALITY_THRESHOLD = _SCHEMA["target"]["classification"]["threshold"]


def _secret(key: str, default: str) -> str:
    """Read from st.secrets, then env, then default — tolerant of a missing secrets file."""
    try:
        import streamlit as st

        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


def get_inference_mode() -> str:
    """'local' (default) runs joblib in-process; 'api' calls the deployed FastAPI."""
    return _secret("INFERENCE_MODE", "local")


def get_api_url() -> str:
    return _secret("API_URL", "http://localhost:8000")
