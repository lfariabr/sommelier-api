"""Prediction service — mirrors 1stMillion's GoogleSheetsService (classmethods + cache).

Sources predictions either from the in-process joblib models ('local') or the deployed
FastAPI ('api'). In 'api' mode a cold/unreachable backend falls back to local so the
public demo never visibly breaks. The compute() logic is separated from the cached
entrypoint so it can be unit-tested without a Streamlit runtime.
"""
from __future__ import annotations

import httpx
import streamlit as st

from ml.predict import predict_both
from ui.config import CACHE_TTL, get_api_url, get_inference_mode


class PredictionService:
    @classmethod
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def predict(_cls, features: tuple) -> dict:
        """Cached entrypoint. `features` is a hashable sorted tuple of (name, value)."""
        return _cls.compute(dict(features), get_inference_mode(), get_api_url())

    @staticmethod
    def compute(feat_dict: dict, mode: str, api_url: str) -> dict:
        if mode == "api":
            try:
                result = PredictionService._via_api(feat_dict, api_url)
                result["served_via"] = "live API"
                return result
            except Exception:
                result = predict_both(feat_dict)
                result["served_via"] = "local model (API cold)"
                return result
        result = predict_both(feat_dict)
        result["served_via"] = "local model"
        return result

    @staticmethod
    def _via_api(feat_dict: dict, api_url: str) -> dict:
        # spaced feature names -> snake_case API payload
        payload = {k.replace(" ", "_"): v for k, v in feat_dict.items()}
        resp = httpx.post(f"{api_url}/predict", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
