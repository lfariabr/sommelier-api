"""sommelier-api ML core — framework-agnostic.

This package knows nothing about FastAPI or Streamlit. Both surfaces are thin
adapters over it. All artifact paths are anchored to ROOT (derived from __file__),
never the current working directory, so inference works identically under pytest,
Render (uvicorn), and Streamlit Cloud.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
