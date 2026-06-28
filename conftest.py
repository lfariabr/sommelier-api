"""Make the repo root importable so `ml`, `api`, and `ui` resolve under pytest
regardless of the directory pytest is invoked from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
