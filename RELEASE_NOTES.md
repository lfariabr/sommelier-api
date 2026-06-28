# Release Notes

## v0.0.1 — First public release (2026-06-29)

The first end-to-end cut of **sommelier-api**: two ML models trained on the UCI Wine
Quality dataset, served by a FastAPI backend and a Streamlit UI over one shared core —
deployed, tested, and documented.

### Models
- **Score (regression):** `RandomForestRegressor(n_estimators=400)` → predicted quality, **R² 0.50 / MAE 0.44 / RMSE 0.61**.
- **Grade (classification):** tuned `DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)` → high (≥6) / low (<6), **accuracy 0.74 / ROC-AUC 0.81**.
- Re-trained deterministically (`random_state=42`) from the public CSVs — **bit-identical to source**. Pinned scikit-learn 1.9.0, so the locally trained joblib is byte-for-byte the one serving in prod.

### ML core (`ml/`)
- `features.py` — single source of truth for the 12-feature contract + `wine_type` encoding (red=1 / white=0).
- `train.py` — reproduces both models, dumps joblib artifacts + `schema.json` + `metrics.json`.
- `predict.py` — framework-agnostic inference shared by both surfaces (incl. the A2 label-inversion guard: class 1 = low, 0 = high).

### FastAPI (`api/`)
- `GET /health`, `/features`, `/model/info`; `POST /predict/score`, `/predict/grade`, `/predict`.
- Pydantic v2 validation, auto Swagger docs, **real** metrics surfaced from training (no hard-coded numbers).
- Lifespan-loaded models. Deployed on Render.

### Streamlit (`ui/`)
- Tasting-room: sliders → quality gauge + high/low grade badge, in separate bordered lenses.
- **Local-default inference with automatic API fallback.** Deployed on Streamlit Community Cloud.

### Quality & ops
- 23 tests (feature contract, metric reproduction, label-inversion guard, API endpoints + 422s, UI fallback paths).
- GitHub Actions CI (ruff + pytest). `Makefile`, `Dockerfile`, `render.yaml`, split + pinned requirements.

### Live
- 🍷 App: https://sommelier-api.streamlit.app/
- 📜 API (Swagger): https://sommelier-api-yd1m.onrender.com/docs
- 📝 Write-up: https://dev.to/lfariaus/i-gave-the-same-6497-wines-to-two-models-and-asked-them-different-questions-4hdn
