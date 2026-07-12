# Release Notes

## v0.1.0 — The leakage audit (2026-07-13)

Honest-metrics release. Auditing the pipeline for MLN601 Assessment 2 surfaced 1,177
exact duplicate rows in the raw UCI files crossing the train/test split and inflating
every published v0.0.1 metric.

### Changed
- **Dedup before split** in `ml/train.py`: 6,497 raw rows → 5,320 unique. Provenance
  (`raw_rows`, `duplicates_removed`) recorded in `metrics.json` and served at `/model/info`.
- **Grade model is now the assessment-approved balanced tree:**
  `DecisionTreeClassifier(gini, max_depth=5, min_samples_leaf=20, class_weight="balanced")`.
  It trades some false alarms for catching 73% of genuinely low wines (was 59% at the
  default weighting).
- **Honest re-trained metrics:** regression R² 0.41 / MAE 0.51 / RMSE 0.66 (was 0.50 /
  0.44 / 0.61); classification ROC-AUC 0.79 / sensitivity 0.73 / specificity 0.73 /
  accuracy 0.73 (was accuracy 0.74 / ROC-AUC 0.81). The models did not get worse —
  the evaluation got corrected.
- `metrics.json` and `/model/info` now include sensitivity, specificity, F1 and the
  full confusion matrix; the Streamlit model card and About page lead with them.
- Tests re-pinned to the deduplicated numbers plus a new gate test mirroring the
  assessment approval criteria (AUC ≥ 0.75, sensitivity ≥ 0.70, specificity ≥ 0.70).

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
