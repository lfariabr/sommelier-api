# Release Notes

## v0.2.1: Dual-model provenance (2026-08-02)

This release closes the A1 regression provenance gap without changing either model's
predictions or any public prediction request/response schema. A2 classification remains
submission-exact; A1 regression is now explicitly assessment-derived and lineage-locked
to both its submitted evidence and corrected production adaptation.

### A1 regression lineage
- Added contract `mln601-a1-derived-v1`, tied to canonical A1 source commit
  `93b39df59185126c5a40ae6e395a4cdc8d1d50aa`.
- Pinned submission SHA-256
  `4db8def424459265b9283eb5d20b0f529a75aa6af3ab4f2530c47d876e46640a`
  and metrics SHA-256
  `358f9e6b009a08e9f5eeb4294a36b7338a42648ff0fe29d8c5ae2a176d8bcca2`.
- Reproduced the submitted 6,497-row final-estimator metrics: R² 0.5002, MAE 0.4364
  and RMSE 0.6075.
- Separately locked the production adaptation: remove 1,177 exact duplicates before
  splitting, model 5,320 rows, and reproduce R² 0.4146, MAE 0.5096 and RMSE 0.6634.
- Regressor construction, parameters, split, row policy, expected metrics and golden
  `5.065` prediction now come from the contract instead of test literals.

### Public provenance
- `/health` preserves the existing A2 `model_contract` and `source_commit` fields, then
  adds `model_contracts` for regression and classification.
- `/model/info` exposes `assessment_derived` regression provenance and
  `submission_exact` classification provenance independently.
- The Streamlit Model Card visibly compares A1 submitted and production rows and metrics,
  and explains that deduplication prevents leakage rather than representing a resubmission.
- FastAPI/OpenAPI metadata now reports version `0.2.1`.

### Compatibility and verification
- `regressor.joblib`, `classifier.joblib` and `schema.json` remained byte-identical.
- Score remains `5.065` for the public example wine; grade labels and probabilities are
  unchanged from v0.2.0.
- Local verification: 51 tests, 30 focused parity tests and a clean Ruff run.
- The v0.2.0 statement that A1 was "unchanged" referred to serving artifact stability.
  It did not mean the deduplicated production metrics were identical to the A1 submission.

## v0.2.0: A2 v8 approved Random Forest (2026-07-30)

The classification lens now serves the model approved by the MLN601 Assessment 2 v8
resubmission. Classification predictions and probabilities intentionally change; API
request and response schemas remain compatible, and the A1 regression lens is unchanged.

### Changed
- Replaced the v7 balanced Decision Tree with
  `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=1,
  class_weight="balanced", random_state=42, n_jobs=1)`.
- The source assessment compared 22 model-and-treatment runs across nine estimators,
  plus one majority baseline. Original-distribution and fold-only SMOTE runs covered all
  nine estimators; class weighting covered four eligible estimators.
- Held-out classification metrics are now ROC-AUC 0.8337, accuracy 0.7716, sensitivity
  0.7136, specificity 0.8063 and F1 0.7004. Against the served v7 tree, specificity gains
  8.1 percentage points while sensitivity loses 2.0 points.
- In lot-screening terms, the new operating point misses 8 additional low-quality lots
  out of 398 and raises 54 fewer false alarms out of 666 on the same 1,064-row test set.
- FastAPI metadata now reports version `0.2.0`. Every endpoint, request field, response
  field, validation type and class-label meaning remains unchanged.
- The Streamlit About and Model Card views now read the v8 estimator and metrics from the
  committed artifact, separate class-weighting rationale from the v7-to-v8 comparison,
  and disclose the sensitivity reduction visibly.
- Local Streamlit and FastAPI inference are parity-tested for the example wine. Numeric
  `wine_type` values from artifact examples are translated to the public API's
  `"red"/"white"` schema instead of causing a silent fallback after a `422` response.

### Provenance and parity
- Contract: `mln601-a2-v8`.
- Assessment source commit: `c5be26cf1bb7cc71f8f057fba45aa5b3ea8dd5b2`.
- Submission SHA-256:
  `b4aeca9b6ed0412d5855f1fa46a3afd4bc95173a8b71bdf3963fb728331dddd9`.
- Held-out metrics SHA-256:
  `e0860e301e5d416047de4451cdaefcb0818e6e5d057aad649f91fab482181d20`.
- Selection summary SHA-256:
  `a011aceb8fe8c9aaf686854483c3193dd9a731e3197bd957a6f99baa30a9043f`.
- The committed serving artifact reproduces every submitted metric and confusion-matrix
  count exactly. Fresh cross-platform retrains retain parameters and estimator seeds,
  with measured bounds of at most two label differences, 0.005 per aggregate metric and
  0.015 per predicted probability between the tested macOS arm64 and Linux x64 runs.

### Unchanged
- A1 `RandomForestRegressor` artifact, metrics and golden score prediction (`5.065`).
- Public paths and methods for `/health`, `/features`, `/model/info`, `/predict/score`,
  `/predict/grade` and `/predict`.
- Class 1 remains low quality (`quality < 6`); class 0 remains high quality
  (`quality >= 6`).
- Per-lot SHAP explanations and probability calibration remain outside the current API.

## v0.1.1 — A2 v7 parity lock (2026-07-20)

The classification serving path is now explicitly and mechanically tied to the final
MLN601 Assessment 2 v7 submission, without changing its predictions.

### Changed
- Added a checked-in parity contract containing source commit and artifact hashes,
  raw-dataset hashes, feature order, target encoding, split, tree parameters, exact
  held-out metrics and confusion matrix.
- Training now fails before writing artifacts when dataset identity, row counts,
  feature order, metrics or confusion matrix diverge from A2 v7.
- Added exact parity tests for the committed classifier versus a fresh deterministic
  retrain, including predictions, probabilities and internal tree arrays.
- `/health` and `/model/info` now expose additive provenance fields identifying the
  `mln601-a2-v7` model contract and source submission commit.
- Added `make parity`; existing prediction request and response contracts are unchanged.

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
