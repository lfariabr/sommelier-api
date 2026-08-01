# 🍷 sommelier-api

> Two-lens wine quality predictor on the UCI Wine Quality dataset. The same wines
> (5,320 unique after a duplicate-leakage audit), two questions: **how good is this
> wine?** (regression) and **is this wine good?** (classification) — served by a FastAPI
> backend and a Streamlit tasting-room UI over one shared, framework-agnostic ML core.

> ▶️ **Live:** **[Streamlit app](https://sommelier-api.streamlit.app/)** · **[FastAPI Swagger](https://sommelier-api-yd1m.onrender.com/docs)**
> — paste a wine's chemistry, get both verdicts. *(Render free tier sleeps when idle; first call may take ~50s.)*

## What it is

| Lens | Question | Model | Headline metric |
|---|---|---|---|
| **Score** (regression) | *How good, on a 0–10 scale?* | A1-derived `RandomForestRegressor` | R² **0.41**, MAE **0.51** |
| **Grade** (classification) | *High (≥6) or low (<6)?* | A2-exact class-weighted `RandomForestClassifier` | ROC-AUC **0.834**, sensitivity **0.714** |

Both read the same 12 features (11 physicochemical measurements + an engineered
`wine_type` flag) and are re-trained deterministically from the raw CSVs, after
removing 1,177 exact duplicate rows (see *v2: the leakage audit* below).

## Architecture

```
            ┌──────────────────────────────┐
            │  ml/  (framework-agnostic)    │
            │  features · train · predict   │
            │  + joblib artifacts           │
            └───────────────┬──────────────┘
                ┌───────────┴───────────┐
        ┌───────▼───────┐       ┌───────▼────────┐
        │  api/ FastAPI │       │  ui/ Streamlit │
        │  Swagger /docs│◄──────│  local | api   │
        └───────────────┘  http │  + fallback    │
                                └────────────────┘
```

`ml/` knows nothing about FastAPI or Streamlit — both surfaces are thin adapters over
it, importing the **same** `build_features()` / `predict_*()` so predictions can never
drift between training and serving. The Streamlit UI runs **local in-process inference
by default** (rock-solid public demo) and can toggle to call the live API, falling back
to local automatically if the API is cold.

## Quickstart

```bash
make install     # venv (Python 3.11.9) + pinned deps
make train       # reproduce both models → ml/artifacts/
make test        # pytest (feature contract, metric reproduction, API, UI fallback)
make parity      # A1 regression lineage + A2 v8 classifier parity checks
make api         # FastAPI at http://localhost:8000/docs
make ui          # Streamlit tasting-room at http://localhost:8501
```

## API surface

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | status + scikit-learn version + both model contract versions |
| `GET` | `/features` | input schema + valid range per feature |
| `GET` | `/model/info` | both models: params, real training metrics, top features |
| `POST` | `/predict/score` | `{ "quality": 5.8 }` |
| `POST` | `/predict/grade` | `{ "grade": "low", "label": 1, "proba_high": 0.1681, "proba_low": 0.8319 }` |
| `POST` | `/predict` | both at once |

## Model card

- **Dataset:** UCI Wine Quality (Cortez et al., 2009) — 1,599 red + 4,898 white = 6,497
  raw rows; 1,177 exact duplicates removed before the split → **5,320 unique wines**.
- **Features (12):** fixed/volatile acidity, citric acid, residual sugar, chlorides,
  free/total SO₂, density, pH, sulphates, alcohol, `wine_type` (red=1, white=0).
- **Score model:** `RandomForestRegressor(n_estimators=400, random_state=42)` → R² 0.41,
  MAE 0.51, RMSE 0.66. It is an A1-derived production retrain on deduplicated data,
  not the submitted A1 artifact.
- **Grade model:** `RandomForestClassifier(n_estimators=200, max_depth=10,
  min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=1)`, threshold
  quality ≥ 6. Held-out results: ROC-AUC 0.8337, accuracy 0.7716, sensitivity 0.7136,
  specificity 0.8063 and F1 0.7004. Class 1 = low and class 0 = high.
- **Selection:** A2 v8 compared 22 model-and-treatment runs across nine estimators, plus
  one majority baseline. Original-distribution and SMOTE runs covered all nine estimators;
  class weighting covered the four estimators that support it. The approved forest cleared
  every predeclared screening gate.
- **Operating trade-off:** against the previously served v7 tree on the same 1,064-row test
  set, v8 gains 8.1 percentage points of specificity and loses 2.0 points of sensitivity.
  That means 8 additional low-quality lots missed out of 398, with 54 fewer false alarms
  out of 666. Class weighting shifts eligible models toward catching low-quality wines;
  that treatment rationale is separate from the v7-to-v8 model comparison.
- **Honesty:** these models predict **human taste-panel scores**, not an objective truth.
  Wine quality is subjective and the performance ceiling on this dataset is genuinely low.
- **Reproducibility:** A2 classification is submission-exact. A1 regression reproduces both
  the submitted 6,497-row final-estimator result and the corrected 5,320-row serving
  adaptation, without claiming they are the same evaluation. Fresh retrains use bounded
  cross-platform tolerances. The real artifact metrics and model-specific provenance are
  surfaced live at `GET /model/info`.

### v2: the leakage audit

The v1 release reported R² 0.50 and ROC-AUC 0.81. Auditing the same pipeline for the
follow-up university assessment surfaced the problem: the raw UCI files contain **1,177
exact duplicate rows**, and with a random split identical wines land on both sides of
the train/test boundary — the model is graded on rows it has already seen. Deduplicating
before the split produced the honest v0.1.0 values (R² 0.41, ROC-AUC 0.79 for its
then-served tree). Nothing about those models improved or regressed; the *evaluation*
was corrected. A2 v8 subsequently changed the classifier and raised its ROC-AUC to
0.8337 on the same deduplicated split.

## Provenance

The two models originate in the author's [Master of Software Engineering (AI)
coursework](https://github.com/lfariabr/masters-swe-ai) (MLN601 regression and
classification), but they provide different guarantees:

| Lens | Contract | Relationship to assessment |
|---|---|---|
| Regression | `mln601-a1-derived-v1` | Reproduces the submitted protocol and separately locks the deduplicated serving adaptation |
| Classification | `mln601-a2-v8` | Reproduces the submitted A2 v8 estimator, metrics and confusion matrix exactly |

The A1 lineage contract points to source commit
`93b39df59185126c5a40ae6e395a4cdc8d1d50aa`, submission SHA-256
`4db8def424459265b9283eb5d20b0f529a75aa6af3ab4f2530c47d876e46640a` and metrics
SHA-256 `358f9e6b009a08e9f5eeb4294a36b7338a42648ff0fe29d8c5ae2a176d8bcca2`.
The submitted protocol used 6,497 rows and reported R² 0.5002, MAE 0.4364 and RMSE
0.6075. Production removes 1,177 exact duplicates before splitting and reports R²
0.4146, MAE 0.5096 and RMSE 0.6634. A1 was not resubmitted under that adaptation.

The A2 contract points to source commit
`c5be26cf1bb7cc71f8f057fba45aa5b3ea8dd5b2` and records submission, metrics and
selection hashes alongside data identity, feature order, target semantics, split,
estimator parameters and exact held-out evidence.

This repository remains an independent **serving layer**: it contains no assessment
notebooks, reports, or identifying data. `make train` rebuilds from the public CSVs and
refuses to write artifacts unless both contracts reproduce their declared serving
evidence in the canonical environment. `make parity` also reconstructs the submitted A1
final-estimator protocol and checks fresh cross-platform retrains using measured bounds.
The A2 notebook's SHAP analysis documents model behaviour, but per-lot SHAP explanations
and probability calibration are intentionally outside the current API contract.

## License

MIT — see [LICENSE](LICENSE).
