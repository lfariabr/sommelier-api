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
| **Score** (regression) | *How good, on a 0–10 scale?* | `RandomForestRegressor` | R² **0.41**, MAE **0.51** |
| **Grade** (classification) | *High (≥6) or low (<6)?* | balanced `DecisionTreeClassifier` | ROC-AUC **0.79**, sensitivity **0.73** |

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
make parity      # exact MLN601 A2 v7 dataset/model/API parity checks
make api         # FastAPI at http://localhost:8000/docs
make ui          # Streamlit tasting-room at http://localhost:8501
```

## API surface

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | status + scikit-learn version + A2 model contract/source commit |
| `GET` | `/features` | input schema + valid range per feature |
| `GET` | `/model/info` | both models: params, real training metrics, top features |
| `POST` | `/predict/score` | `{ "quality": 5.8 }` |
| `POST` | `/predict/grade` | `{ "grade": "high", "proba_high": 0.73 }` |
| `POST` | `/predict` | both at once |

## Model card

- **Dataset:** UCI Wine Quality (Cortez et al., 2009) — 1,599 red + 4,898 white = 6,497
  raw rows; 1,177 exact duplicates removed before the split → **5,320 unique wines**.
- **Features (12):** fixed/volatile acidity, citric acid, residual sugar, chlorides,
  free/total SO₂, density, pH, sulphates, alcohol, `wine_type` (red=1, white=0).
- **Score model:** `RandomForestRegressor(n_estimators=400, random_state=42)` → R² 0.41, MAE 0.51, RMSE 0.66.
- **Grade model:** `DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced",
  random_state=42)`, threshold quality ≥ 6 → ROC-AUC 0.79, sensitivity 0.73, specificity 0.73,
  accuracy 0.73. Class 1 = low, class 0 = high (low is the minority class). The class weighting
  is deliberate: a missed low-quality wine costs more than a false alarm, so the model trades
  some precision for catching 73% of genuinely low wines instead of 59%.
- **Honesty:** these models predict **human taste-panel scores**, not an objective truth.
  Wine quality is subjective and the performance ceiling on this dataset is genuinely low.
- **Reproducibility:** `make train` is deterministic; metrics in `ml/artifacts/metrics.json`
  are the real re-trained numbers, surfaced live at `GET /model/info`.

### v2: the leakage audit

The v1 release reported R² 0.50 and ROC-AUC 0.81. Auditing the same pipeline for the
follow-up university assessment surfaced the problem: the raw UCI files contain **1,177
exact duplicate rows**, and with a random split identical wines land on both sides of
the train/test boundary — the model is graded on rows it has already seen. Deduplicating
before the split drops the metrics to their honest values (R² 0.41, ROC-AUC 0.79).
Nothing about the models improved or regressed; the *evaluation* was corrected. The
lower numbers are the real ones, and this section exists because publishing them
matters more than keeping the prettier v1 headline.

## Provenance

The two models originate in the author's [Master of Software Engineering (AI)
coursework](https://github.com/lfariabr/masters-swe-ai) (MLN601 — regression +
classification). The served classifier is locked to the submitted **Assessment 2 v7**
notebook at source commit `029b4b14c52b4b19ea111bca6f43a3e75e180e0f`.
`ml/assessment_contract.json` records the notebook, source-metrics and raw-dataset
SHA-256 hashes, feature order, target semantics, split, estimator parameters, exact
held-out metrics and confusion matrix.

This repository remains an independent **serving layer**: it contains no assessment
notebooks, reports, or identifying data. `make train` rebuilds from the public CSVs and
refuses to write artifacts if the A2 v7 contract is not reproduced; `make parity`
additionally compares the committed classifier's predictions, probabilities and tree
structure with a fresh retrain. The notebook's SHAP analysis documents model behaviour,
but per-lot SHAP explanations are intentionally not part of the current API contract.

## License

MIT — see [LICENSE](LICENSE).
