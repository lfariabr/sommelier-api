# 🍷 sommelier-api

> Two-lens wine quality predictor on the UCI Wine Quality dataset. The same 6,497
> wines, two questions: **how good is this wine?** (regression) and **is this wine
> good?** (classification) — served by a FastAPI backend and a Streamlit tasting-room UI
> over one shared, framework-agnostic ML core.

> ▶️ **Live:** **[Streamlit app](https://sommelier-api.streamlit.app/)** · **[FastAPI Swagger](https://sommelier-api-yd1m.onrender.com/docs)**
> — paste a wine's chemistry, get both verdicts. *(Render free tier sleeps when idle; first call may take ~50s.)*

## What it is

| Lens | Question | Model | Headline metric |
|---|---|---|---|
| **Score** (regression) | *How good, on a 0–10 scale?* | `RandomForestRegressor` | R² **0.50**, MAE **0.44** |
| **Grade** (classification) | *High (≥6) or low (<6)?* | `DecisionTreeClassifier` | accuracy **0.74**, ROC-AUC **0.81** |

Both read the same 12 features (11 physicochemical measurements + an engineered
`wine_type` flag) and are re-trained deterministically from the raw CSVs.

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
make api         # FastAPI at http://localhost:8000/docs
make ui          # Streamlit tasting-room at http://localhost:8501
```

## API surface

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | service status + scikit-learn version |
| `GET` | `/features` | input schema + valid range per feature |
| `GET` | `/model/info` | both models: params, real training metrics, top features |
| `POST` | `/predict/score` | `{ "quality": 5.8 }` |
| `POST` | `/predict/grade` | `{ "grade": "high", "proba_high": 0.73 }` |
| `POST` | `/predict` | both at once |

## Model card

- **Dataset:** UCI Wine Quality (Cortez et al., 2009) — 1,599 red + 4,898 white = 6,497 wines.
- **Features (12):** fixed/volatile acidity, citric acid, residual sugar, chlorides,
  free/total SO₂, density, pH, sulphates, alcohol, `wine_type` (red=1, white=0).
- **Score model:** `RandomForestRegressor(n_estimators=400, random_state=42)` → R² 0.50, MAE 0.44, RMSE 0.61.
- **Grade model:** `DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)`,
  threshold quality ≥ 6 → accuracy 0.74, ROC-AUC 0.81. Class 1 = low, class 0 = high (low is the minority class).
- **Honesty:** these models predict **human taste-panel scores**, not an objective truth.
  Wine quality is subjective and the R² ceiling on this dataset is genuinely low —
  the regressor explains about half the variance, hence the name.
- **Reproducibility:** `make train` is deterministic; metrics in `ml/artifacts/metrics.json`
  are the real re-trained numbers, surfaced live at `GET /model/info`.

## Provenance

The two models originate in the author's [Master of Software Engineering (AI)
coursework](https://github.com/lfariabr/masters-swe-ai) (MLN601 — regression +
classification). This repository is an independent **serving layer**: it re-implements
the pipeline cleanly from the public UCI CSVs in `ml/train.py` and contains **no
assessment notebooks, reports, or identifying data** — just the public dataset and a
fresh, deterministic training script.

## License

MIT — see [LICENSE](LICENSE).
