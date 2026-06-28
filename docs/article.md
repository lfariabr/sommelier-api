# I gave the same 6,497 wines to two models and asked them different questions

> Draft for dev.to / luisfaria.dev. Live links get filled in after deploy.

Most ML tutorials stop at a notebook with a green `R²` cell and a shrug. I wanted to
go one step further: take two models I'd actually trained, and turn them into something
you can *poke at* — a typed API and a little web app anyone can open.

So I built **[sommelier-api](https://github.com/lfariabr/sommelier-api)**: one dataset,
two questions, two surfaces.

## Two lenses on the same wine

The [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
(Cortez et al., 2009) has 6,497 wines — 1,599 red, 4,898 white — each with 11
physicochemical measurements (acidity, residual sugar, sulphates, alcohol…) and a
quality score from 0 to 10 assigned by human tasters.

You can ask that data two different questions:

1. **How good is this wine?** — a regression problem. Predict the score.
2. **Is this wine good?** — a classification problem. High (≥6) or low (<6)?

Same features, two lenses. I trained one model for each:

- a `RandomForestRegressor` for the score, and
- a tuned `DecisionTreeClassifier` for the grade.

## The modelling (and one honest number)

Both models share the exact same 12 inputs — the 11 chemical readings plus an
engineered `wine_type` flag (red=1, white=0) so a single model can see both colours.
Because both are tree-based, there's **no feature scaling at inference** — one of the
small things that makes serving them clean.

Here's the part most posts skip: **the regressor's R² is about 0.50.** It explains
roughly half the variance in the scores. That's not a bug to hide — it's the nature of
the problem. Wine quality is a *subjective human judgement*; there's a real ceiling on
how well chemistry alone predicts a tasting panel. The classifier does better on its
easier yes/no question — **ROC-AUC ≈ 0.81** — but the honest framing matters more than
a vanity metric. (It's also where the project gets its name: it can bottle about half
the lab; the other half is human.)

What the models *do* agree on is what matters most: **alcohol** and **volatile
acidity** dominate both — high alcohol and low volatile acidity track with better wine.

## Turning models into a service

The interesting engineering isn't the `.fit()` call — it's everything around it. The
repo is built around a **framework-agnostic core** that knows nothing about web
frameworks:

```
ml/
  features.py   # build_features() + FEATURE_ORDER — the single source of truth
  train.py      # deterministic re-train from the raw CSVs -> joblib artifacts
  predict.py    # load_artifacts(), predict_score(), predict_grade()
```

Everything else is a thin adapter over `ml/`:

- **FastAPI** exposes the models over a typed REST API with auto-generated Swagger
  docs. Pydantic validates every input (and returns a clean `422` when your wine has
  negative alcohol). `GET /model/info` returns the *real* metrics straight from the
  training run — no hard-coded numbers.

```python
@app.post("/predict")
def predict_endpoint(wine: WineFeatures):
    both = predict_both(wine.to_features())
    return PredictResponse(score=both["score"], grade=GradeResponse(**both["grade"]))
```

- **Streamlit** is the friendly face: drag some sliders, hit *Taste it*, watch a gauge
  and a grade badge update. It runs the models **in-process by default** (so the public
  demo never depends on a sleeping backend), but it can flip to calling the live API —
  and if that API is cold, it *falls back to local* automatically and tells you so.

One discipline ties it together: the training scikit-learn version is **pinned**, the
artifacts are committed, and that same version is surfaced at `/model/info`. The joblib
I trained on my laptop is bit-for-bit the joblib that serves in production. No
"works-on-my-machine" drift.

## Try it

- 🍷 **App:** `<streamlit-link>` — paste a wine's chemistry, get both verdicts.
- 📜 **API docs:** `<render-link>/docs` — the same models over REST.
- 💻 **Code:** [github.com/lfariabr/sommelier-api](https://github.com/lfariabr/sommelier-api)

## What's next

A few things I'd add: a gradient-boosting comparator for the score, SHAP explanations so
the app can say *why* a wine scored low, and probability calibration on the classifier.
But the point of v1 wasn't the perfect model — it was the full path from a notebook to a
deployed, typed, tested service. The half you can bottle.
