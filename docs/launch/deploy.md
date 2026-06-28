# Deploy + launch checklist

The repo is public and CI-green. These are the remaining steps to take it fully live.
Each needs Luis's own accounts (Streamlit Cloud, Render, dev.to) — the code is ready.

## 1. Streamlit Community Cloud (the public app)
1. share.streamlit.io → **New app** → repo `lfariabr/sommelier-api`, branch `master`,
   main file `ui/app.py`.
2. **Advanced settings → Python 3.11** (must match `.python-version` / the pinned wheels).
3. Deploy. It installs from the root `requirements.txt` and runs **local inference**, so
   it works with no backend. URL: `https://sommelier-api.streamlit.app` (or similar).

## 2. Render (the FastAPI service)
1. render.com → **New → Blueprint** → connect `lfariabr/sommelier-api` (reads `render.yaml`).
2. It builds with `requirements-api.txt`, starts uvicorn, health-checks `/health`.
3. Smoke test: `curl https://<render-url>/health` → `{"status":"ok",...}`;
   open `https://<render-url>/docs` and "Try it out" on `/predict`.

## 3. Wire the UI to the live API (optional, shows integration)
On Streamlit Cloud → app **Settings → Secrets**, add:
```
INFERENCE_MODE = "api"
API_URL = "https://<render-url>"
```
The badge under a prediction flips to "live API"; if Render is cold it shows
"local model (API cold)" and still works.

## 4. Add the row to the masters README
Once the app + article URLs exist, add this row to the Projects table in
`masters_SWEAI/README.md` (github.com/lfariabr/masters-swe-ai):

```markdown
| **Sommelier API** | Two-lens wine quality predictor on the UCI Wine Quality dataset (6,497 reds + whites). FastAPI service serving a RandomForest regressor (R² 0.50) and a DecisionTree classifier (ROC-AUC 0.81), with a Streamlit tasting-room UI and a local/API inference toggle. | ✅ | [Repo](https://github.com/lfariabr/sommelier-api) | [App](https://sommelier-api.streamlit.app/) / [Article](REPLACE-AFTER-PUBLISH) |
```

## 5. Publish the article
`docs/article.md` → dev.to / luisfaria.dev. Fill in the `<streamlit-link>` and
`<render-link>` placeholders first, then backfill the article URL into the README row.

## Smoke test (after each deploy)
- [ ] `curl <render>/health` ok; `/model/info` metrics match local (prod artifact == local).
- [ ] `<streamlit>` cold-loads in seconds and predicts in local mode.
- [ ] Flip to api-mode → first call wakes Render, then returns; kill API → fallback still works.
- [ ] `/docs` "Try it out" on the example returns a prediction.
