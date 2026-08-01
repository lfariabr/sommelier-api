# Deploy + launch checklist

The repo is public and CI-green. Streamlit and Render are already live; this checklist
documents their configuration and the verification required for the v0.2.1 release.

## 1. Streamlit Community Cloud (the public app)
1. share.streamlit.io → **New app** → repo `lfariabr/sommelier-api`, branch `master`,
   main file `ui/app.py`.
2. **Advanced settings → Python 3.11** (must match `.python-version` / the pinned wheels).
3. Deploy. It installs from the root `requirements.txt` and runs **local inference**, so
   it works with no backend. URL: `https://sommelier-api.streamlit.app`.

## 2. Render (the FastAPI service)
1. render.com → **New → Blueprint** → connect `lfariabr/sommelier-api` (reads `render.yaml`).
2. It builds with `requirements-api.txt`, starts uvicorn, health-checks `/health`.
3. Smoke test: `curl https://sommelier-api-yd1m.onrender.com/health` returns status `ok`
   and contracts `mln601-a1-derived-v1` plus `mln601-a2-v8`; open the live `/docs` and
   use "Try it out" on `/predict`.

## 3. Wire the UI to the live API (optional, shows integration)
On Streamlit Cloud → app **Settings → Secrets**, add:
```
INFERENCE_MODE = "api"
API_URL = "https://sommelier-api-yd1m.onrender.com"
```
The badge under a prediction flips to "live API"; if Render is cold it shows
"local model (API cold)" and still works.

## 4. Add the row to the masters README
Once the app + article URLs exist, add this row to the Projects table in
`masters_SWEAI/README.md` (github.com/lfariabr/masters-swe-ai):

```markdown
| **Sommelier API** | Two-lens wine quality service with an A1-derived, deduplicated Random Forest regressor (R² 0.41) and an A2 v8 submission-exact, class-weighted Random Forest classifier (ROC-AUC 0.834, sensitivity 0.714, specificity 0.806). Both contracts are exposed through FastAPI and Streamlit. | ✅ | [Repo](https://github.com/lfariabr/sommelier-api) | [App](https://sommelier-api.streamlit.app/) / [Article](https://dev.to/lfariaus/i-gave-the-same-6497-wines-to-two-models-and-asked-them-different-questions-4hdn) |
```

## 5. Publish the article
`docs/article.md` is the maintained source for the original published dev.to article. The
v0.2.1 sequel is tracked in `lfariabr/luisfaria.dev#265` and must link the final release.

## Smoke test (after each deploy)
- [ ] `/health` reports `mln601-a1-derived-v1` and `mln601-a2-v8` while preserving the legacy A2 fields.
- [ ] `/model/info` reports assessment-derived regression and submission-exact classification provenance.
- [ ] `/openapi.json` reports API version `0.2.1`.
- [ ] `/predict/grade` returns low, label 1, `proba_high=0.1681` and `proba_low=0.8319` for the example wine.
- [ ] `/predict/score` still returns `5.065` for the example wine within the existing tolerance.
- [ ] Streamlit cold-loads, predicts in local mode and shows the A1 submitted-versus-serving comparison.
- [ ] API mode matches local inference; an unavailable API still falls back locally.
