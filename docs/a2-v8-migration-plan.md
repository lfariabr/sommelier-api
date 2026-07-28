# Migration plan: serve the MLN601 A2 v8 approved classifier

Status: proposed, not started.
Target release: `v0.2.0` (the served classification model changes).

## Why

The classification lens is parity-locked to the **A2 v7** submission, which approved a
class-weight-balanced `DecisionTreeClassifier`. The A2 v8 resubmission replaced the
single-model selection with a 22-row model matrix (9 estimators x 3 imbalance treatments)
and approved a different model:

```
# outputs/selection_summary_v8.csv
best untreated,SVM,Original
best SMOTE,SVM,SMOTE
best class weighted,Random Forest,Class weight
best ensemble,Random Forest,Class weight
approved,Random Forest,Class weight
```

The repository therefore serves a model the source assessment no longer approves, while
`/health` and `/model/info` assert `mln601-a2-v7` provenance. That is the defect: the
provenance claim is accurate about v7 and stale about the assessment.

## Verified facts, measured before planning

Everything below was measured, not assumed.

**The v8 model reproduces exactly in this serving environment.** Trained
`RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=1,
class_weight="balanced", random_state=42, n_jobs=1)` inside `venv` (scikit-learn 1.9.0,
Python 3.11.9) on the same dedup-then-split pipeline `ml/train.py` already uses. Result
matches `outputs/finalist_test_metrics_v8.csv` to full float precision, confusion matrix
included. No version negotiation is needed.

| Test-set metric (n=1064) | v7 tree (served today) | v8 RF (approved) | Delta |
|---|---|---|---|
| ROC-AUC | 0.7923 | **0.8337** | +0.0414 |
| Accuracy | 0.7284 | **0.7716** | +0.0432 |
| Specificity (high) | 0.7252 | **0.8063** | +0.0811 |
| F1 (low) | 0.6690 | **0.7004** | +0.0314 |
| Sensitivity (low) | **0.7337** | 0.7136 | -0.0201 |
| Confusion (tn/fp/fn/tp) | 483/183/106/292 | 537/129/114/284 | |

**Sensitivity goes down, and that changes the public story.** The current README sells the
class weighting as "catching 73% of genuinely low wines instead of 59%". Under v8 it is
71%. Still above the 0.70 screening gate, but the honest framing is different: the Random
Forest wins by buying +8.1pp of specificity for -2.0pp of sensitivity. In lot-screening
terms, it lets 8 more bad lots through out of 398 and raises 54 fewer false alarms out of
666. That trade has to be stated, not glossed.

**Dataset identity is unchanged.** Both raw UCI hashes in the v7 contract still match the
files in the source assessment. No dataset migration, no re-audit of the 1,177 duplicates.

**Source provenance resolved.** The submitted v8 notebook lives at
`2026-T2/MLN/assignments/Assessment2/submission/MLN601FariaLuisBrief2.ipynb` in the
masters repo, committed at `c5be26cf1bb7cc71f8f057fba45aa5b3ea8dd5b2` (2026-07-21). The
committed blob matches the on-disk file:
`b4aeca9b6ed0412d5855f1fa46a3afd4bc95173a8b71bdf3963fb728331dddd9`. Working tree clean.

**Artifact size is acceptable.** The v8 classifier serializes to **2.2 MB** with
`compress=3`, against 3.4 KB for the tree. The repo already ships a 10.6 MB regressor, so
this is a 20% increase in artifact weight, not a new class of problem. No Git LFS needed.

**`api/main.py` and `ml/predict.py` need no changes.** Both are estimator-agnostic; they
read `predict` / `predict_proba` and echo `metrics.json`. The request and response
schemas of every endpoint stay byte-identical. This is a model swap, not an API change.

## Decisions needed before starting

1. **Does the sensitivity drop change the product position?** The v8 model is better on
   four metrics and worse on the one the README argues matters most. Options: (a) adopt
   v8 and rewrite the model card around the specificity gain, (b) adopt v8 and keep
   sensitivity as the headline with the honest 71% number, (c) stay on v7 and document
   why. Recommendation: **(a)**, because the repo's stated principle is that the served
   model tracks the approved assessment model, and v8 clears every gate.
2. **Keep `plot_tree`-style interpretability anywhere?** v7's selling point was a readable
   5-deep tree. A 200-estimator forest is not readable the same way. The API never exposed
   tree structure, so nothing breaks, but `docs/article.md` leans on "both are tree-based"
   and the UI copy implies a single inspectable tree.
3. **Release number.** Proposing `v0.2.0` rather than `v0.1.2`: the served predictions
   change, which is a behavioural change for anyone calling `/predict/grade`.

## Work order

### 1. Contract (`ml/assessment_contract.json`)

Rewrite as `mln601-a2-v8`:

| Field | New value |
|---|---|
| `contract_version` | `mln601-a2-v8` |
| `submission_version` | `v8` |
| `source_commit` | `c5be26cf1bb7cc71f8f057fba45aa5b3ea8dd5b2` |
| `submission_sha256` | `b4aeca9b6ed0412d5855f1fa46a3afd4bc95173a8b71bdf3963fb728331dddd9` |
| `source_metrics_sha256` | `e0860e30…` of `outputs/finalist_test_metrics_v8.csv` |
| `estimator.type` | `RandomForestClassifier` |
| `estimator.params` | `n_estimators` 200, `max_depth` 10, `min_samples_leaf` 1, `class_weight` balanced, `random_state` 42, `n_jobs` 1 |
| `expected_test_metrics` | the v8 row above, full precision, plus 537/129/114/284 |

`dataset`, `feature_order`, `target` and `split` carry over unchanged.

Note on `source_metrics_sha256`: v7 pointed at `model_metrics_v7.csv`. v8 has no
`model_metrics_v8.csv`; the equivalent held-out-metrics artifact is
`finalist_test_metrics_v8.csv`. Also record `selection_summary_v8.csv` so the *approval
decision* is hashed, not only the numbers - that file is what makes the Random Forest the
approved model rather than merely the best-scoring one.

### 2. `ml/train.py`

Today `DecisionTreeClassifier` is hardcoded in both the import and the instantiation, and
only `params` comes from the contract. That must become a dispatch on
`contract["estimator"]["type"]`, otherwise the contract file documents a model the code
does not actually build. Also update: module docstring, the "A2 v7" strings in the four
parity `RuntimeError` messages, the comment above the classifier, the console print label,
and `metrics["classification"]["model"]`.

Keep `n_jobs=1` for the classifier to match the notebook exactly, even though RF training
is deterministic across `n_jobs`. Cheap insurance, and the parity test is the whole point.

### 3. `ml/contract.py`

Docstring and the two "A2 v7" references. Cosmetic but it is the file that defines what
parity means.

### 4. Retrain

`make train`. Regenerates `classifier.joblib` (2.2 MB) and `metrics.json`. The regressor
is untouched by this migration but will be rewritten by the same run; confirm its metrics
land on the same R² 0.4146 / MAE 0.5096 / RMSE 0.6634 before committing, so the diff shows
only the intended change.

### 5. Tests (this is where it breaks)

- `tests/test_train_metrics.py:76` `test_served_classifier_exactly_matches_fresh_a2_v7_retrain`
  reaches into `served.tree_` for seven internal arrays. A forest has no `.tree_`. Rewrite
  to assert `len(served.estimators_) == len(fresh.estimators_)` and then loop the same
  seven arrays over each paired `estimators_[i].tree_`. Rename to `…_a2_v8_retrain`.
  Keep the `predict` / `predict_proba` equality assertions as they are.
- `test_classification_metrics_reproduce` and `test_classification_passes_screening_gates`
  read from the contract and from fixed gate constants, so they pass unchanged once the
  contract is updated. Update their v7 comments.
- `tests/test_api.py` v7 references.
- Add one new test: assert `metrics["classification"]["model"]` equals
  `CONTRACT["estimator"]["type"]`. That is the assertion that would have caught this drift
  on its own.

### 6. Narrative

- `README.md` - 5 hits. The lens table row, the model-card grade bullet, the
  "73% instead of 59%" paragraph (see decision 1), the provenance paragraph naming v7 and
  its commit, and the `make parity` line.
- `ui/views/about.py:16-17` - `DecisionTreeClassifier`, ROC-AUC 0.79, sensitivity 0.73.
- `docs/article.md` - line 27 ("a tuned `DecisionTreeClassifier`"), line 33 ("because both
  are tree-based, there's no feature scaling at inference" - still true for a forest, keep
  the claim, fix the wording), line 45 mermaid node label.
- `docs/launch/deploy.md:33` - the portfolio table description.
- `Makefile:24` - the `parity` target comment.
- `RELEASE_NOTES.md` - new `v0.2.0` section at the top. It must state plainly that served
  predictions change and that sensitivity moved from 0.734 to 0.714, in the same spirit as
  the v0.1.0 leakage-audit entry. Do not bury the one metric that got worse.

### 7. Deploy

Render redeploys from the repo; no config change. Streamlit Cloud picks up the new
artifact on push. Verify `/health` reports `mln601-a2-v8` in production after deploy.

## Verification checklist

- `make train` completes and writes both artifacts without raising a parity error.
- `make test` green, including the rewritten forest parity test.
- `make parity` green.
- `metrics.json` classification block matches `finalist_test_metrics_v8.csv` exactly.
- Regression metrics unchanged from the current `metrics.json`.
- `grep -rn "v7" --exclude-dir=venv --exclude-dir=.git .` returns only intentional
  historical mentions (the RELEASE_NOTES v0.1.1 entry, the migration narrative).
- `GET /model/info` returns `RandomForestClassifier` and the v8 numbers locally.
- `POST /predict` request and response shapes unchanged against the current Swagger.
- No em-dashes in new text.

## Out of scope

- The A1 regressor. It is untouched by A2 v8.
- Any API schema change. Endpoints, request bodies and response keys stay as they are.
- Per-lot SHAP explanations. Still deliberately outside the API contract.
- Threshold calibration. The 0.5 decision threshold is inherited from the assessment; the
  notebook lists calibration as future work and this repo should not diverge from it.
