# Decision 40: How the closed loop was demonstrated, and what the demonstration left behind

- **Date:** 2026-08-29
- **Status:** Accepted

## Context

Phase 8's Definition of Done ends with a criterion no test can satisfy: *"**The key test:** You
inject drifted data, the system detects it, the alert is triggered, and retraining is initiated."*
Every prior Phase 8 interaction was forbidden from running a pipeline. This one had to, because
"the system retrains itself" is a claim about behaviour over time, not about code.

Three questions had to be settled before touching anything: how to inject the traffic, how to
calibrate the shift, and what to do with the prediction log afterwards.

## Decision

**Inject over real HTTP.** `scripts/simulate_drift.py` posts 300 transactions to
`POST /predict` on the running API. Nothing writes `logs/predictions.jsonl` directly.

**Calibrate against the measured reference, not the reference material's figures.**

**Trigger the monitoring deployment manually** (`make prefect-monitor`) rather than waiting for the
06:00 cron.

**Archive the drifted log and restore the pre-demo baseline** once all evidence was captured.

**Test `_drifted_share` directly**, private name and all, with an autouse fixture redirecting
`DRIFT_REPORT_PATH` to `tmp_path`.

**Expand `[tool.mypy].files` to include `scripts/`.**

## Alternatives considered

- **Writing records straight into `logs/predictions.jsonl`.** Rejected: it fakes the outcome and
  proves nothing. The point is to exercise the Phase 4 path — Pydantic validation, model scoring,
  JSONL append — which a file write skips entirely.
- **Calling `monitoring_pipeline()` as a plain Python function.** Rejected on the same ground: it
  would bypass the deployment resolution that `run_deployment()` depends on, which is the exact
  link under test.
- **Copying the reference material's shift constants unexamined.** They happen to be reasonable
  for this dataset, but "happens to work" is not a justification. Each is now stated in
  σ-units of the measured reference in `scripts/simulate_drift.py`.
- **Leaving the drifted log in place.** Rejected — see consequences. It would leave the detector
  permanently tripped.
- **Deleting the drifted log.** Rejected: no destructive mutation. It is archived alongside the
  restored baseline.
- **Renaming `_drifted_share` to the document's public `drift_share_between`.** Rejected as churn
  in code two interactions had closed; the underscore means "internal to the package", not
  "untestable".

## Justification

**Baseline before anything ran** (2026-08-29T16:54:29+02:00): 6 registered versions,
`@production` → v1, `pr_auc = 0.875996`, `logs/predictions.jsonl` at **3 records**.

**A prediction made before the run, so it could be checked rather than rationalised:** training is
deterministic (`random_state: 42`) on unchanged data, and `promote_if_better()` requires a
*strictly* greater PR-AUC — therefore the demo should produce **v7 without promoting it**.

**The chain, as it actually happened:**

```
16:54:32  300/300 transactions accepted by POST /predict          (0 failures)
          log 3 -> 303 records; injected Amount mean 2910.20, V1 mean 3.080
16:54:50  prefect deployment run monitoring-pipeline/daily -> 'masterful-myna'
16:55:01  Task 'Check drift-310'  - Drift detected? True
16:55:01  Flow 'masterful-myna'   - Drift detected: triggering retraining
16:55:01  src.monitoring.drift    - DRIFT ALERT: Significant drift detected. Triggering retraining.
16:55:01  flow run 'sarcastic-avocet' created, source=deployment
16:55:12  Validate data-295        Validation OK: 284807 rows, fraud rate 0.0017
16:55:14  Preprocess data-687      Completed
16:55:20  Train model-996          Completed
16:55:21  Register and promote-ed9 Completed  -> fraud-detector v7
16:55:22  Flow 'sarcastic-avocet'  Finished in state Completed()
```

`sarcastic-avocet` carries `deployment` provenance and was created one second after the drift
verdict, by `run_deployment()` from inside `masterful-myna`. **No human touched anything between
the injection and v7.**

**Outcome matched the prediction exactly:** v7 created 2026-08-29T16:55:21 with
`pr_auc = 0.8759962787477742` — identical to v1 — and `@production` still v1. The promotion gate
refused an equal model, which is the gate working, not the demo failing.

**The measured share was 1.0000 across 303 rows**, and this corrected a prediction of mine. I had
designed `Time ~ U(0, 200_000)` to sit *in* distribution (reference range [22, 172783], means
94,895 vs 100,540) and expected 29/30 columns to drift. All 30 did. The reasoning was wrong in an
instructive way: a K-S test compares **distributional shape**, not range or mean, and the real
`Time` column is bimodal — two days of transaction cycles — while a uniform draw is flat. Matching
the mean does not make a distribution the same distribution.

**The report from the run** is at `reports/drift/drift_report.html`, 5,783,562 bytes, valid HTML,
containing the Data Drift summary and per-column sections. Column means it depicts:

| | reference | current |
|---|---|---|
| `Amount` | 87.858 | 2882.215 |
| `V1` | 0.010 | 3.042 |
| `V14` | 0.011 | 3.161 |
| `Time` | 94894.914 | 100540.188 |

**Test isolation was proven, not assumed.** The demonstration report's md5 was
`a566e11b6bc274f600f13f4a871d89c2` before a full `pytest` run and identical after — the autouse
fixture genuinely redirects the ~6 MB write. And the extraction path is genuinely covered: mutating
`DRIFTED_COLUMNS_METRIC` to a wrong identifier turned 11 passing tests into **4 failures**, then
restoring it returned all 11 to green.

**`dvc status` reports "Data and pipelines are up to date"** after a real training run, because the
pipeline reproduced byte-identical outputs — the same determinism the equal PR-AUC demonstrates.

## Trade-offs / consequences

- **v7 is permanent and unpromoted**, joining v4–v6 from Phase 7. Four registered versions now
  exist solely as evidence that the gate refuses equal models. That is the honest cost of
  demonstrating a loop whose every run trains for real.
- **The prediction log was restored, so the repository does not show the demo traffic.** Left in
  place, its 303 records — 99% drifted — would have made *every* future monitoring run return
  `True`, retraining daily off stale demo data the moment `serve.py` was started again. That is a
  stuck alarm, not a working system. Both files are kept:
  `logs/predictions.drifted-demo-20260829-165356.jsonl` (303 records) and
  `logs/predictions.pre-demo-20260829-165356.jsonl`, with `logs/predictions.jsonl` restored
  byte-identically (md5 `ec8e4fa826a27e41b196308e403d461b`). Phase 9 can replay the detection
  instantly by pointing `CURRENT_DATA_PATH` at the archive. All of `logs/` is git-ignored, so none
  of this enters the repository.
- **The restored state was verified safe:** `detect_drift()` now logs *"Drift check skipped: 3
  rows … below the 100-row minimum"* and returns `False`.
- **The measured share never reaches the Prefect dashboard.** `detect_drift()` logs it at INFO on
  the module logger, which propagates to a root logger left at WARNING, so only the `DRIFT ALERT`
  line appears. The dashboard shows the boolean (`Drift detected? True`) from the task's
  `get_run_logger()`, not the number behind it. Not fixed here — it is a logging-configuration
  question that belongs with the phase closure, not a defect in the loop.
- **Three `SCHEDULED` monitoring runs sit in the Prefect database** from the cron's look-ahead when
  `serve.py` started. With the log restored they would hit the row guard and do nothing.
- **The demonstration is not reproducible by a third party**, needing the dataset (local DVC
  remote, [0005](0005-dvc-local-remote.md)), a live MLflow, a Prefect server and the API. Same
  limitation Phase 7's closure recorded.
- **`scripts/` is now type-checked**, so the demo script carries full annotations. The first file
  this project has placed outside `src/` and `pipelines/`.
- **All three servers were shut down** and ports 8000 and 4200 released;
  `~/.prefect/prefect.db` retains the run history.
