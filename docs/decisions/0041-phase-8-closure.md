# Decision 41: Phase 8 closure — verification by rerun, and what the tag does and does not claim

- **Date:** 2026-08-29
- **Status:** Accepted

## Context

Phase 8's Definition of Done ends with a criterion unlike any other in this project: *"**The key
test:** You inject drifted data, the system detects it, the alert is triggered, and retraining is
initiated."* Six of the seven boxes describe properties of code, and every one of them would still
be true if the loop had never fired once. The seventh is the only claim about **behaviour**, and it
is what the `phase-8-complete` tag actually asserts.

Phase 7's closure faced one source of mutable evidence, `~/.prefect/`. This one faced three, and
they had diverged independently since the Steps 7–8 interaction:

| Source | State at closure |
|---|---|
| Prefect history | **Intact** — `masterful-myna` and `sarcastic-avocet`, both COMPLETED with `deployment` provenance |
| `logs/predictions.jsonl` | **Did not contain the batch** — 3 records, all dated 2026-08-23 |
| Model Registry | **Intact and unambiguous** — v7 at 16:55:21, `aliases=[]`, `@production` still v1 |

The middle row is not a loss. [0040](0040-closed-loop-demonstration.md) archived the drifted log and
restored the 3-record baseline on purpose, because a permanently drifted log makes every future
scheduled run report drift and retrain on stale demo data. But it does mean the live log no longer
*showed* the injection.

## Decision

**Rerun the key test in full rather than reconfirm it from surviving artifacts**, and treat the
rerun as the evidence for item 7.

**Correct `scripts/simulate_drift.py` in place** rather than leave a comment contradicting
[0040](0040-closed-loop-demonstration.md).

**Record the dashboard logging gap as a known limitation rather than fix it here.**

**Write one closure record**, not four. The threshold and current-data source are already fixed in
[0037](0037-reference-dataset.md)/[0038](0038-evidently-dependency-and-api.md), alerting in
[0039](0039-alerting-and-report-placement.md), batch calibration in
[0040](0040-closed-loop-demonstration.md).

**State in the README and here that Phase 9 is all that remains.**

## Alternatives considered

- **Lightweight reconfirmation by re-querying Prefect and the Registry**, the method
  [0036](0036-phase-7-closure.md) used and defended for Phase 7. Rejected here, and the difference
  is worth naming: Phase 7's evidence was audited and found *intact in place*. Here one of the
  three links had been deliberately moved, and a closure that certifies "the system detects drift
  in its own traffic" should not rest on a log that no longer contains any. The rerun also cost
  about forty seconds.
- **Restoring the archived drifted log and re-running the check against it.** Rejected: it would
  have exercised the detector but skipped the API entirely, proving nothing about the path from
  HTTP request to logged record — the half that makes this a *system* test rather than a function
  test.
- **Leaving the wrong comment in `simulate_drift.py`** since an ADR already corrected it. Rejected.
  A reader meets the code before the record, and this repository has twice corrected stale claims
  in place rather than superseding them (`9f7cb1f`, `7c36cb7`).
- **Fixing the dashboard logging gap during closure.** Rejected: it means changing logging
  configuration or adding a line to `pipelines/monitoring_pipeline.py` — a behaviour change to
  verified wiring, on the day that wiring is certified. Documented instead.
- **Claiming the webhook path as verified.** Rejected — see deviations.

## Justification

**The rerun, as a controlled experiment.** The restored baseline made it possible to show both
branches in one session, which the Steps 7–8 demonstration never did:

```
20:47:12  BEFORE  Registry v1..v7, @production v1, pr_auc 0.875996
                  logs/predictions.jsonl: 3 records, md5 ec8e4fa8…
20:47:12  BEFORE  detect_drift() -> False
                  "Drift check skipped: 3 rows … below the 100-row minimum"
20:47:47  INJECT  make simulate-drift -> Accepted: 300/300   (0 failures)
                  log 3 -> 303; injected Amount mean 2910.20 (reference 87.86),
                  V1 mean 3.080 (reference 0.010)
20:47:54  TRIGGER make prefect-monitor -> flow run 'orchid-capybara'
20:48:02          Task 'Check drift-813'  Drift detected? True
20:48:02          Flow 'orchid-capybara'  Drift detected: triggering retraining
20:48:02          src.monitoring.drift    DRIFT ALERT: Significant drift detected.
20:48:02          flow run 'comical-turaco' created, source=deployment
20:48:11          Validate data-871        Validation OK: 284807 rows, fraud rate 0.0017
20:48:12          Preprocess data-3fb      Completed
20:48:19          Train model-7d8          Completed
20:48:20          Register and promote-ce1 Completed
20:48:21          Flow 'comical-turaco'    Finished in state Completed()
20:48:20  AFTER   fraud-detector v8 created, aliases=[], pr_auc 0.8759962787477742
                  @production -> v1  (unchanged)
```

Both flow runs carry `deployment` provenance and all four training tasks completed with
`run_count=1`. `comical-turaco` was created by `run_deployment()` from inside `orchid-capybara`,
**no human input between the first HTTP request and v8**.

**The outcome was predicted before the run, so it could be checked rather than rationalised.**
Training is deterministic (`random_state: 42`) on unchanged data and `promote_if_better()` requires
a *strictly* greater PR-AUC, so v8 should be created and refused promotion. It was: `pr_auc`
identical to v1 to the last digit, `aliases=[]`, `@production` still v1. **The gate refusing an
equal model is the gate working.** Five unpromoted versions (v4–v8) now exist for exactly this
reason.

**The `False` before the injection is as important as the `True` after it.** It is not merely "no
drift" but a refusal to answer on three rows — evidence that the `True` twenty seconds later came
from the injected batch and not from a detector that reports drift on anything.

**`dvc status` reports "Data and pipelines are up to date"** after a real training run, because the
pipeline reproduced byte-identical outputs — the same determinism the equal PR-AUC shows.

## Trade-offs / consequences

### Two DoD items met with qualifications, not clean passes

- **The webhook half of the alert is unproven.** `DRIFT_WEBHOOK_URL` is unset in this environment
  and has only ever been exercised against `http://localhost:1`, which verifies the *failure*
  path — `requests.RequestException` caught, logged, execution continuing — and nothing else. No
  message has ever reached a real Slack or Discord endpoint. The DoD wording (*"logs and,
  **optionally**, a webhook"*) makes logging the requirement and the webhook optional, so this is
  recorded as met-with-a-caveat rather than failed.
- **The Evidently report was inspected programmatically, not visually.** Size, well-formedness and
  the presence of the drift sections and per-column names were asserted from the file; a person
  looking at rendered distributions is evidence this environment cannot capture. Same footing as
  Phase 7's dashboard item ([0036](0036-phase-7-closure.md)).

### The drifted share never reaches the Prefect dashboard

`detect_drift()` logs it at INFO on a module logger that propagates to a root logger left at
WARNING, so only the `DRIFT ALERT` line surfaces. The dashboard shows `Drift detected? True` from
the task's `get_run_logger()` — the verdict, not the number behind it. Deliberately not fixed
during a closure; it belongs to whoever next touches logging configuration.

### Other consequences

- **Five permanent unpromoted Registry versions.** The honest price of demonstrating a loop whose
  every run trains for real.
- **The prediction log is restored, not preserved.** `logs/predictions.jsonl` is back to its
  3-record baseline (md5 `ec8e4fa826a27e41b196308e403d461b`), with
  `predictions.drifted-rerun-20260829-204712.jsonl` (303 records) and the earlier
  `predictions.drifted-demo-20260829-165356.jsonl` archived beside it. Nothing was deleted; all of
  `logs/` is git-ignored. The restored state was re-verified safe: the row guard trips and
  `detect_drift()` returns `False`.
- **This verification is not reproducible by a third party.** It needs the dataset (local DVC
  remote, [0005](0005-dvc-local-remote.md)), a live MLflow, a Prefect server and the API. Unlike
  Phase 6's, whose evidence lives on GitHub.
- **All three servers were stopped**; ports 8000 and 4200 released.

### What the tag claims

`phase-8-complete` asserts that this system detects a change in its own input distribution and
retrains itself without human intervention — observed twice, on 2026-08-29, at 16:55 and 20:48.
It does **not** assert that the model has been shown to degrade in production, nor that any
retrained model was ever promoted: no production traffic exists, and no ground truth is ever
received. Those are properties of a deployed service with real users, which is Phase 9's territory
and beyond.

**With this tag, the eight phases that build the system are complete.** Phase 9 — deployment,
final README and demo — is all that remains on the roadmap.
