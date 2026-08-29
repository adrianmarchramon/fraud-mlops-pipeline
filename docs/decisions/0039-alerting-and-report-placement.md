# Decision 39: Two-tier alerting, and why the drift report is not a dashboard module

- **Date:** 2026-08-29
- **Status:** Accepted

## Context

Steps 4–6 of `project_context/mlops_phase8.md` add the two things that turn a drift *number* into a
system that acts: a report someone can look at, and a notification that fires when the number
crosses the line. Both arrived with a placement question the document answers only by implication.

The report question was sharpened by a gap. The document's Step 4 opens *"Take a moment to open the
`drift_report.html` file that Evidently generated"* — but this repository never generated one:
Steps 1–3 deliberately shipped `detect_drift()` as a pure predicate and deferred report writing.
Meanwhile `src/monitoring/dashboard.py` has existed as an empty Phase 0 placeholder since the
beginning, the README architecture diagram names it inside the Monitoring box, and an earlier
recommendation in this phase had proposed giving it the report.

## Decision

**The report is written inside `_drifted_share()` in `src/monitoring/drift.py`**, from the same
`Report.run()` evaluation that produces the verdict, to `config.DRIFT_REPORT_PATH`
(`reports/drift/drift_report.html`), under a fixed filename, on every check.

**`src/monitoring/dashboard.py` stays empty**, its docstring rewritten to say where the report
actually lives and why nothing is here.

**`send_alert(message: str) -> None` lives in `src/monitoring/drift.py`** and is called from
`pipelines/monitoring_pipeline.py`, immediately before `run_deployment(...)`.

**A webhook failure is caught and logged; a report write failure is warned and skipped.** Neither
can stop a drift verdict or a retrain.

**`DRIFT_WEBHOOK_URL` is read in `src/config.py`**, not inline in `send_alert()`.

**`requests` is a production dependency.**

## Alternatives considered

- **Rendering the report from `dashboard.py`**, as recommended earlier in this phase and as the
  architecture diagram suggests. **Rejected on a fact that recommendation did not have:** the
  report and the drifted share are two views of *one* evaluation. A separate module would have to
  re-run the entire Evidently comparison to draw what the first run had already computed — seconds
  of duplicated work per monitoring run, and two code paths that could disagree about what was
  measured.
- **Deleting `dashboard.py`.** Rejected: the diagram references it, and an empty file that explains
  itself is more useful to a reader than a dangling name in a picture.
- **A timestamped report filename**, also recommended earlier in this phase. Rejected on
  reconsideration: nothing in this project reads report history, the README and the Step 7 demo
  want one stable path to point at, and timestamped files would accumulate indefinitely in a
  git-ignored directory with no retention policy.
- **`reports/drift_report.html`**, where the document puts it. Rejected — `reports/` is tracked in
  Git (`reports/metrics.json`, `reports/validation.json`) with no ignore rule, so this would have
  committed a 5.8 MB HTML file on every run. A blanket `reports/` ignore was rejected in turn: CI's
  model-quality gate reads `reports/metrics.json` from a bare checkout. Hence the subdirectory and
  a rule scoped to it.
- **Calling `send_alert()` from inside `detect_drift()`** rather than from the flow, which an
  earlier recommendation in this phase preferred because it avoided editing `pipelines/`. Rejected
  once editing that file was authorised: alerting is a consequence of the *decision to retrain*,
  which the flow makes, not of the measurement, which `drift.py` makes. A predicate that notifies
  people is no longer a predicate.
- **`os.getenv("DRIFT_WEBHOOK_URL")` inline in `send_alert()`**, exactly as the document writes it.
  Rejected for `CLAUDE.md`'s standing rule that configuration lives in `src/config.py`. The cost is
  real and recorded below.
- **`requests` as a dev dependency**, which the document implies by pairing `uv add --dev requests`
  with the demo script. Rejected on the same rule as `evidently` ([0038](0038-evidently-dependency-and-api.md)):
  `src/` imports it, and `.dockerignore` ships all of `src/` into an image built `--no-dev`.

## Justification

**The report was generated and inspected, not assumed:**

```
INFO src.monitoring.drift: Drift report written to reports/drift/drift_report.html
INFO src.monitoring.drift: Drift check: 1.0000 of columns drifted across 500 rows -> drift=True

bytes: 5,775,206     <!DOCTYPE html> … </html>      5 <script> blocks
contains 'Data Drift': True   'Share of Drifted Columns': True
contains 'V14': True   'Amount': True   'reference': True   'current': True
git check-ignore -v  ->  .gitignore:29:reports/drift/
```

Re-run against an unshifted window it was overwritten in place (5,775,206 → 5,793,774 bytes,
`share` 1.0000 → 0.0667), confirming both states render and that the fixed filename behaves as
intended.

**Why a swallowed webhook error is not laxity.** `send_alert()` is called inside a flow whose task
carries `retries=2`. An escaping `ConnectionError` would fail the monitoring run, burn both
retries, and — decisively — prevent the `run_deployment(...)` line *after* it from ever executing.
A Slack outage would become a model outage. Measured:

```
no webhook   WARNING  DRIFT ALERT: Significant drift detected. Triggering retraining.
             returned: None
unreachable  WARNING  DRIFT ALERT: Significant drift detected. Triggering retraining.
             ERROR    Could not send the alert to the webhook: HTTPConnectionPool(
                      host='localhost', port=1): Max retries exceeded …
             returned: None  — no exception propagated
```

Only `requests.RequestException` is caught. A `TypeError` from a malformed payload is this module's
bug, not the network's, and still surfaces.

**The same reasoning, already precedent.** [0021](0021-prediction-log-and-api-tests.md) decided that
a prediction-log write failure is warned rather than raised, because *"an observability fault would
suppress a fraud decision the model already produced"*. The report write is the identical shape and
takes the identical answer.

**`requests` is genuinely type-checked**, unlike `evidently`: it ships its own `py.typed`
(`site-packages/requests/py.typed` exists), so no override was added. Verified by mutation — adding
a bogus keyword to the `post()` call produced `error: Unexpected keyword argument "nonsense_kwarg"
for "post" [call-arg]`, and removing it restored `Success: no issues found in 23 source files`.

**`timeout=10` is load-bearing.** Without it a webhook that accepts the connection and never
answers would hang the monitoring flow indefinitely, holding its scheduled slot open until the next
cron tick collides with it.

## Trade-offs / consequences

- **`detect_drift()` now has a filesystem side effect on every call.** It was a pure predicate
  through Step 3. The Step 8 tests must redirect `DRIFT_REPORT_PATH` to a `tmp_path`, or the suite
  will write 5.8 MB into the repository on each run — the same trap `tests/test_api.py` avoids with
  its autouse `PREDICTIONS_LOG` fixture.
- **`DRIFT_WEBHOOK_URL` binds at import**, being a `config.py` constant. A test or a shell that
  changes it after import must patch `src.monitoring.drift.DRIFT_WEBHOOK_URL`, not the environment
  and not `src.config` — the exact edge [0021](0021-prediction-log-and-api-tests.md) documents for
  `PREDICTIONS_LOG`. This is the price of the centralised-config rule over the document's inline
  `os.getenv`.
- **No webhook is configured**, so the real-time channel is unproven against a live endpoint. Only
  the failure path has been exercised. Setting `DRIFT_WEBHOOK_URL` to a Slack or Discord URL is all
  that is required.
- **The report is overwritten every run**, so there is no visual history of past drift events. Fine
  while nothing reads history; it would need revisiting if the README ever wants a timeline.
- **Two log lines announce one event** — the flow's `get_run_logger()` warning and `send_alert()`'s
  module-logger warning. Deliberate: they go to different sinks (the Prefect dashboard and the
  process log), and the second is what carries to a webhook.
- **`pipelines/monitoring_pipeline.py` changed for the first time since Phase 7 closed.** One
  import, one call, and a module docstring that had gone stale — it still described the predicate
  as a placeholder pinned to `False` and promised the loop would energise "with no change here".
  Corrected in place, on the precedent of `9f7cb1f` and `7c36cb7`.
- **Two README claims are now false and deliberately left for the phase closure:** the *"loop is
  wired, not yet armed"* callout, and the repository-structure line describing `monitoring/` as
  *"drift detection (Evidently), dashboard"*.
