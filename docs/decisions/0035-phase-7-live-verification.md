# Decision 35: Phase 7 live verification — method, findings, and the ephemeral-server trap

- **Date:** 2026-08-28
- **Status:** Accepted

## Context

Steps 1–4 shipped both flows, the deployments file and 17 tests, all green. But
[0034](0034-testing-prefect-flows.md) is explicit that those tests patch the Prefect engine out
entirely: *"these tests verify our wiring, never Prefect's engine."* Everything the engine owns —
real retries against a real exception, real logging into a run context, real integration with
MLflow and the Registry, `serve()` holding two deployments with a live schedule — was still
unverified when this interaction began. Phase 7's Definition of Done is observational, not an
assertion, so this is where that debt is paid.

## Decision

**Induce the failure deliberately rather than wait for one.** `data/raw/creditcard.csv` was
renamed in place, the flow run and observed, then restored. Renaming inside the same directory is
an atomic inode operation, not a 150 MB copy across filesystems.

**Verify the restore against DVC's recorded hash, not by eye.** `creditcard.csv.dvc` records
`md5: e90efcb83d69faf99fcab8b0255024de`; `dvc status` re-hashes the file and confirmed it.

**Accept `register.py`'s automatic promotion decision with no extra gate**, and predict the
outcome from the code *before* running so it could be checked rather than rationalised afterwards.

**Demonstrate the closed loop with an in-memory patch, never a committed one.** `detect_drift` was
overridden inside a throwaway `python -c` process. No file on disk was modified, so there is no
half-reverted patch to forget in a deliverable of this phase.

**Substitute Prefect CLI/API output for dashboard screenshots**, stating plainly that it reads the
same database the UI renders but is not the same as watching the states transition.

## Alternatives considered

- **Catching a retry mid-run without inducing one** — not reproducible; it depends on hitting a
  timing window.
- **Moving the CSV outside the repository** — a real 150 MB byte copy with a genuine failure
  window mid-transfer, for no benefit over an in-place rename.
- **Editing `drift.py` to return `True`** — rejected. It would leave an artificial signal inside a
  formal deliverable of this phase, to be reverted by hand and possibly forgotten.
- **A bare `run_deployment(...)` one-liner** to prove event-driven triggering — weaker: it proves
  the deployment is addressable but never runs `monitoring_pipeline()`, so the flow under test is
  bypassed. The in-memory patch exercises the real flow, the real task, and the real trigger.
- **Adding a manual approval gate before promotion** — rejected: it contradicts
  [0016](0016-promotion-quality-gate.md) and undercuts the point of orchestration. The
  compensating control is documentation, not intervention.

## Justification

**Retries, proven against a real exception.** Flow run `dexterous-porpoise`, task `Validate data`,
`run_count=4` — one attempt plus exactly `retries=3`, at 10-second intervals matching
`retry_delay_seconds=10`:

```
14:38:41  DataIngestionError(...) - Retry 1/3 will start 10 second(s) from now
14:38:51  DataIngestionError(...) - Retry 2/3 will start 10 second(s) from now
14:39:01  DataIngestionError(...) - Retry 3/3 will start 10 second(s) from now
14:39:11  DataIngestionError(...) - Retries are exhausted
```

Prefect surfaced `src.exceptions.DataIngestionError` unwrapped, confirming
[0033](0033-orchestration-design.md)'s no-masking rule. Only `Validate data` ever ran: the gate
stopped the flow before anything could train on missing data.

**All three trigger mechanisms, observed in one listing.** Direct invocation
(`audacious-cockatrice`), a manual deployment trigger (`solid-koel`, `from_deployment=True`), the
event-driven retrain (`favorite-coyote`, created by `run_deployment()` from inside the monitoring
flow with no human in the path), and three `SCHEDULED` monitoring runs the server pre-computed for
06:00 on 2026-08-29/30/31 from `cron='0 6 * * *'`. Scheduling was verified this way rather than by
waiting a day: the scheduler either resolves the expression into future runs or it does not.

**`timeout=0` measured, not assumed.** The monitoring flow that fired a retrain finished in 1.0s,
while the training run it started continued independently.

**The promotion gate refused, three times.** Predicted before running: training is deterministic
(`random_state: 42`, `resampling: none`), so a fresh run ties the incumbent's
`0.8759962787477742`, and `promote_if_better()` requires *strictly* greater. Observed exactly
that — v4, v5 and v6 all registered, `@production` never moved off v1. Three independent runs
producing byte-identical PR-AUC is also a decent determinism check in its own right.

**The repository was untouched by six real runs.** `git status` clean and `dvc status` "up to
date" afterwards. The `dvc.lock` reconciliation anticipated before the run (Phase 6 saw an 11-byte
parquet shift after a toolchain move) proved unnecessary, because nothing in the toolchain has
moved since the last `dvc repro`.

## Trade-offs / consequences

### The finding: `PREFECT_API_URL` was never set

The real run exposed a gap no `.fn()` test could have caught. Prefect resolves its server from
`PREFECT_API_URL`; when unset, `PREFECT_SERVER_EPHEMERAL_ENABLED` (true by default) makes it start
a temporary one instead. Proven with no server running:

```
$ env -u PREFECT_API_URL uv run python -m pipelines.monitoring_pipeline
INFO | prefect - Starting temporary server on http://127.0.0.1:8883
INFO | Flow run 'excellent-chipmunk' - Finished in state Completed()
INFO | prefect - Stopping temporary server on http://127.0.0.1:8883
```

The flow **succeeds** and nothing errors, so the condition is invisible at the point of use. Every
run in this interaction reached the intended server only because the variable was set by hand.

> **Correction (2026-08-28, during the Phase 7 closure).** This record originally continued: *"the
> run simply never reaches the dashboard, and the database holding it is destroyed at process
> exit."* **Both halves were wrong**, and the error was over-reading the `Stopping temporary
> server` line above as data loss. Prefect's ephemeral server is a temporary **API server
> process**, not a temporary database: it uses the same `PREFECT_HOME/prefect.db`. Verified at
> closure by re-querying that database — the `excellent-chipmunk` run produced by the command
> above is still present, `COMPLETED`, with its task run and all six log lines intact.
>
> The decision below stands, for corrected reasons. Setting `PREFECT_API_URL` still matters:
> without it a run is not observable **live** in an already-open dashboard, every invocation pays
> seconds of temporary-server startup, two API servers end up writing the same SQLite file
> concurrently, and `serve.py` would register deployments that no running server is serving. What
> is *not* true is that runs are lost.

**Fixed in the `Makefile`**, which is already the canonical interface: a `PREFECT_API_URL ?=
http://localhost:4200/api` variable, exported by `prefect-serve` and a new `prefect-train` target,
overridable for a non-local server. `localhost` rather than `127.0.0.1` for readability; verified
that it resolves IPv4-only here, so it cannot land on `::1` while the server binds to `127.0.0.1`.
Both flow module docstrings now warn against the bare `uv run python -m ...` form.

- **The fix only protects the `make` path.** A bare `uv run python -m pipelines.training_pipeline`
  still silently uses an ephemeral server. The docstrings say so; nothing enforces it.
- **`prefect config set PREFECT_API_URL=...` was rejected** as the fix, though the reference
  material suggests it: it writes to `~/.prefect/profiles.toml`, invisible machine state outside
  the repository, so a fresh clone would behave differently from this one for no discoverable
  reason.
- **No test guards this.** A pytest asserting the content of a Makefile target would be testing a
  build file from the wrong layer; the docstrings and this record carry it instead.

### The second finding: `src/` logs do not reach the dashboard

Zero log lines from `src/` appear in any flow run. The tasks' own `get_run_logger()` messages
arrive; the `logging.getLogger(__name__)` calls inside the wrapped modules do not — including
`register.py`'s *"NOT promoted: ... does not beat"* decision log. The dashboard therefore shows
"Registration and promotion completed" without showing what was decided.

Deliberately **not fixed here**. `PREFECT_LOGGING_EXTRA_LOGGERS=src` would forward them, but that
is an environment-wide logging change to a Steps 1–4 deliverable, and Phase 8 will revisit logging
when drift reports need surfacing. Recorded so it stays visible.

### Other consequences

- **Three Registry versions were created** (v4, v5, v6) purely to demonstrate three trigger
  mechanisms, since each necessarily runs the real pipeline. None was promoted. Future
  verification runs will keep accumulating versions; nothing prunes them.
- **Event-driven triggering is proven as a mechanism, not as a behaviour.** The `True` came from a
  patch, because `detect_drift()` returns `False` by design until Phase 8. Every link downstream
  of the predicate is real and was exercised; the predicate itself is Phase 8's deliverable.
- **This verification is not reproducible in CI.** It needs the dataset (whose DVC remote is a
  local path, [0005](0005-dvc-local-remote.md)), a live MLflow, and a Prefect server. It is a
  local, manual gate — like Phase 5's container check, and unlike Phase 6's, which lives on
  GitHub.
