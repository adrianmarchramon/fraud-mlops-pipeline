# Decision 34: Testing Prefect flows without a server, and what those tests do not cover

- **Date:** 2026-08-28
- **Status:** Accepted

## Context

Phase 7's Definition of Done is observational: *"you run the training pipeline with a command and
see each stage execute in the dashboard, with automatic retries if a task fails."* That is not a
`pytest` assertion, and it is deferred to Step 6. But the placeholder docstrings this phase
replaces state the standard plainly — *"Pytest coverage is non-negotiable as soon as testable
logic exists"* — and this phase creates testable logic. The constraint is that a flow test must
not need a Prefect server, MLflow, the Registry, or the dataset, none of which a CI runner has
(the dataset's DVC remote is a local filesystem path, [0005](0005-dvc-local-remote.md)).

## Decision

**`tests/test_pipelines.py`, 17 tests, offline, ~0.4s.** No Prefect server, no MLflow, no data.

**Mock at the module level in the flow's own namespace**, patching two distinct things:

- `get_run_logger` → a plain `logging.Logger`, in both flow modules.
- **the task objects themselves** (`validate_task`, `preprocess_task`, …, `check_drift_task`) when
  testing a flow body; the **wrapped functions** (`validate_raw_data`, `run_train`, …) when testing
  a task body via `.fn()`.

**Coverage:** each task calls exactly its wrapped function once; the flow runs the four stages in
order; the retry budget of all five tasks; both branches of the drift trigger, including that
`run_deployment` receives `name="training-pipeline/on-demand", timeout=0`; the real unpatched
`detect_drift()` returns `False`; the flow names; and the deployment-name coupling between
`monitoring_pipeline.py` and `serve.py`.

**`prefect.testing.utilities.prefect_test_harness` is not used.**

## Alternatives considered

- **`prefect_test_harness`** — the official utility, already packaged with Prefect so it costs no
  new dependency. Rejected as the default: it spins up an ephemeral Prefect API and database,
  which is exactly the dependency these tests exist to avoid, and it would verify Prefect's engine
  rather than our wiring. Reserved for a specific case that `.fn()` cannot express.
- **Patching only the wrapped functions and letting the real tasks run** — the obvious reading of
  "call `.fn()` on the flow", and it is a trap. Measured on Prefect 3.8.4: calling a `@task`
  outside a flow context **executes it**, starting a temporary server
  (`Starting temporary server on http://127.0.0.1:8437`) and running the function for real. The
  flow test would have taken tens of seconds and, without the function patches, would have
  triggered real training.
- **Calling `training_pipeline()` rather than `training_pipeline.fn()`** — runs the whole Prefect
  engine, same objection, and would need a server.
- **No tests at all this phase**, deferring everything to Step 6's observation — rejected: it
  would leave the loop's only silent-failure coupling unguarded.

## Justification

Two properties of Prefect 3 were measured before the strategy was chosen, and both contradict the
obvious approach:

```
get_run_logger() outside a run context  → raises MissingContextError
flow.fn()  (unpatched logger)           → raises MissingContextError
bare task() outside a flow              → EXECUTES, starting a temporary server
```

Hence patching the task objects, not merely the functions beneath them.

The suite was then verified to **refuse** a broken pipeline rather than pass vacuously — the same
standard [0028](0028-model-quality-gate-threshold.md) applied to the model gate. Two mutations
were introduced and reverted:

| Mutation | Result |
|---|---|
| `"training-pipeline/on-demand"` → `"training-pipeline/ondemand"` | 2 tests fail: the trigger assertion and the `serve.py` coupling check |
| `train_task()` called before `preprocess_task()` | 1 test fails: `assert calls == ["validate", "preprocess", "train", "register"]` |

Both files were restored and confirmed byte-identical by `sha256sum`; the suite returned to 66
passing.

## Trade-offs / consequences

- **These tests verify our wiring, never Prefect's engine.** That Prefect *actually* retries,
  *actually* logs to the dashboard, and *actually* honours a cron schedule is untested here by
  design — it is Prefect's code, and it is what Step 6 observes directly.
- **`test_task_retry_budget` pins numbers, not behaviour.** It will catch an accidental edit to the
  calibration; it cannot tell whether the calibration is right.
- **Patching module globals couples the tests to module structure.** Renaming `run_train` inside
  `training_pipeline.py` breaks a test that does not mention training — the usual cost of
  monkeypatching by name, accepted because the alternative is a live server.
- **Suite size grew 49 → 66.** Runtime is effectively unchanged (~4.8s), because nothing here
  imports the ML stack that was not already imported.
