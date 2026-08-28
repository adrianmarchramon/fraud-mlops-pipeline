# Decision 33: Orchestration design — task granularity, retry calibration, and no exception masking

- **Date:** 2026-08-28
- **Status:** Accepted

## Context

`pipelines/training_pipeline.py` and `pipelines/monitoring_pipeline.py` add an orchestration layer
over logic that already exists and is already tested. The decisions worth recording are not *what*
the flows do — the reference fixes that — but how much they are allowed to do, how failure is
handled, and why DVC stays.

## Decision

**Prefect is added alongside DVC, not instead of it.** `dvc.yaml` keeps answering "what ran and
with what result" (deterministic, cached, content-addressed); the flow answers "when it ran and
how reliably" (schedule, retries, observability, event triggering).

**Tasks call the Python functions directly**, not `dvc repro`.

**One task wraps exactly one existing function.** `validate_task` → `validate_raw_data`,
`preprocess_task` → `preprocess`, `train_task` → `train`, `register_task` → `register.main`. No
business logic is rewritten, reordered, or reconfigured. The flow's sole responsibility is
sequencing.

**`validate_task` returns `ValidationReport`, not `dict`.** The reference's bare `dict` is a hard
`mypy --strict` error (`Missing type arguments for generic type "dict"  [type-arg]`), and the real
signature at `src/data/validate.py:45` is a `TypedDict` with `n_rows`, `n_fraud`, `fraud_rate`,
`status`.

**Retry budgets are calibrated per task:**

| Task | Retries | Delay | Why |
|---|---|---|---|
| Validate data | 3 | 10s | Cheap (one CSV read + Pandera contract); realistic failures are transient I/O, so tolerance is nearly free |
| Preprocess data | 2 | 10s | Data faults already caught upstream, so a failure here is more likely deterministic |
| Train model | 2 | 30s | Minutes per attempt; longer delay in case MLflow is briefly unreachable |
| Register and promote | 2 | 10s | Network-bound to the Registry — genuinely transient-prone — but each attempt is cheap |
| Check drift | 2 | 30s | Phase 8 will make this read prediction logs and compute distributions |

**No task catches the exceptions raised by the function it wraps.** Prefect sees the original
`src/exceptions.py` error.

**`from src.monitoring.drift import detect_drift` sits at module level**, not inside
`check_drift_task` as the reference shows.

**The deployment name is a module constant**, `TRAINING_DEPLOYMENT = "training-pipeline/on-demand"`,
and `pipelines/serve.py` exposes `TRAINING_DEPLOYMENT_NAME`/`MONITORING_DEPLOYMENT_NAME`/
`MONITORING_CRON` at module level rather than inline.

**`from __future__ import annotations` is omitted**, matching every other module in the repo.

## Alternatives considered

- **A task invoking `dvc repro`** — the reference names this as the other valid integration. It
  would preserve DVC's cache and skip unchanged stages, but the dashboard would show one opaque
  box instead of four stages, losing precisely the per-stage observability this phase exists to
  demonstrate. Worth revisiting if flow runtime becomes a problem.
- **Uniform retries across all tasks** — simpler, and wrong. Retrying a deterministic failure in a
  multi-minute training run is pure waste; not retrying a transient file read is a needless
  failure. Uniformity would make the setting a default rather than a decision.
- **`try/except` inside tasks, re-raising a new orchestration exception** — rejected. It would put
  a wrapper between Prefect and the real cause, so retries would act on the wrapper and the
  dashboard would report it instead of `DataValidationError`. It would also require a new
  exception class for no gain.
- **Adding `OrchestrationError`/`DriftDetectionError` to `src/exceptions.py`** — rejected for now.
  Nothing in this phase can raise them: the tasks re-raise what they receive, and `detect_drift()`
  returns a constant. The repo's pattern (`ModelRegistrationError`, `PredictionError`) is one
  exception per module, introduced when that module can actually fail. Phase 8 should add one
  alongside the code that needs it.
- **Inlining the deployment name at the `run_deployment()` call**, as the reference does —
  rejected. See below.
- **`.submit()` for concurrency** — inapplicable: each stage consumes what the previous produced.

## Justification

**Wrapping is cheap only because of a decision made six phases ago.** All four targets are
zero-argument, side-effect-complete entry points with no CLI parsing and no notebook state — each
already the body behind a `make` target or a `dvc.yaml` `cmd`. That is what lets an orchestration
layer be four decorators instead of a rewrite, and it is the concrete payoff of the separation of
concerns held since Phase 1.

**The deployment name is the loop's single point of silent failure.** `run_deployment()` resolves
`"<flow-name>/<deployment-name>"` at runtime. If either half drifts from what `serve.py`
registers, nothing raises: monitoring runs, reports no drift or reports drift, and retraining
simply never fires. That is the same class of hazard as the CI check-name coupling in
[0027](0027-branch-protection-boundary.md), and it is handled the same way — made explicit and
pinned by a test ([0034](0034-testing-prefect-flows.md)).

## Trade-offs / consequences

- **DVC's cache is bypassed on this path.** `training_pipeline()` re-runs every stage regardless
  of whether inputs changed, while `dvc repro` would skip the unchanged ones. Accepted for
  per-stage visibility; it also means the flow is slower than `dvc repro` on a warm cache.
- **Two entry points now train the model** — `dvc repro` and the flow — and they do not share
  state. Whichever ran last determines what is on disk.
- **The retry table is a starting calibration, not a measurement.** No stage has yet failed
  transiently in practice; the numbers encode reasoning about cost and failure mode, and should be
  revised once Step 6 and Phase 8 produce real failure data.
- **Retries make some failures slower to surface.** A deterministic training failure now takes
  three attempts and 60s of delay before the flow gives up.
