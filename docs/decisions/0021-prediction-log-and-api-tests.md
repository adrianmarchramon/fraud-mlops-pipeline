# Decision 21: The prediction log as Phase 8's contract, and how the API is tested

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

Phase 8 detects drift by comparing what the model sees in production against what it was trained
on. It needs a record of production traffic, and `CLAUDE.md` requires one "from day one of the
API". Unlike code, this cannot be added retroactively: traffic that was never logged is gone.

`project_context/mlops_phase8.md:77` and `:89-91` show how that phase will read it:

```python
from src.config import PREDICTIONS_LOG, RAW_DATA, TARGET
...
records.append(json.loads(line)["input"])
```

So the constant name and the `"input"` key are a contract with a phase that does not exist yet.

## Decision

**The log.** `PREDICTIONS_LOG = PROJECT_ROOT / "logs" / "predictions.jsonl"`. One JSON object per
line, appended by `log_prediction()` after the response is built and before it is returned:

```json
{"timestamp": "...+00:00", "input": {30 raw fields}, "fraud_probability": ..., "is_fraud": ..., "model_version": "..."}
```

The directory is created lazily on first write; `logs/` is git-ignored, with no tracked
placeholder, because nothing needs the directory to pre-exist.

**Timestamps are UTC**, via `datetime.now(UTC).isoformat()` — offset-carrying, unambiguous, and
lexicographically sortable. Local time is non-monotonic across DST (an hour repeats or vanishes)
and means different things in a container than on a laptop. Phase 8 will window these records by
time, so the window has to mean one thing everywhere. *(The reference material writes
`timezone.utc`; ruff's `UP017` requires the `datetime.UTC` alias under `target-version = "py312"`.
Same value, different spelling.)*

**A write failure is warned, not raised.** `OSError` is caught, `logger.warning(..., exc_info=True)`
records it, and the prediction is returned. Rejected payloads (422) never reach this code, so they
never enter the baseline.

**Tests inject a double into `app.state`** and build `TestClient(app)` *without* its context
manager, so `lifespan` never runs and no Registry lookup happens. An **autouse** fixture redirects
`PREDICTIONS_LOG` to `tmp_path` for every test in the module.

## Alternatives considered

- **A database or event stream** — correct at real scale, rejected here. JSONL is crash-safe under
  append, needs no schema migration, and `pandas.read_json(lines=True)` reads it directly. Knowing
  which regime you are in is the judgment being demonstrated.
- **Fail-closed on a log write error** (`HTTPException(500)`) — rejected. It inverts the severity:
  an observability fault would suppress a fraud decision the model already produced. The `WARNING`
  is the compensating trace saying a record is missing.
- **An opt-in log-redirect fixture** — rejected in favour of autouse. Opt-in means a future test
  that forgets it appends synthetic rows to the real drift baseline; that bug would surface phases
  later as drift nobody can explain.
- **Patching `src.config.PREDICTIONS_LOG`** — rejected because it silently does nothing. `main`
  imported the name by value, so the fixture patches `src.api.main.PREDICTIONS_LOG`; patching
  `config` would leave tests writing to the real file while appearing to pass.
- **Letting the real `lifespan` run in tests** — rejected. It would make the suite depend on a
  populated `mlflow.db`, which CI will not have.

## Justification

The double reports version **`"7"`** while the live Registry serves `"1"`, so a passing
`model-info` test is itself evidence the suite read the double rather than reaching MLflow.
Isolation was measured, not asserted: `logs/predictions.jsonl` was byte-identical (same md5,
2 lines) before and after `make test`.

Eleven tests cover `/health` loaded and unloaded, `/model-info` loaded and 503, a successful
`/predict`, **two** 422 cases — a *missing* field and a present-but-negative `Amount` — a 503
`/predict`, that a prediction appends exactly one line, that the line matches Phase 8's key
contract, and that a rejected payload logs nothing. The two 422 cases matter separately: the first
proves required-field enforcement, the second proves the *constraint* is enforced on well-formed,
correctly-typed input.

## Trade-offs / consequences

- **Fail-open loses records.** A prediction served during a disk-full window is absent from the
  Phase 8 baseline, and the drift analysis silently sees less traffic than was served.
- **The log grows without bound.** No rotation or retention policy exists; at this scale that is
  acceptable, and it is a Phase 8 concern.
- **`"input"` and `PREDICTIONS_LOG` are now frozen.** Renaming either breaks Phase 8 before it is
  written.
- **The suite proves nothing about the real Registry.** That is the point of the isolation, and it
  is why the phase's Definition of Done is a live `make serve` check rather than a green suite.
