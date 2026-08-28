# Decision 32: `mypy --strict` extended to `pipelines/`, and the `serve()` cast

- **Date:** 2026-08-28
- **Status:** Accepted

## Context

Phase 6 enabled `mypy --strict` with `files = ["src"]`
([0026](0026-ci-workflow-shape.md)) and deliberately deferred `tests/`. `pipelines/` was never
considered, because it held nothing but two docstring placeholders. Phase 7 turns it into a real
source directory — three modules that import from `src/`, wrap its entry points, and carry the
closed loop's only runtime coupling. Leaving it unchecked would mean "this repository is strictly
typed" stopped being true in the same phase that gave it a new top-level package.

## Decision

**`files = ["src", "pipelines"]`.**

**`pipelines/__init__.py` is added**, making the directory a real package like `src/` and its four
subpackages, rather than a PEP 420 namespace directory.

**`pipelines/serve.py` casts the result of `to_deployment()`:**

```python
training_deploy = cast(
    RunnerDeployment, training_pipeline.to_deployment(name=TRAINING_DEPLOYMENT_NAME)
)
```

`RunnerDeployment` is imported from `prefect.deployments.runner` — it is **not** re-exported from
`prefect.deployments`.

**No `[[tool.mypy.overrides]]` entry was needed for Prefect:** it ships `py.typed`, unlike
sklearn, imblearn and joblib.

## Alternatives considered

- **Leaving the scope at `["src"]`** — rejected. It would exempt the newest code in the repository
  from the standard the previous phase existed to enforce, and `pipelines/` is where the loop's
  silent-failure coupling lives.
- **`# type: ignore[arg-type]` on the `serve()` call** instead of the cast — narrower, but it
  records only that an error was suppressed, not what the sync path actually returns.
- **A per-module override excluding `pipelines/serve.py`** — rejected: it carves a hole in the
  scope this record just widened, and the hole would sit exactly on the file wiring the
  deployments together.
- **Relying on namespace packages** rather than adding `__init__.py` — works (both
  `python -m pipelines.training_pipeline` and `from pipelines.x import y` resolve), but leaves
  `pipelines/` the only source directory in the repo that is not a package. The roadmap's
  directory diagram shows it without one; this is a deliberate, small deviation.

## Justification

The cost was measured before committing to it, not estimated. A realistic `training_pipeline.py`
— real repo imports, `-> ValidationReport`, four decorated tasks and a flow — type-checked clean
under `--strict` on the first attempt. After the change, `uv run mypy` reports
**"Success: no issues found in 22 source files"**, up from 18.

The one genuine failure was `serve()`:

```
error: Argument 1 to "serve" has incompatible type
       "RunnerDeployment | Coroutine[Any, Any, RunnerDeployment]";
       expected "RunnerDeployment"  [arg-type]
```

`Flow.to_deployment()` carries Prefect's `@async_dispatch` sync/async union, and mypy cannot
narrow it in a synchronous context. The `cast` states what the sync path returns — the same
technique `src/models/train.py` already uses for `cast(dict[str, float], metrics)`, where a
TypedDict does not satisfy `SupportsKeysAndGetItem`. Both are cases of the checker being correct
in general and wrong about this call site specifically, which is what `cast` is for.

## Trade-offs / consequences

- **`mypy` is now a merge blocker for orchestration code too.** New flows must be strictly typed
  or CI goes red.
- **The cast is a small, deliberate blind spot.** If a future Prefect release changes what
  `to_deployment()` returns on the sync path, the cast will keep mypy quiet while the code breaks
  at runtime. It is annotated in place with the reason so the next reader can re-test the
  assumption rather than inherit it.
- **`tests/` is still out of scope**, unchanged from [0026](0026-ci-workflow-shape.md) — so
  `tests/test_pipelines.py` is linted and formatted but not type-checked.
- **Import-time coupling:** because `files` now includes `pipelines/`, mypy resolves `src.*` from
  it. Anything that breaks that import path fails type-checking as well as tests.
