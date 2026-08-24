# Decision 26: CI workflow shape — triggers, check order, and enforced strict typing

- **Date:** 2026-08-24
- **Status:** Accepted

## Context

Phase 6 introduces `.github/workflows/ci.yml`, the first automation in this project that runs
outside the developer's machine. Three shape decisions had defensible alternatives, and the
reference material for the phase proposed a different answer to two of them.

Separately, `mypy` had been declared a standard since Phase 0 — the header docstring of every
module in `src/` says *"Strict typing (mypy --strict as reference; avoid unjustified `Any`)"* —
and had been a real dev dependency since then (`mypy>=2.2.0`, locked at `2.2.0`). But nothing
ever ran it: no `[tool.mypy]` section, no pre-commit hook, no Makefile target. The standard
existed only as prose.

## Decision

**Triggers are `push: branches: [main]` + `pull_request:`**, not the reference's
`on: [push, pull_request]`. A bare pair runs the workflow **twice** for every push to a branch
with an open pull request — double the minutes, duplicate checks on one commit.

**Checks run cheapest-first**: `ruff check` → `ruff format --check` → `mypy` → `pytest`. Each
step is roughly an order of magnitude more expensive than the one before, so a misplaced import
costs seconds of feedback rather than minutes.

**Installation is `uv sync --locked --dev`**, never a bare `uv sync`.

**`mypy --strict` is now enforced in CI**, configured in `pyproject.toml` rather than as CLI
flags, so `uv run mypy` means exactly the same thing on a laptop and on a runner:

```toml
[tool.mypy]
python_version = "3.12"
files = ["src"]
strict = true

[[tool.mypy.overrides]]
module = ["sklearn.*", "imblearn.*", "joblib"]
ignore_missing_imports = true
```

`pandas-stubs` and `types-PyYAML` were added as dev dependencies so those two libraries are
genuinely type-checked; only the three that ship no inline types and publish no stub package are
silenced, and they are silenced **by name**.

**Raw commands, not `Makefile` targets.** `make lint` matches `uv run ruff check .` exactly, but
there is no `format-check` target, no `mypy` target, and `make format` runs `ruff format .`
*without* `--check` — it would silently fix the very problem CI exists to detect.

## Alternatives considered

- **`on: [push, pull_request]` as written in the reference** — rejected for the double-run. The
  reference's stated purpose (verify every proposed change) is fully served by our shape, and
  `pull_request` is the more truthful signal anyway: it evaluates the ephemeral merge commit,
  i.e. the code that would actually land, rather than the branch head in isolation.
- **Bare `uv sync`** — rejected decisively. It *repairs* a stale `uv.lock` in place and carries
  on, so CI would pass on a resolution nobody committed and leave the mismatch for the next
  clone — or for `docker/Dockerfile`, which runs `uv sync --frozen` and would fail the image
  build *after* CI had already gone green. `--locked` refuses to re-resolve and exits non-zero.
- **Omitting `mypy`, as the reference `ci.yml` does** — rejected. It would leave an installed
  tool unexecuted and reduce a five-times-repeated standard to a docstring comment, in the very
  phase dedicated to automating quality enforcement.
- **Blanket `ignore_missing_imports = true`** — rejected in favour of named overrides. A blanket
  ignore silences every untyped import, including libraries that gain stubs later. Measured
  difference: 26 errors under `--strict` → 4 either way, but the named form keeps `pandas` and
  `yaml` actually checked.
- **Including `tests/` in the mypy scope** — deferred. It costs 42 errors across four Phase-1/3/4
  test files (23 of them missing fixture annotations); a separate piece of work, not a Phase 6
  deliverable.

## Justification

Enabling `mypy --strict` surfaced **26 errors, 22 of which were missing-stub noise**. The four
real ones were fixed in `src/data/preprocess.py` and `src/models/train.py`, and one of them was
not a typing nicety at all:

```
src/models/train.py:471: error: Item "None" of "ActiveRun | None" has no attribute "info"
```

`run_id` was read from `mlflow.active_run().info.run_id`, which returns `ActiveRun | None` — a
latent `AttributeError` on any path where the run failed to start. It now reads the binding
yielded by the `start_run()` context manager, which cannot be `None` inside the block. Finding a
real bug on the first strict run is itself the argument for the step.

## Trade-offs / consequences

- **A push to a feature branch with no open pull request runs nothing.** That is the cost of
  avoiding the double-run, and it is deliberate: with `main` protected, work reaches `main` only
  through a pull request, so any change that matters is covered.
- **The DoD's literal wording is not met.** `mlops_phase6.md` says CI runs "on every push and
  pull request"; ours restricts the push half to `main`. Recorded in
  [0030](0030-phase-6-verification-and-dod-deviations.md) rather than hidden.
- **`mypy` is now a merge blocker.** New code in `src/` must be strictly typed or CI goes red.
  That is the intent, but it raises the cost of a quick fix.
- **Two files behind closed phase tags were modified** (`preprocess.py`, `train.py`). Also
  recorded in [0030](0030-phase-6-verification-and-dod-deviations.md).
