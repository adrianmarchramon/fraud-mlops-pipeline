# Decision 31: `prefect` as a development dependency, not a production one

- **Date:** 2026-08-28
- **Status:** Accepted

## Context

Phase 7 adds the first tool this project uses that is neither imported by `src/` nor invoked by
`dvc.yaml`. `project_context/mlops_phase7.md` says simply `uv add prefect`, which places it in
`[project] dependencies`. The question is whether that is right here, because this repository
ships a container image and the group decides what lands inside it.

## Decision

**`uv add --dev prefect`** — `prefect>=3.8.4` in `[dependency-groups] dev`, alongside `dvc`.

## Alternatives considered

- **A production dependency**, as the reference command does by default and as `mlflow`,
  `xgboost` and `imbalanced-learn` were declared in Phase 2. Rejected, and the analogy is the
  thing to examine rather than the conclusion. Those three are production because **shipped code
  imports them**: `src/api/predict.py` imports `mlflow` to resolve `@production`, and the
  unpickled model needs sklearn and xgboost at inference time. Nothing under `src/` imports
  Prefect. Only `pipelines/` does, and `pipelines/` is not in the image.
- **Deferring the decision** by leaving Prefect uninstalled and documenting flows only —
  rejected as incoherent with a phase whose deliverable is runnable orchestration.

## Justification

**The true analogue is `dvc`, not `mlflow`.** DVC orchestrates the reproducible pipeline, is
executed by `dvc repro` rather than only by a developer's editor, and is nonetheless a dev
dependency — because no module under `src/` imports it. Prefect occupies exactly that position:
a coordination layer above the application, not part of it. Applying the same rule to both keeps
the grouping principled instead of case-by-case.

**The cost is measured, not hypothetical.** Adding Prefect pulled **105 additional packages**
(`apprise`, `asyncpg`, `redis`, `pydocket`, `typer`, …). `docker/Dockerfile:38,49` runs
`uv sync --frozen --no-dev` precisely so pytest, ruff, mypy, dvc and Jupyter stay out of the
runtime image; a production placement would push all 105 into an image whose only job is to load
a model and answer HTTP.

**Nothing is lost locally.** `uv sync` installs the `dev` group by default — which is *why* the
Dockerfile has to opt out explicitly — so `make setup`, `uv run`, and CI's
`uv sync --locked --dev` all get Prefect with no extra flag.

## Trade-offs / consequences

- **If a later phase containerises the orchestrator, this must be revisited** — that image would
  need `--group dev`, or Prefect would have to be promoted. Judged unlikely: Phase 9 deploys the
  *API* to a lightweight PaaS, and the project's no-Kubernetes stance argues against shipping a
  scheduler.
- **The resolution moved one existing pin:** `websockets` was downgraded `17.0.1` → `16.1.1` to
  satisfy Prefect alongside `uvicorn[standard]`. The API test suite passes unchanged, so the
  downgrade is compatible in practice, but it is a real change to a transitive dependency of the
  shipped API and is recorded here rather than left in the lockfile diff alone.
- **CI installs 105 more packages on every run.** Mitigated by `setup-uv`'s cache, but the
  install step is measurably heavier than it was in Phase 6.
- **`pyproject.toml` no longer tells the whole story of what the deployed image contains** — the
  group boundary does. Anyone reading dependencies must read the group, not just the list.
