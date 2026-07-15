# Decision 11: MLflow tracking backend — SQLite from day one

- **Date:** 2026-07-15
- **Status:** Accepted

## Context

Phase 2 introduces experiment tracking. MLflow can persist runs either to a **flat directory**
(the default `./mlruns` file store, needing no configuration) or to a **database backend** via a
SQLAlchemy URI. The choice looks inconsequential while only tracking runs — both record
parameters, metrics and artifacts identically — but it is not: the **Model Registry**, which
Phase 3 is built around, **requires a database backend**. The file store cannot host it.

Evidence in repo: `src/config.py` sets `MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"`, with a
comment recording this exact rationale. `mlflow.db` exists (~848 KB) and its schema contains the
`registered_models`, `model_versions` and `registered_model_aliases` tables — the registry
machinery is already present and unused.

## Decision

Track to **SQLite** (`sqlite:///mlflow.db`) from the first run of Phase 2, rather than starting
on the file store and migrating when Phase 3 needs the registry. `mlflow.db` is git-ignored;
artifacts continue to live in `mlruns/`, which the backend references but does not contain.

## Alternatives considered

- **Flat `mlruns/` file store** — rejected: zero-config, but a dead end. Phase 3 would force a
  backend migration, and re-homing existing run history mid-project is exactly the kind of
  avoidable rework the roadmap's phase gating exists to prevent. The cost of choosing correctly
  now is one line of config.
- **PostgreSQL** — rejected: the correct answer at team scale, but it means running and
  containerizing a database server for a single-author portfolio project. Same judgement as the
  "no Kubernetes" call in [0004](0004-stack-summary.md): reach for the heavyweight option only
  when the problem is heavyweight.
- **A remote/hosted tracking server** — deferred, not rejected: relevant once Phase 5 puts the
  API in a container that must reach a shared MLflow service.

## Justification

SQLite is the smallest thing that satisfies both clocks: it is a file (no server, no ops burden,
nothing to start before `make train`), yet it is a real SQL backend and therefore
registry-capable. Verified during the Phase 2 closure: `uv run mlflow ui --backend-store-uri
sqlite:///mlflow.db` starts and serves the `fraud-detection` experiment over HTTP, and the
registry tables are in place awaiting Phase 3.

## Trade-offs / consequences

- **Single-writer.** SQLite serializes writes, so genuinely concurrent training runs would
  contend. Irrelevant at this scale (runs are sequential), but it is the first thing that breaks
  if training is ever parallelized or moved behind a shared service.
- **The run history is local and unshared.** `mlflow.db` and `mlruns/` are both git-ignored (data
  and artifacts never go in Git), so a fresh clone starts with an **empty** experiment history and
  rebuilds it via `make train` / `dvc repro`. The *reproducibility* of a run is guaranteed by DVC
  and `params.yaml`, not by shipping the tracking store around.
- **The URI is currently a literal**, not read from the environment. Phase 5 will need it
  overridable (env var with this value as the default) so a containerized service can point at an
  MLflow service instead of a local file. Flagged here rather than pre-built, since nothing today
  can exercise it.
