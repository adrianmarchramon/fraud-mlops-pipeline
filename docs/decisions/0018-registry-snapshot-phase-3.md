# Decision 18: Record the Registry state at Phase 3 closure

- **Date:** 2026-08-22
- **Status:** Accepted (point-in-time snapshot, not an ongoing claim)

## Context

Everything else this project asserts is reproducible from tracked files: code from Git, data and
the pipeline from DVC, hyperparameters from `params.yaml`. The Model Registry is the exception.
It lives in `mlflow.db`, which [0011](0011-mlflow-sqlite-backend.md) records as git-ignored, and
a registered version with an alias is by nature a **mutable side effect** of running
`register.py`, not a deterministic output of a DVC stage. Nothing in version control would
otherwise record which model was actually in production on the day this phase closed.

## Decision

Record the Registry state at the moment of Phase 3 closure, as a dated snapshot.

**As of 2026-08-22, immediately before the `phase-3-complete` tag:**

| Version | `pr_auc` tag | `threshold` tag | Alias | Registration run |
|---|---|---|---|---|
| **v1** | `0.8759962787477742` | `0.03` | **`@production`** | `efe7826fc4b44736ac4903c9f3470043` |
| v2 | `0.7249139606556327` | `0.03` | — | `baf4be8e4e394afe8bc49c1b6e9186d9` |
| v3 | `0.7249139606556327` | `0.03` | — | `852bdd7709bf4f49a8e51ff4ca4805a6` |

`models:/fraud-detector@production` resolves to **v1**, the XGBoost model of
[0013](0013-winning-model-xgboost.md) at the cost-optimal threshold of
[0014](0014-cost-optimal-threshold.md). v1 is the highest `pr_auc` of the three, verified by
explicit comparison rather than inferred from the alias.

v2 and v3 are the logistic-regression baseline, kept on purpose: v2 was promoted as the
bootstrap case and later superseded, v3 was registered and **refused** promotion. They are the
audit trail proving the gate in [0016](0016-promotion-quality-gate.md) compares and blocks, so
deleting them would delete the evidence.

## Alternatives considered

- **No snapshot** — rejected. It would leave a gap precisely where the system is least
  reproducible, and "which model was live when we closed Phase 3?" would become unanswerable
  from the repository alone.
- **Committing `mlflow.db`** — rejected. It is a mutable binary that grows with every run;
  [0011](0011-mlflow-sqlite-backend.md) already establishes that run history is not shipped in
  Git, and reproducibility is guaranteed by DVC and `params.yaml`, not by moving the tracking
  store around.
- **Putting the snapshot in the README** — rejected. Per the Phase 2 closure precedent, concrete
  results belong in Phase 9; the README must not carry values that go stale on the next
  promotion.

## Trade-offs / consequences

- **This record goes out of date by design.** The next `make register` that promotes — Phase 6's
  CI/CD, or any retraining — supersedes it. Read it as "the state on 2026-08-22", never as
  "the current state"; `git log` on this file gives the timestamp.
- **It is not a backup.** Losing `mlflow.db` loses the artifacts; this table records only what
  was where.
- **Version numbering does not match registration order semantically.** v1 was registered before
  the demonstration sequence began (by a direct `python -m src.models.register` run that
  registered a version and then failed at promotion — the bug recorded in
  [0016](0016-promotion-quality-gate.md)), so the bootstrap promotion landed on v2 and the
  comparison promotion moved the alias back to v1. The three gate branches were all exercised;
  only the numbering reads oddly.
- **The `threshold` tag is uniform at `0.03` across all versions** because `params.yaml` did not
  change during Phase 3. Each value was read from that version's own artifact, not assumed.
