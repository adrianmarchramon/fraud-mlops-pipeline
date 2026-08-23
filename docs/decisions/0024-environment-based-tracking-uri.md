# Decision 24: The MLflow address comes from the environment, with a working local default

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

ADR 0011 put MLflow on a SQLite backend from day one and explicitly deferred to Phase 5 the
question of how the same code would reach a *containerized* MLflow instead of a local file.
`project_context/mlops_phase5.md` §"Step 3" states the requirement plainly: the MLflow address
must be configurable, so that one build behaves correctly in every environment.

The constraint that makes this non-trivial is that the *same* `train.py` and `register.py` must
populate either backend without modification — otherwise the containerized registry would need
its own code path, and the two would drift.

## Decision

One line in `src/config.py`, the project's single configuration module:

```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
```

The default preserves the pre-Phase-5 behaviour exactly: with the variable unset, everything
resolves to the local SQLite file and `make train`, `make register` and `make serve` work as they
did in Phases 2–4.

The value reaches the `api` container through **`env_file`**, not a literal `environment:` entry:

```yaml
env_file:
  - .env.example
```

`docker/.env.example` is tracked and contains the single line
`MLFLOW_TRACKING_URI=http://mlflow:5000`. `docker/.env` is git-ignored (`.gitignore:38`) as the
local override.

## Alternatives considered

- **Hardcoding a different URI per environment** — rejected outright. It is the specific failure
  this project's "centralized, versioned config" rule exists to prevent, and it would guarantee
  that local and containerized runs diverge.
- **Two config modules, or a `--tracking-uri` flag on every entry point** — rejected. Both push
  the environment distinction into the call sites, so every script and every future caller has
  to know which world it is in. An environment variable keeps that knowledge in exactly one
  place.
- **No default, failing loudly when unset** — rejected. It would break `make train` and
  `make serve` outside containers for no gain, and Phases 2–4 already established the local
  SQLite path as the normal development mode.
- **A literal `environment:` block in `docker-compose.yml` instead of `env_file`** — rejected.
  A tracked `.env.example` gives a discoverable, documented template and an obvious override
  point (`docker/.env`) without editing the compose file.
- **Committing a real `.env`** — rejected. Nothing secret lives here today, but establishing the
  habit now is what keeps Phase 6's CI credentials out of the repository later.

## Justification

This is the decision that makes the containerization work at all, and it was verified end to
end: `train.py` and `register.py` ran **unmodified**, prefixed only with
`MLFLOW_TRACKING_URI=http://localhost:5000`, and populated the containerized registry —
`Successfully registered model 'fraud-detector'` / `Created version '1'`.

The two worlds were confirmed to be genuinely separate rather than merely differently addressed.
Resolving the constant on the host with no variable set prints `sqlite:///mlflow.db`; reading it
inside the running API container prints `http://mlflow:5000`. The host's `mlflow.db` held md5
`2589f47305093998710b61926f6cb78f` before and after every containerized operation, and its
registry — versions `[3, 2, 1]`, `@production` → v1 from run `efe7826f…` — is a different object
from the container's single version 1 from run `55d6691e…`.

## Trade-offs / consequences

- **An unset or misspelled variable fails silently into the local default.** A typo in the
  variable name yields a service that starts, reports healthy against the *wrong* registry, and
  gives no indication anything is off. The default is a convenience with a real edge.
- **`.env.example` is a template, not a validated schema.** Nothing checks that a local
  `docker/.env` contains the keys the system needs, or only those.
- **The pattern is now the precedent.** Anything that differs between local and containerized
  execution should follow this shape rather than inventing a second mechanism — which is exactly
  what Phase 6's CI and Phase 9's deployment target will need.
