# Decision 23: Compose topology — two services, a readiness gate, and MLflow's Host allowlist

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

`project_context/mlops_generalRoadmap.md:253` asks for "a `docker-compose.yml` that spins up the
API + MLflow + (optional) the logging database together".
`project_context/mlops_phase5.md` §"Step 4" shows two services with a plain
`depends_on`, and notes only in passing that "in a more demanding system, you would add a
`healthcheck` … For this project, the basic ordering is usually sufficient."

Both points needed resolving against what this repository had already decided. ADR 0021 froze
`logs/predictions.jsonl` as the contract Phase 8 reads, and ADR 0020 established that the API
resolves `@production` **exactly once** during `lifespan` and never retries.

## Decision

**Two services, not three.** `mlflow` and `api`. No logging database.

**Pinned image tags**, never `:latest`: `ghcr.io/mlflow/mlflow:v3.14.0`, matching the MLflow
version in `uv.lock` so the server and the client library in the API agree.

**A named volume, `mlflow-data`, mounted at `/mlflow`**, holding both the SQLite backend store
(`--backend-store-uri sqlite:////mlflow/mlflow.db`) and the artifact root
(`--artifacts-destination /mlflow/artifacts --serve-artifacts`).

**`mlflow` carries a real health check**, written in Python rather than `curl`:

```yaml
test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:5000/health')\" || exit 1"]
```

**`api` waits on readiness, not merely on start order:**

```yaml
depends_on:
  mlflow:
    condition: service_healthy
```

**`MLFLOW_SERVER_ALLOWED_HOSTS` is set explicitly** to
`mlflow:5000,localhost:5000,127.0.0.1:5000`.

**The API addresses MLflow by service name** (`http://mlflow:5000`), resolved by Compose's
internal DNS. No IP address appears anywhere.

## Alternatives considered

- **Adding the optional logging database the roadmap mentions** — rejected. Predictions are
  appended to `logs/predictions.jsonl`, whose path and record shape ADR 0021 froze as a contract
  with Phase 8. Swapping the sink now would break a phase that has not been written yet.
- **A bare `depends_on: [mlflow]`, as the reference material shows** — rejected. It orders
  container *starts*, not readiness. Because `lifespan` resolves the alias once and never
  retries, losing that race does not crash the API; it strands it in `no_model` until someone
  restarts it. A silent failure is worse than a loud one.
- **A `curl`-based health check for `mlflow`** — rejected. The official MLflow image is not
  guaranteed to ship an HTTP CLI, but it necessarily has a Python interpreter.
- **Leaving `MLFLOW_SERVER_ALLOWED_HOSTS` unset** — rejected, because it does not work. See below.
- **Setting it to `*`** — rejected. It switches the protection off entirely rather than naming who
  may reach the server.
- **A bind mount instead of a named volume** — rejected. It would place registry state inside the
  repository working tree, where `.gitignore` and `.dockerignore` already work hard to keep it
  out.

## Justification

The readiness gate was observed doing its job: bringing the stack up prints
`Container docker-mlflow-1 Waiting` → `Healthy` → `Container docker-api-1 Starting`, with
`mlflow` reaching `healthy` in roughly 24 s and `api` shortly after. The volume outlives its
containers: after `docker compose down`, reading the backend store from a throwaway container
returned `registered_models: [('fraud-detector',)]` and
`aliases: [('fraud-detector', 'production', 1)]`.

The Host allowlist is not defensive boilerplate; it is required, and commit `24d8a07` records
why. MLflow 3.x validates the `Host` header of every request as DNS-rebinding protection, and its
default allowlist covers loopback and the RFC 1918 ranges but **not container names**. The API's
`http://mlflow:5000` was rejected with `403 Invalid Host header - possible DNS rebinding attack
detected`, so the service started with no model — while host-side training over
`http://localhost:5000` worked fine, which is what made the fault so easy to miss. It was
invisible to the health check too, because MLflow exempts `/health` and `/version` from that
validation: the service reported healthy while every registry call failed.

Because MLflow resolves the setting as *from-environment **or** defaults*, providing it
**replaces** the default list rather than extending it. Both loopback forms are therefore named
explicitly — the health check and host-side `make train` against the published port depend on
them.

## Trade-offs / consequences

- **Compose derives the project name from the file's directory**, so the deployed names are
  `docker-api`, `docker_mlflow-data` and `docker_default`. Renaming the project later would
  orphan the populated volume and require repopulating the registry.
- **The allowlist is a maintenance point.** Renaming the service, changing its port, or adding a
  second client that addresses MLflow by another name all require editing this list, and the
  failure mode is a 403 that the health check cannot see.
- **Both ports are published on all interfaces.** Acceptable for local development; a deployment
  target would want MLflow reachable only on the internal network.
- **One volume holds both the database and the artifacts.** Simple, and it makes "back up the
  registry" a single operation — but it also means one careless `docker volume rm` takes both.
