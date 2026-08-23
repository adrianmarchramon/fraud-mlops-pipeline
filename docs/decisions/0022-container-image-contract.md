# Decision 22: The container image contract — Python 3.12, non-root, and a body-matching health check

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

Phase 5 packages the inference API so it runs identically anywhere.
`project_context/mlops_generalRoadmap.md:252` prescribes "a lightweight image
(`python:3.11-slim`), properly cached layers, and a non-root user", and
`project_context/mlops_phase5.md` §"Step 1" shows a two-stage `Dockerfile` whose `HEALTHCHECK`
is a bare `curl -f` against `/health`.

Two of those specifics collide with decisions this repository has already made. The project pins
Python **3.12** (`.python-version`, `requires-python`, ruff's `target-version`), and ADR 0020
made `/health` answer **HTTP 200 even when no model is loaded**, reporting the degraded state in
the body rather than raising.

## Decision

**Two stages, both `FROM python:3.12-slim`.** `uv` is copied from its own published image at a
pinned version (`COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/`), not installed by a
piped script.

**Dependencies before source.** The builder runs
`uv sync --frozen --no-install-project --no-dev` with `uv.lock` and `pyproject.toml`
*bind-mounted*, so the application source is not part of that layer; `COPY . /app` and a second
`uv sync --frozen --no-dev` follow. Editing `src/` therefore reuses the cached dependency layer.

**The runtime stage starts clean** — no uv, no build cache, no dev dependencies — and adds only
`curl`, solely because the health check needs it.

**Non-root, with one writable directory.** `useradd --create-home appuser` then `USER appuser`.
Because `WORKDIR` creates `/app` as root and `COPY --chown` applies only to the entries it
writes, an explicit `RUN mkdir -p /app/logs && chown appuser:appuser /app/logs` hands over the
log directory and nothing else. `/app` stays root-owned.

**The health check matches the response body, not just the status code:**

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"status":"ok"' || exit 1
```

**The model is not baked in.** It is pulled at runtime through the `@production` alias, so the
same image serves whatever version holds that alias.

## Alternatives considered

- **`python:3.11-slim`, as the roadmap prescribes** — rejected. `uv sync --frozen` refuses this
  project's lock on 3.11, so the build would fail outright. The roadmap predates the 3.12 pin.
- **Installing uv with `curl … | sh`** — rejected. It puts an unpinned network fetch inside the
  build; every other dependency here is pinned, and this one should be too.
- **A single-stage image** — rejected. uv, its cache and the build toolchain would all ship to
  production for no runtime benefit.
- **Running as root** — rejected. An application compromise would then hold root inside the
  container, which is a materially shorter path to host escape.
- **`HEALTHCHECK CMD curl -f …/health`, as the reference material shows** — rejected, and this is
  the consequential one. Since `/health` returns 200 with `{"status":"no_model"}`, `curl -f`
  reports a container that never reached MLflow as perfectly healthy. Matching the body is what
  makes the signal mean "loaded and serving" — which is what Compose's `service_healthy` and
  Phase 9's platform actually consume.
- **`chown -R appuser /app`** — rejected. It would let the running service rewrite its own source
  and virtualenv, discarding most of the benefit of dropping root.
- **Baking the model into the image** — rejected. It would tie one image to one model version and
  force a rebuild on every promotion, contradicting ADR 0015.

## Justification

Verified against the built image and a running container: `id` reports `uid=1000(appuser)`;
`docker history` puts **1.47 GB** in the venv layer with every other layer at or below 87 MB;
inside the container `.venv` is **1.4 G** against **360 K** of `src/`; `pytest`, `ruff`, `mypy`,
`dvc` and `jupyter` are all absent, confirming `--no-dev`.

The health-check decision was not theoretical. Commit `24d8a07` records the case it was designed
for: the API could not reach the Registry, came up in `no_model`, and answered `/health` with
HTTP 200 — the container was correctly marked **unhealthy** only because the check inspects the
body. A `curl -f` check would have reported it healthy and the fault would have surfaced later,
as missing predictions.

The `/app/logs` ownership line exists for the symmetric reason, recorded in commit `854484a`:
without it `log_prediction()`'s `mkdir` raised `EACCES`, the `except OSError` branch swallowed
it by design, and every prediction was dropped with a `WARNING` while `/predict` still answered
200 — an endpoint that looked flawless over an empty log.

## Trade-offs / consequences

- **`nvidia-nccl-cu12` ships in a CPU-only image.** It is a transitive dependency of
  `xgboost 3.3.0` and occupies **400 MB**, roughly 29% of the virtualenv. Removing it means
  constraining dependency resolution in `uv.lock`, which also affects training; left as a
  deliberate follow-up rather than a Phase 5 change.
- **The health check couples container health to Registry reachability.** A running, correctly
  built container reports unhealthy whenever MLflow is unreachable. That is intended — it is the
  signal an orchestrator should act on — but it means "unhealthy" does not always mean "broken
  image".
- **`/app` being root-owned means any future write path needs its own explicit `chown`.** The
  failure mode is silent by design, since the application swallows `OSError` on the log path.
- **Dependency changes invalidate the expensive layer.** That is the correct trade: source edits
  stay cheap, dependency edits pay the full resolve.
