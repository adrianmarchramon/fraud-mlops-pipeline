# syntax=docker/dockerfile:1
#
# Multi-stage image for the inference API (Phase 5).
#
# The model is pulled at runtime from the MLflow service through the
# @production alias, so this same image serves whatever version currently holds
# that alias — promoting a model needs no rebuild
# (docs/decisions/0015-packaged-model-contract.md). That remains the behaviour
# everywhere this image is run against a Registry, Compose included.
#
# Phase 9 adds one exception without changing any of the above. `COPY . /app`
# now also carries deploy/model/, an exported copy of the @production artifact
# (~1.4 MB), because the public free-tier deployment has no MLflow to ask. It
# is inert unless MODEL_PATH is set, which happens only there
# (docs/decisions/0042-bundled-model.md).
#
# Python is pinned to 3.12 to match .python-version, requires-python and ruff's
# target-version. The reference material shows 3.11; on 3.11 `uv sync --frozen`
# would refuse this project's lock.

# ===== Stage 1: builder =====
# Resolves the environment. Needs uv and its cache; none of that reaches the
# final image.
FROM python:3.12-slim AS builder

# uv comes from its own published image rather than an install script: no
# curl|sh in the build, and the version is pinned like every other dependency
# in this project.
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Dependencies ONLY. uv.lock and pyproject.toml are bind-mounted rather than
#    copied, so the application source is not yet part of this layer: editing
#    src/ leaves this cached and rebuilds only the cheap tail below. Reversing
#    these two steps would re-resolve the whole ML dependency tree on every
#    code change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2) Now the source. --frozen pins the exact versions in uv.lock; --no-dev
#    drops pytest, ruff, mypy, dvc and Jupyter, which have no place in a
#    runtime image.
#    This project declares no [build-system], so uv treats it as a virtual
#    project and this second sync installs no package of its own — src/ is
#    importable via PYTHONPATH below. Kept so the step starts working if the
#    project ever becomes installable.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ===== Stage 2: runtime =====
# Starts clean. Carries no uv, no build cache, no dev dependencies.
FROM python:3.12-slim

# curl is the only addition, and only because HEALTHCHECK needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Containers run as root by default: an application compromise would then hold
# root inside the container, which is a far shorter path to host escape.
RUN useradd --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app

# WORKDIR created /app as root, and --chown above only applies to the entries
# COPY writes, never to a pre-existing destination directory. So appuser could
# read everything and create nothing: log_prediction()'s mkdir raised EACCES,
# its except OSError branch swallowed it, and every prediction was dropped with
# a WARNING while /predict still answered 200.
#
# That would have left Phase 8 with an empty baseline, since
# 0021-prediction-log-and-api-tests.md makes this file the contract monitoring
# reads. Only the log directory is handed over: /app stays root-owned so the
# service cannot rewrite its own code, which is the point of dropping root.
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# PATH puts the resolved venv first; PYTHONPATH is what actually makes `src`
# importable, since the project itself is not installed as a package.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

USER appuser

EXPOSE 8000

# /health answers HTTP 200 even when no model is loaded — it reports the
# degraded state in the body rather than raising, by design
# (docs/decisions/0020-api-structure-and-state.md). So `curl -f` alone would
# mark a container that never reached MLflow as healthy. Matching the body is
# what makes this signal mean "loaded and serving", which is what Compose's
# service_healthy and Phase 9's platform actually need.
# The body is pinned by tests/test_api.py::test_health_reports_ok_when_a_model_is_loaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"status":"ok"' || exit 1

# 0.0.0.0 so the port is reachable from outside the container. No --reload:
# that is a development convenience and would watch for edits that cannot happen
# in an immutable image.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
