# Decision 4: Technology stack summary

- **Date:** 2026-07-09
- **Status:** Accepted

## Context

The stack was chosen during planning (`project_context/`) and is already evidenced in the
repository. This record does **not** introduce new stack decisions — it reconstructs the
existing ones from the code so the README (Phase 0, Step 10) can consume a single summary.
Sources: the placeholder module docstrings from Step 4, `pyproject.toml`,
`.pre-commit-config.yaml`, `Makefile`, and the `project_context/` roadmap.

## Decision — the stack, by lifecycle stage

| Stage | Tool | Evidence in repo |
|-------|------|------------------|
| Environment & deps | **uv** | `uv.lock`, `pyproject.toml`, `Makefile` (`uv sync`, `uv run`) |
| Lint **and** format | **ruff** (not black) | `pyproject.toml` `[tool.ruff]`, `.pre-commit-config.yaml` |
| Pre-commit hooks | **pre-commit** | `.pre-commit-config.yaml` (ruff, whitespace, `check-added-large-files`) |
| Testing | **pytest** | `tests/`, `Makefile` `test` target |
| Data versioning | **DVC** | `data/` git-ignored, roadmap Phase 1 |
| Data validation | **Pandera** | `src/data/validate.py` docstring, roadmap Phase 1 |
| Modelling | **scikit-learn / XGBoost + imbalanced-learn** | `src/models/train.py`, roadmap Phase 2 |
| Experiment tracking & registry | **MLflow** | `src/models/train.py`, `register.py` docstrings, roadmap Phases 2–3 |
| Inference API | **FastAPI + Pydantic** | `src/api/main.py`, `schemas.py` docstrings, roadmap Phase 4 |
| Containerization | **Docker + docker-compose** | `docker/Dockerfile`, `docker/docker-compose.yml` |
| CI/CD | **GitHub Actions** | roadmap Phase 6 |
| Orchestration | **Prefect** | `pipelines/*.py` docstrings, roadmap Phase 7 |
| Monitoring & drift | **Evidently** | `src/monitoring/drift.py` docstring, roadmap Phase 8 |
| Deployment | **Render / Railway / Fly.io / Modal** (lightweight, no Kubernetes) | roadmap Phase 9 |

## Alternatives considered (key ones)

- **black + isort + flake8** vs **ruff** → ruff, one tool for lint + format.
- **pip / poetry / conda** vs **uv** → uv, for speed and a committed lockfile.
- **Kubernetes** vs **lightweight PaaS** → PaaS; k8s is unjustified overhead at this scale and
  reads as poor tool-to-problem judgement.
- Full rationale for each lives in `project_context/mlops_fundamentals.md`.

## Justification

Every choice favours a **reproducible, low-friction, production-shaped** system over maximal
power — consistent with the project's thesis that the infrastructure, not the model, is the
deliverable. Environment-specific values (e.g. `MLFLOW_TRACKING_URI`) are injected via env
vars so the same code runs locally and in containers.

## Trade-offs / consequences

- Python **3.14** is pinned (`.python-version`); the stack resolved cleanly on it, but it is
  new enough that some third-party tools may lag. (Note: `pyproject.toml` still declares
  `requires-python = ">=3.12"` — a known minor inconsistency flagged for a later phase.)
- Committing to this stack now keeps later phases focused on wiring, not re-selection.
