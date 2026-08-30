.PHONY: setup lint format test train register serve prefect-server prefect-serve prefect-train prefect-monitor simulate-drift export-model clean

setup:        ## Install dependencies and configure pre-commit
	uv sync
	uv run pre-commit install

lint:         ## Lint code with ruff
	uv run ruff check .

format:       ## Format code with ruff
	uv run ruff format .

test:         ## Run tests
	uv run pytest

train:        ## Train model (available starting from Phase 2)
	uv run python -m src.models.train

register:     ## Register best run in the Model Registry, promote if better (Phase 3)
	uv run python -m src.models.register

serve:        ## Start API (available starting from Phase 4)
	uv run uvicorn src.api.main:app --reload

# Where the flows and the Prefect CLI look for the server. Without it Prefect
# starts a temporary one instead (PREFECT_SERVER_EPHEMERAL_ENABLED is true by
# default). The run is still recorded -- that temporary server shares the same
# ~/.prefect/prefect.db -- but it is not visible LIVE in an already-open
# dashboard, each invocation pays seconds of startup, two API servers end up
# writing one SQLite file, and deployments get registered with nothing serving
# them. Override for a non-local server with
# `make prefect-serve PREFECT_API_URL=http://host:4200/api`.
PREFECT_API_URL ?= http://localhost:4200/api

prefect-server: ## Start the local Prefect server and dashboard on :4200 (Phase 7)
	uv run prefect server start

prefect-serve:  ## Serve both flow deployments: on-demand training + daily monitoring (Phase 7)
	PREFECT_API_URL=$(PREFECT_API_URL) uv run python -m pipelines.serve

prefect-train:  ## Run the training flow against the local Prefect server (Phase 7)
	PREFECT_API_URL=$(PREFECT_API_URL) uv run python -m pipelines.training_pipeline

prefect-monitor: ## Trigger the monitoring deployment now instead of waiting for 06:00 (Phase 8)
	PREFECT_API_URL=$(PREFECT_API_URL) uv run prefect deployment run monitoring-pipeline/daily

simulate-drift: ## Send a deliberately shifted batch to the running API (Phase 8, Step 7)
	uv run python -m scripts.simulate_drift

# Refreshes the copy of the model that ships inside the public image. Needed
# after any promotion that should reach the public demo: the bundled artifact
# is frozen at build time, so a promotion alone changes nothing there until the
# export is re-run, committed and redeployed (docs/decisions/0042-bundled-model.md).
export-model: ## Export the @production model into deploy/ for the public image (Phase 9)
	uv run python -m scripts.export_model

clean:        ## Clean caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
