.PHONY: setup lint format test train register serve prefect-server prefect-serve prefect-train clean

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
# falls back to an ephemeral one (PREFECT_SERVER_EPHEMERAL_ENABLED is true by
# default): the flow still SUCCEEDS, but on a throwaway database destroyed when
# the process exits, so the run never reaches the dashboard and orchestration
# silently stops being observable. Override for a non-local server with
# `make prefect-serve PREFECT_API_URL=http://host:4200/api`.
PREFECT_API_URL ?= http://localhost:4200/api

prefect-server: ## Start the local Prefect server and dashboard on :4200 (Phase 7)
	uv run prefect server start

prefect-serve:  ## Serve both flow deployments: on-demand training + daily monitoring (Phase 7)
	PREFECT_API_URL=$(PREFECT_API_URL) uv run python -m pipelines.serve

prefect-train:  ## Run the training flow against the local Prefect server (Phase 7)
	PREFECT_API_URL=$(PREFECT_API_URL) uv run python -m pipelines.training_pipeline

clean:        ## Clean caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
