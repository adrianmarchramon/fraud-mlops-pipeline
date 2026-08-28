.PHONY: setup lint format test train register serve prefect-server prefect-serve clean

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

prefect-server: ## Start the local Prefect server and dashboard on :4200 (Phase 7)
	uv run prefect server start

prefect-serve:  ## Serve both flow deployments: on-demand training + daily monitoring (Phase 7)
	uv run python -m pipelines.serve

clean:        ## Clean caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
