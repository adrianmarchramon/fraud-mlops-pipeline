.PHONY: setup lint format test train serve clean

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

serve:        ## Start API (available starting from Phase 4)
	uv run uvicorn src.api.main:app --reload

clean:        ## Clean caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
