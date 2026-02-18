.PHONY: lint format check fix clean test

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

check:
	uv run ruff check src/ && uv run ruff format --check src/

fix:
	uv run ruff check --fix src/ || true
	uv run ruff format src/

clean: fix

test:
	uv run pytest tests/
