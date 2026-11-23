.PHONY: install install-sumo lint format test test-fast

install:
	pip install --upgrade pip && pip install -e ".[dev]"

# SUMO binaries/tools as a wheel (alternative to a system install)
install-sumo:
	pip install -e ".[dev,sumo]"

lint:
	ruff check src tests scripts
	ruff format --check src tests

format:
	ruff check --fix src tests scripts
	ruff format src tests

test:
	pytest

# skip the end-to-end SUMO runs
test-fast:
	pytest -m "not sumo"
