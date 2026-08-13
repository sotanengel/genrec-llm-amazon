.PHONY: install lint test-fast test-slow data-prepare data-stats

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	mypy --strict src

test-fast:
	ruff check src tests
	mypy --strict src
	pytest -x -n auto --timeout=10 tests/ -m "not slow"

test-slow:
	pytest tests/ -m slow --timeout=600

data-prepare:
	python -m genrec_lite data prepare --dataset amazon_video_games

data-stats:
	python -m genrec_lite data stats --dataset amazon_video_games
