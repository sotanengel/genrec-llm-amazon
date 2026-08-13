.PHONY: install lint test-fast test-slow data-prepare data-stats eval-baselines report verbalize-samples encode validate-sasrec

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

eval-baselines:
	python -m genrec_lite eval run --exp m1_baselines

report:
	python -m genrec_lite report build --exp m1_baselines

verbalize-samples:
	python -m genrec_lite verbalize render --dataset amazon_video_games --verbalizer v1_full --n 20

encode:
	python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base --verbalizer v1_full

validate-sasrec:
	python scripts/validate_sasrec_literature.py --dataset amazon_video_games_literature
