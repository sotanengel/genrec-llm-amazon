.PHONY: install lock lock-upgrade lint format format-check typecheck \
	check-model-configs test-fast test-cpu test-precommit test-slow test-gpu \
	data-prepare data-stats eval-baselines report verbalize-samples encode validate-sasrec

# Dev tooling lives in [dependency-groups] (pyproject.toml), synced by `uv sync`
# by default (the "dev" group). This is what makes `uv run mypy`/`uv run pytest`
# work out of the box instead of silently missing tools or re-resolving the
# whole (torch-sized) dependency tree — see issue #13, P2.
install:
	@command -v uv >/dev/null 2>&1 || pip install uv
	uv sync
	uv run pre-commit install

lock:
	uv lock

lock-upgrade:
	uv lock --upgrade

# Every check below delegates to scripts/dev.py, which holds the single
# definition of each command. `make` does not exist on Windows (where this repo
# is authored) but pre-commit still has to run there, so the hooks call dev.py
# directly rather than going through make — see scripts/dev.py's docstring.
#
# CI's `pre-commit` job runs `pre-commit run --all-files`, which invokes the
# ruff-pre-commit hooks directly (not these targets) so that ruff only ever
# runs from one place/version; tests/test_toolchain_consistency.py enforces
# that the pinned ruff version matches the hook's rev.
DEV := uv run --frozen python scripts/dev.py

lint:
	$(DEV) lint

format:
	$(DEV) format

format-check:
	$(DEV) format-check

typecheck:
	$(DEV) typecheck

check-model-configs:
	$(DEV) check-model-configs

test-fast:
	$(DEV) test-fast

test-cpu:
	$(DEV) test-cpu

test-precommit:
	$(DEV) test-precommit

test-slow:
	$(DEV) test-slow

test-gpu:
	$(DEV) test-gpu

data-prepare:
	uv run --frozen python -m genrec_lite data prepare --dataset amazon_video_games

data-stats:
	uv run --frozen python -m genrec_lite data stats --dataset amazon_video_games

eval-baselines:
	uv run --frozen python -m genrec_lite eval run --exp m1_baselines

report:
	uv run --frozen python -m genrec_lite report build --exp m1_baselines

verbalize-samples:
	uv run --frozen python -m genrec_lite verbalize render --dataset amazon_video_games --verbalizer v1_full --n 20

encode:
	uv run --frozen python -m genrec_lite encode run --dataset amazon_video_games --model qwen3-1.7b-base --verbalizer v1_full

validate-sasrec:
	uv run --frozen python scripts/validate_sasrec_literature.py --dataset amazon_video_games_literature
