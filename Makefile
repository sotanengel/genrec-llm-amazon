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

# CI's `pre-commit` job runs `pre-commit run --all-files`, which invokes the
# ruff-pre-commit hooks directly (not this target) so that ruff only ever
# runs from one place/version. This target exists for local convenience and
# is guaranteed to use the *same* pinned ruff version
# (tests/test_toolchain_consistency.py enforces that pin equality).
lint:
	uv run --frozen ruff check src tests scripts

format:
	uv run --frozen ruff format src tests scripts

format-check:
	uv run --frozen ruff format --check src tests scripts

# mypy --strict covers src and scripts, which are clean today. tests/ is
# intentionally excluded: enabling --strict there surfaces ~100 pre-existing
# type errors in test files this work unit does not own (see issue #13
# report). Flagged for a follow-up unit to either fix or scope down.
typecheck:
	uv run --frozen mypy --config-file=pyproject.toml --strict --explicit-package-bases src scripts

check-model-configs:
	uv run --frozen python scripts/check_model_configs.py configs/model/llm

# Fast loop: excludes slow tests. Network tests (see the `network` marker in
# pyproject.toml) run unless HF_HUB_OFFLINE=1 / GENREC_NO_NETWORK=1 is set.
test-fast:
	uv run --frozen pytest -x -n auto --timeout=60 tests/ -m "not slow"

# CPU-only sweep: excludes slow and gpu explicitly (gpu also auto-skips via
# conftest.py when torch.cuda.is_available() is False, this just documents
# intent for CI).
test-cpu:
	uv run --frozen pytest -n auto --timeout=60 tests/ -m "not slow and not gpu"

# What the local pre-commit pytest hook runs: offline-safe (network tests are
# auto-skipped, not attempted) so you can commit without a network connection
# (see issue #13, P4).
test-precommit:
	GENREC_NO_NETWORK=1 uv run --frozen pytest -x -n auto --timeout=60 tests/ -m "not slow"

# No test currently carries @pytest.mark.slow (see issue #13 report: this
# unit does not own any tests/test_*.py file to apply the marker to). Until
# one does, `pytest -m slow` legitimately collects 0 tests and exits 5;
# tolerate that specific case rather than treating it as a target failure.
test-slow:
	uv run --frozen pytest tests/ -m slow --timeout=600; \
	code=$$?; \
	if [ $$code -eq 5 ]; then \
		echo "no tests marked 'slow' yet (see issue #13) - tolerating pytest exit 5"; \
		exit 0; \
	fi; \
	exit $$code

# GPU-only tests never run in CI (GitHub-hosted runners have no CUDA device);
# run locally, typically in WSL.
test-gpu:
	uv run --frozen pytest tests/ -m gpu --timeout=600

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
