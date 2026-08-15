#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${GENREC_DATASET:-amazon_video_games}"
MODEL="${GENREC_MODEL:-qwen3-8b-base}"
VERBALIZER="${GENREC_VERBALIZER:-v1_full}"
EXP="${GENREC_EXP:-m3_frozen}"

echo "==> data prepare (skip if parquet exists)"
DATA_DIR="$(uv run --frozen python -c "
from genrec_lite.config import find_project_root, load_data_config
cfg = load_data_config('${DATASET}', config_dir=find_project_root() / 'configs')
print(find_project_root() / cfg.output_dir)
")"
if [[ ! -f "${DATA_DIR}/samples.parquet" ]]; then
  uv run --frozen python -m genrec_lite data prepare --dataset "${DATASET}"
fi

echo "==> encode eval (cache hit exits 0)"
uv run --frozen python -m genrec_lite encode run \
  --dataset "${DATASET}" --model "${MODEL}" --verbalizer "${VERBALIZER}" --scope eval || true

echo "==> encode train"
uv run --frozen python -m genrec_lite encode run \
  --dataset "${DATASET}" --model "${MODEL}" --verbalizer "${VERBALIZER}" --scope train

echo "==> train head"
uv run --frozen python -m genrec_lite train head --exp "${EXP}"

echo "==> report"
uv run --frozen python -m genrec_lite report build --exp "${EXP}"

echo "==> done"
