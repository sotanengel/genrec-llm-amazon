# genrec-llm-amazon

Netflix の GenRec (LLM を prefill-only エンコーダとして使い、カタログ制約付きランキングヘッドで推薦する構成) を、公開データセット + RTX 3060 Ti (8GB) で再現可能な最小構成に落とし込む。

詳細は [DESIGN.md](DESIGN.md) を参照。

## クイックスタート

### 前提

- Python 3.11+
- Linux / WSL2 推奨（Windows では `make` の代わりに下記コマンドを直接実行）

### セットアップ

```bash
pip install -e ".[dev]"
pre-commit install
```

### テスト

```bash
make test-fast
# または
ruff check src tests
mypy --strict src
pytest -x -n auto --timeout=10 tests/ -m "not slow"
```

### データ準備 (M0)

```bash
# Amazon Reviews 2023 - Video_Games カテゴリ
python -m genrec_lite data prepare --dataset amazon_video_games

# 統計表示
python -m genrec_lite data stats --dataset amazon_video_games
```

### 5-core 統計の公式値との照合

1. [Amazon Reviews 2023 公式サイト](https://amazon-reviews-2023.github.io/) の `Video_Games` 5-core 統計を確認
2. `data stats` のユーザー数・アイテム数・インタラクション数と比較
3. 差分がある場合は `split_strategy` と 5-core フィルタ条件を確認

## ライセンス

Apache 2.0 — 個人の学習・検証目的。商用利用しない。
