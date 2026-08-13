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

## WSL2 GPU セットアップ (Windows + RTX 3060 Ti)

Windows 11 + WSL2 + NVIDIA GPU でこのリポジトリを動かすための、チェックイン済みで冪等なスクリプト群を `scripts/wsl/` に用意している。**このマシンで実行するものは必ずスクリプトに書いて実行する**という方針に従い、下記はすべて `scripts/wsl/*.sh` / `scripts/wsl/*.ps1` を呼び出すだけで、手打ちのコマンドは想定していない。

### 前提

- Windows 側で WSL のカーネルと `VirtualMachinePlatform` / `Microsoft-Windows-Subsystem-Linux` optional feature が有効化済みであること（未有効の場合は別途 Windows の再起動を伴う管理者作業が必要）。
- NVIDIA GPU 用の Windows ドライバがインストール済みであること。**WSL 内に Linux 用 NVIDIA ドライバ（`nvidia-driver-*` や `cuda-drivers`）を入れてはいけない** — Windows ドライバが提供する `/usr/lib/wsl/lib/libcuda.so.1` を上書きしてしまい、WSL 上で最も多い「CUDA が使えない」原因になる。
- `%USERPROFILE%\.wslconfig` の `[wsl2] memory=` は **物理 RAM の 75% 以下**に保つこと。このマシンでは 24GB/31.8GB (75.5%) が上限で、これを超えると `Wsl/Service/E_UNEXPECTED` で WSL VM がクラッシュする（2026-07 に 29GB 設定で 3 回実測）。`scripts/wsl/doctor.sh` はこの不変条件を**確認するだけ**で、`.wslconfig` 自体は書き換えない。

### セットアップ手順

Windows 側 (PowerShell) から順に実行する。

```powershell
# 1. Ubuntu-24.04 を登録し、非対話でユーザーを作成する（既に登録済みなら即終了する冪等スクリプト）
.\scripts\wsl\install_distro.ps1

# 2. リポジトリを ext4 (~/src/genrec-llm-amazon) に clone し、uv / Python 3.12 / 依存関係をセットアップする
#    (初回は ~/src にまだ clone されていないため、Windows 側からマウントした /mnt/c 上のこのファイルを直接指定する)
wsl.exe -d Ubuntu-24.04 -u <あなたのLinuxユーザー名> -- bash -lc "/mnt/c/Users/<you>/genrec-llm-amazon/scripts/wsl/bootstrap.sh"

# 3. 以降は ext4 上の clone を使って Invoke-Wsl.ps1 経由で実行する
.\scripts\wsl\Invoke-Wsl.ps1 -Command "scripts/wsl/doctor.sh"
```

リポジトリは **GitHub を正とし**、WSL 側の `~/src/genrec-llm-amazon` は実行専用の clone、この Windows 側 worktree は編集専用として扱う（この worktree の `.git` は Windows パスを指すファイルであり、WSL 内からは意味を持たないため、`/mnt/c` 上で直接 git 操作をすることもできない）。`bootstrap.sh` は未 push のブランチを GitHub を経由せず転送できるよう、`win` という名前で Windows 側 worktree をセカンダリ remote として追加する（ベストエフォート）。

### `doctor.sh` が確認する項目

`scripts/wsl/doctor.sh`（`Invoke-Wsl.ps1 -Command "scripts/wsl/doctor.sh"` で実行）が「WSL2 GPU bring-up 完了」の定義である。以下を**それぞれ独立に**チェックし（`;` 連結は最後の終了コードしか伝播しないため使わない）、PASS/FAIL 表を表示して 1 つでも FAIL があれば非ゼロで終了する。

- WSL バージョン / カーネル
- `/usr/lib/wsl/lib/libcuda.so.1` の存在
- `nvidia-smi` が動くか
- `torch.cuda.is_available()`
- `torch.cuda.get_device_capability() == (8, 6)`（Ampere / RTX 3060 Ti）
- `torch.cuda.is_bf16_supported()`
- 空き VRAM
- `HF_HOME` が ext4 上にあるか（`/mnt/c` = 9p ではないこと）
- `PYTORCH_CUDA_ALLOC_CONF` が unset であること（後述）
- WSL の MemTotal が `.wslconfig` の設定 (~24GB) と一致し、Windows 物理 RAM の 75% 以下であること

### `PYTORCH_CUDA_ALLOC_CONF` を絶対に設定しない理由

このマシン（NVIDIA driver 610.47）では、WSL2 上で `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` と VRAM オーバーサブスクリプションを組み合わせると、`RuntimeError: CUDA driver error: device not ready` が実行のたびに違う op で発生する（`dmesg | grep make_resident` に `dxgkio_make_resident: Ioctl failed: -12` (ENOMEM) が出る）。原因は `expandable_segments` が CUDA VMM API (`cuMemCreate`/`cuMemMap`) を使い、通常の `cudaMalloc` とは別の residency パスを WSL2 の dxgkrnl パラバーチャライゼーション層に通すため。`scripts/wsl/env.sh` はこの変数を `unset` した上で、なお設定されていたら（何かが再度 export していたら）大きなエラーメッセージを出して失敗する。**どのスクリプトもこの変数を export してはならない**（`tests/test_wsl_scripts.py` がこれを回帰テストとして強制している）。

### `HF_TOKEN` / `HF_HUB_OFFLINE` ワークフロー

1. Hugging Face のトークンを `~/.config/genrec/hf_token` に保存し、`chmod 600` する（コミットしない・echo しない）。認証なしの Hub アクセスはレート制限がかかり不安定なため。
2. `scripts/wsl/fetch_models.sh` で `configs/model/llm/*.yaml` に定義されたモデル（`model_id` / `revision`）を `hf download`（`huggingface-cli download` は非推奨）でダウンロードする。ダウンロードは再開可能。`gated: true` のモデルでダウンロードが失敗した場合は、生の 401 トレースバックの代わりに該当モデルのライセンスページへのリンクを表示する。
3. モデル取得後は `scripts/wsl/run.sh --offline ...` でオフライン実行し、ネットワークに依存しない再現可能な実行にする（`HF_HUB_OFFLINE=1`）。

### WSL クラッシュ後に不可解なモデルロードエラーが出たら

WSL VM がハードクラッシュした直後は `~/.cache/huggingface` の中身（`config.json` / `tokenizer.json` など）が書き込み途中で壊れていることがあり、`Unrecognized model` や `Couldn't instantiate the backend tokenizer` のような不可解なエラーになる。**該当モデルのキャッシュディレクトリを削除して `scripts/wsl/fetch_models.sh` で再取得する**のが対処法。

## ライセンス

Apache 2.0 — 個人の学習・検証目的。商用利用しない。
