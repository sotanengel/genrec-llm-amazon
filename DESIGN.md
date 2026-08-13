# GenRec-lite: 設計書

> 個人の学習・検証目的のリポジトリ。商用利用しない。
> Netflix の GenRec (LLM を prefill-only エンコーダとして使い、カタログ制約付きランキングヘッドで推薦する構成) を、
> 公開データセット + RTX 3060 Ti (8GB) で再現可能な最小構成に落とし込む。

---

## 0. このドキュメントの読み方 (Cursor向け)

- **§3 のデータ契約 (スキーマ) は絶対に守ること。** モジュール間の唯一のインターフェース。
- **§9 のマイルストーン順に実装すること。** M0 の受け入れ条件を満たさないうちに M2 に進まない。
- **§10 の落とし穴は実装前に必ず読むこと。** 特に「full-catalog ranking」「left padding」「temporal split」は
  ここを外すと数値が全部無意味になる。
- 実装しないもの: 分散学習、Web UI、リアルタイム配信、Docker本番化。ローカル1台での実験に集中する。

---

## 1. ゴールと非ゴール

### 1.1 検証したい仮説

| ID | 仮説 | 検証方法 |
|----|------|----------|
| H1 | LLM を凍結したまま prefill させた hidden state だけで、ID ベース手法に匹敵するランキングができる | M3: 学習可能なのはヘッドとアイテム埋め込みのみ |
| H2 | **cold-start アイテム** (学習データでの出現が少ない商品) で、ID ベース手法 (SASRec) を明確に上回る | M3: cold-start スライス評価 |
| H3 | コンテキスト長 (= 履歴イベント数) には elbow point があり、トークンを 1/3 に削っても品質はほぼ落ちない | M5: トークン予算アブレーション |
| H4 | 報酬重み付けランキング損失で、精度を大きく落とさずに beyond-accuracy 指標 (カバレッジ/新規性) を改善できる | M5 |
| H5 | LoRA による post-training (Phase 2 相当) は、凍結エンコーダより大きく改善する | M4 |

### 1.2 非ゴール

- Netflix 論文の完全再現 (元記事の時点でハイパラ・プロンプト・reward model 構造は非公開)
- SOTA 更新
- 商用転用

### 1.3 「成功」の定義

**全体指標で SASRec に勝てなくてよい。** 以下のどちらかが言えれば成功:

1. cold-start スライス / explore スライスで有意に勝つ
2. 全体で同等の性能に、**タスク固有の学習データ 1/10 で**到達する (GenRec の主張するデータ効率の再現)

---

## 2. ハードウェア制約と設計への影響

### 2.1 前提環境

```
GPU : NVIDIA RTX 3060 Ti (8GB GDDR6, Ampere sm_86, ~448GB/s)
      → bf16 可 / FP8 不可 / FlashAttention-2 可 (sm80+)
RAM : 32GB 想定 (16GB でも動くよう parquet + memmap で設計)
Disk: 100GB 以上の空き (生データ + 中間 parquet + hidden state cache)
OS  : Linux / WSL2
```

### 2.2 前提ツールチェイン

Ampere sm86 なので以下がフル活用できる。8GB での中〜大モデル運用は **これらの併用が前提**。

- **bf16 演算** (fp16 は使わない、オーバーフローの罠を避ける)
- **FlashAttention-2** (メモリ効率的な attention)
- **bitsandbytes nf4 double-quant** (4bit 重み、~0.55 bytes/param)
- **Unsloth** (QLoRA を VRAM ~30% 削減 + 高速化。8B QLoRA on 8GB の公式サポートの根拠)
- **gradient checkpointing** (学習時の活性化メモリ削減)
- **adamw_8bit** (オプティマイザ状態を 8bit)

**FP8 は 3060 Ti (Ampere) では非対応。** RTX 40 / 50 系用の記事に惑わされないこと。

### 2.3 VRAM 予算 (量子化を前提とした概算)

#### 2.3.1 使えるメモリ

物理 8GB のうち OS / デスクトップに 0.5〜1GB 取られるため、**実効 ~7GB** を予算とする。
以下のツール前提:

- **bitsandbytes** の nf4 (double-quant): 重み ≈ 0.5〜0.6 bytes/param
- **Unsloth** (2025-12 リリースで QLoRA VRAM さらに 30% 削減): 8GB で 7B QLoRA が公式サポート範囲
- **FlashAttention-2** (Ampere sm80+ 対応)
- **gradient checkpointing**、**adamw_8bit** オプティマイザ

#### 2.3.2 VRAM 概算式

**M2 (凍結 prefill 推論)**:

```
VRAM ≈ Weights + KV cache + Activations
     = (params × bytes_per_param)
     + (2 × n_layers × n_kv_heads × head_dim × seq_len × batch × 2B)   ※KVはbf16
     + FA2下の活性化 (無視できる程度)
```

**M4 (QLoRA + Unsloth)**:

```
VRAM ≈ Weights_4bit + LoRA_adapters + Optimizer_states(LoRA分) + Activations(grad-ckpt後)
```

Unsloth を使う場合、Mistral-7B QLoRA が **12.4GB → ~5-6GB** に落ちるという公表実測がある。
これがそのまま 7-8B クラスへの窓口を開く。

#### 2.3.3 実測値 (WSL2 / RTX 3060 Ti 8GB, driver 610.47, 2026-08-13)

環境: Ubuntu 24.04 (WSL2), CUDA 12.x, `scripts/bench_prefill.py` + `scripts/wsl/doctor.sh` 12/12 PASS。
GPU: NVIDIA GeForce RTX 3060 Ti (8192 MiB, compute capability 8.6), bf16 対応。

**bf16 凍結推論 (`padding=longest`, `scripts/bench_prefill.py`)**

| モデル | seq | batch | peak VRAM | tok/s (real) | tok/s (padded) |
|--------|-----|-------|-----------|--------------|----------------|
| Qwen3-0.6B-Base | 128 | 2 | 1.12 GB | ~1,012 | ~7,620 |
| Qwen3-1.7B-Base | 512 | 2 | 3.22 GB | ~3,346 | ~26,356 |

**4bit (nf4) 凍結推論**

| モデル | seq | batch | peak VRAM | tok/s (real) | tok/s (padded) |
|--------|-----|-------|-----------|--------------|----------------|
| Qwen3-8B-Base (nf4) | 512 | 1 | 4.52 GB | ~658 | ~5,180 |

> 8B nf4 は seq512 bs1 で 8GB 内に収まる。bs2 以上は VRAM 余裕が少ないため `check_vram.py --find-max-batch` で都度確認すること。
> 生 JSON: `reports/bench/prefill_qwen3-*.json`

**QLoRA + Unsloth (seq 512, bs 1〜2, grad-accum, grad-ckpt)** — 未実測（M4 着手時に更新）

| モデル | 想定VRAM | 30k サンプル / epoch |
|--------|---------|---------------------|
| Qwen3-1.7B | ~3GB | 2〜3h |
| SmolLM3-3B | ~4GB | 3〜5h |
| Qwen3-4B | ~5GB | 5〜7h |
| Qwen3-8B | ~6-7GB | 10〜15h |
| Llama-3.1-8B | ~6-7GB | 10〜15h |

> 上記 QLoRA 時間は「実効 prefill/backward 込みで 500〜1,500 tok/s」という粗い仮定。±2〜3倍のブレを見込む。

#### 2.3.4 8GB という制約が設計に与える帰結

| 制約 | 帰結 |
|------|------|
| LLM 本体を学習中ずっと載せられない | **Stage を分離。** 凍結エンコード (M2) と ヘッド学習 (M3) を別プロセスにし、hidden state を disk にキャッシュ |
| 全カタログ softmax の勾配が乗らない場合がある | sampled softmax + in-batch negatives + logQ correction を既定に |
| 4bit 量子化前提でも 14B 以上は載らない | 本命は **7-8B クラスまで**。それ以上は他所で |
| Unsloth 依存が実質必須 | M4 のトレーナは Unsloth を優先、標準 PEFT はフォールバック扱い |

#### 2.3.5 量子化の質的トレードオフ

4bit の質劣化は **モデルサイズが小さいほど大きい**。文献的な相場観:

| モデルサイズ | 4bit nf4 の質劣化 (下流タスク) | 推奨用途 |
|-------------|-------------------------------|---------|
| ≥ 7B | ほぼ無視できる (≤1%) | M4 の QLoRA と M2 の推論、両方 4bit で OK |
| 3〜4B | 軽微 (1〜2%) | M4 は 4bit、M2 は可能なら bf16 |
| ≤ 1.7B | 目立つ (2〜4%) | **M2 は bf16 を推奨**、M4 は 4bit で問題なし |

hidden state 抽出 (今回の用途) は、生成タスクよりも量子化耐性が高いという実務経験がある
(次トークン確率分布の微妙な差が積み上がらないため) が、これは仮説なので M5 のアブレーションで
「同モデルの bf16 vs 4bit で MRR がどう変わるか」を必ず測ること。

### 2.4 モデル選定 (ランキング付き)

#### 2.4.0 選定の絶対原則

**汎用モデルのみを採用する。ドメイン特化モデル (BLaIR 等の Amazon 事前学習モデル、
医療特化モデル、法律特化モデル等) は候補から除外する。** 理由は 2 つ:

1. **転用可能性**: 本リポジトリは「小売企業が自社データで GenRec 構成を検証する」ための
   参照実装として設計する。会社のデータドメインで事前学習された都合の良い OSS モデルは
   通常存在しないため、汎用モデルで動く構成でなければ再現性がない
2. **実験の交絡除去**: たとえば BLaIR は Amazon Reviews で事前学習されているため、
   Amazon データセットで cold-start 性能を測ると「モデルがそのアイテムを事前に見ていた」
   だけで指標が上振れする可能性がある。汎用モデルに揃えることで、性能差は
   verbalizer / 損失 / ヘッド構造という**手法側の要因**にきれいに帰属する

したがって以下のランキングは「汎用テキスト事前学習のみで、特定商用ドメインで
追加学習されていないモデル」に限定する。

#### 2.4.1 既定構成 (迷ったらこれ)

**2 段構えで進める。** 疎通用の小型で縦串を通してから、本命の 7-8B クラスに上げる。

| 用途 | モデル | 量子化 | 使用 MS |
|------|--------|-------|---------|
| 疎通・高速反復 | `Qwen/Qwen3-1.7B-Base` | bf16 (M2) / nf4 (M4) | M2, M4 早期 |
| **本命の性能検証** | `Qwen/Qwen3-8B-Base` | **nf4 + Unsloth** | M2 (推論), M4 本番 |
| アイテム埋め込み初期化 (メイン) | `Qwen/Qwen3-Embedding-0.6B` | bf16 | M3, M4 |
| アイテム埋め込み初期化 (対照群) | `BAAI/bge-m3` | bf16 | M3 アブレーション |

いずれも **base 版** (Instruct でない方) を選ぶ。prefill-only で hidden state を取るだけなので
chat template は不要、余計な post-training バイアスも避ける。

#### 2.4.2 コンテキストエンコーダ (M2 凍結 + M4 QLoRA 対象)

**用途**: verbalize されたプロンプトを prefill し、pooled hidden state を返す。
**制約**: M2 は 4bit 推論で 8GB / M4 は Unsloth + QLoRA で 8GB に収まること。

**3 ティア構造で管理する。** Tier A が本命、B は疎通、C は推論のみの対照実験。

##### Tier A: 本命 (M2 + M4 両方で使う主戦力)

| 順位 | モデル | パラメータ | ライセンス | 公開時期 | 4bit 推論 VRAM | QLoRA VRAM(Unsloth) | 選定理由 |
|------|--------|-----------|-----------|----------|---------------|---------------------|----------|
| **1** | **`Qwen/Qwen3-8B-Base`** | 8.2B | Apache 2.0 | 2025-04 | ~5.5GB | ~6-7GB | **本命。** 36T トークン / 119 言語で事前学習、128K context。8GB で QLoRA 可能な dense モデルとしては現行最強クラス。ecosystem 対応が最も成熟 |
| 2 | `meta-llama/Llama-3.1-8B` | 8B | Llama 3.1 CL | 2024-07 | ~5.5GB | ~6-7GB | **Meta 系対照として必須。** 世代がやや古いが実績と再現性は圧倒的。Qwen3-8B が単に「新しいから良い」だけなのかを切り分ける対照 |
| 3 | `Qwen/Qwen3-4B-Base` | 4B | Apache 2.0 | 2025-04 | ~3.5GB | ~5GB | Tier A の安全策。8B が想定より重い / 時間がかかる場合の代替。Qwen3-8B との差で「モデルサイズの寄与」を測る |
| 4 | `Qwen/Qwen3.5-4B-Base` | 4B | Apache 2.0 | 2026-02 | ~3.5GB | ~5GB | Qwen 系最新世代。3 との比較で世代差 (2025→2026) を測る。**⚠️ 注意**: 標準 decoder-only Transformer ではなく、Gated DeltaNet (線形アテンション) と Gated Attention (通常アテンション) を交互配置したハイブリッド構造 (8×(3×DeltaNet→FFN → 1×Attention→FFN))。`output_hidden_states=True` で得られる最終層表現の意味が Qwen3 系と同一かは自明でないため、**M2 で他モデルと同じプーリング手法が使えるか最初に検証すること**。うまくいかない場合は Tier A から外し、Tier C (対照参考) に格下げする |

##### Tier B: 疎通・高速反復 (M2 で最初に通す、M4 のハイパラ探索用)

| 順位 | モデル | パラメータ | ライセンス | 選定理由 |
|------|--------|-----------|-----------|----------|
| 5 | `Qwen/Qwen3-1.7B-Base` | 1.7B | Apache 2.0 | **M2 疎通の推奨開始点。** bf16 でも余裕、QLoRA は 2〜3h/epoch。Tier A への橋渡し |
| 6 | `HuggingFaceTB/SmolLM3-3B-Base` | 3B | Apache 2.0 | **完全 OSS モデルの参照実装。** 訓練データ・スクリプト・中間 ckpt 全公開。「公開データセットしか使わない」本検証の思想と最も整合 |
| 7 | `Qwen/Qwen3-0.6B-Base` | 0.6B | Apache 2.0 | 最速反復用。デバッグや CI / smoke test で使う。本結果には載せない |

##### Tier C: 推論のみの対照 (M2 で使う、M4 では使えない)

| 順位 | モデル | パラメータ | ライセンス | 選定理由 |
|------|--------|-----------|-----------|----------|
| 8 | `google/gemma-3-12b-pt` | 12B | Gemma | 4bit 推論のみ、seq 短め (~256) で辛うじて載る。「モデルをさらに大きくしたら伸びるか」の頭打ちを見る用。**QLoRA は不可**。Gemma ライセンスは商用可だが利用規約が Apache/MIT より複雑 |

##### 除外したもの (なぜ入れないか)

| モデル | 除外理由 |
|--------|---------|
| `Qwen/Qwen3-14B-Base` | 4bit で重み 7.7GB、KV/活性化を入れると 8GB を超え実用不可 |
| `Qwen/Qwen3.6-*` (2026-07) | 現時点で公開が 27B / 35B-A3B のみ、3060 Ti では動かない |
| `google/gemma-4-*` (2026-04) | 「effective params」表記で実 VRAM が読みにくく、PEFT / bitsandbytes 対応が発売浅い |
| Llama 4 系 | 最小サイズが Scout の 17B active、動かない |
| MoE 系全般 (Qwen3-30B-A3B, DeepSeek-V3 等) | 疎活性でもロード時に全パラメータ VRAM が要る |
| `mistralai/Mistral-7B-v0.3` | 世代が古く、Qwen3-8B / Llama-3.1-8B に総合力で劣る。追加する意味が薄い |
| BERT / DeBERTa / RoBERTa | encoder-only、decoder-only で prefill する GenRec 構成の再現にならない |
| BLaIR 系 (Amazon 事前学習) | §2.4.0 の原則により除外 |
| OpenAI / Anthropic 等の API 型モデル | 個人検証で API 課金・データ送信、hidden state 取り出し不可 |

##### モデル切替の運用ルール

1. **M2 疎通 → 必ず Tier B の 5 (Qwen3-1.7B) で始める。** ここで詰まる問題は Tier A でも詰まる
2. **M3 の main 実験 → Tier A の 1 (Qwen3-8B) を本命に据える。** これが GenRec-lite の中心結果
3. **論文比較・対照 → Tier A の 2 (Llama-3.1-8B) を並走させる。** Qwen 依存を切り分ける
4. **モデルサイズ効果の測定 → Tier A の 3 (Qwen3-4B) と 1 (Qwen3-8B) を比較**
5. **完全 OSS ストーリーが必要 → Tier B の 6 (SmolLM3-3B) を追加**
6. **頭打ちの探索 → Tier C の 8 (Gemma 3 12B) を M2 の凍結エンコードのみで**

**M3 の主結果テーブルに載せるのは 1, 2, 3 の 3 モデルまで。** それ以上はアブレーション章に閉じる。

#### 2.4.3 アイテム埋め込み初期化 (M3)

**用途**: `title + brand + category_path` (+ description 先頭) をベクトル化し、
アイテム埋め込み行列 `E` の初期値とする。事前計算が 1 回走るだけなので、モデルサイズ制約は緩い。

**汎用テキスト埋め込みモデルのみを候補とする。** ドメイン特化モデルは §2.4.0 の理由で除外。

| 順位 | モデル | パラメータ | 次元 | ライセンス | 公開時期 | 選定理由 |
|------|--------|-----------|------|-----------|----------|----------|
| 1 | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 | Apache 2.0 | 2025-06 | **主推奨。** MMTEB の 1B 以下クラスで上位。Qwen3 家系なのでコンテキストエンコーダとトークナイザ資産を共有できる。多言語対応 (100+ 言語) で TaFeng (中国語) にも使える |
| 2 | `BAAI/bge-m3` | 568M | 1024 | MIT | 2024-01 | **必須の対照群。** 汎用テキスト埋め込みの現行標準。「新旧世代の差」を測るための reference。dense/sparse/multi-vector を単一モデルで出せるため、後段でハイブリッド検索に拡張しやすい |
| 3 | `jinaai/jina-embeddings-v3` | 570M | 1024 | CC BY-NC 4.0 | 2024-09 | 3 番手の対照。LoRA アダプタ付きで task ごとに切り替え可能。**CC BY-NC のため個人検証専用**、社内転用時は不採用 |
| 4 | `Qwen/Qwen3-Embedding-4B` | 4B | 2560 | Apache 2.0 | 2025-06 | 1 のスケール版。事前計算はバッチで 1 回だけなので 3060 Ti でも回せる (推論のみで学習しない)。「テキスト埋め込みモデルのサイズがどこまで cold-start に効くか」を測る用途 |
| 5 | `intfloat/multilingual-e5-base` | 278M | 768 | MIT | 2024 | 軽量な baseline。上位が使えないときのフォールバック。多言語対応 |
| 6 | `sentence-transformers/all-MiniLM-L6-v2` | 23M | 384 | Apache 2.0 | 2021 | 極小フォールバック。CPU でも動く。性能上限は低いのでデバッグ用途に限定 |

**除外したもの:**
- `nvidia/NV-Embed-v2` (7B): サイズ的に扱いにくく、事前計算コストも大きい
- `Salesforce/SFR-Embedding-*` 系: ライセンスがコンペ制限あり
- OpenAI/Cohere/Voyage/Gemini 等の API 型: 個人利用で課金・データ送信、
  かつ将来モデルが差し替わって再現性が失われる

**必ずやるアブレーション:** `--item-init` を以下 4 パターンで比較。

| 設定 | 意味 |
|------|------|
| `random` | 埋め込みをランダム初期化。テキスト情報の寄与ゼロが基準 |
| `qwen3-emb` | 主推奨。1 位のモデルで初期化 |
| `bge-m3` | 対照群。世代差・アーキ差の確認 |
| `qwen3-emb-frozen` | E を凍結し projection だけ学習。**「LLM で履歴を読む」より「商品テキストを読む」だけで済んでいないか**を検出する critical な対照 |

`qwen3-emb-frozen` が本命の GenRec-lite に迫るなら、
「LLM で履歴文脈をエンコードすること」の付加価値が薄いということになる。**この分離が本検証の核心**。

#### 2.4.4 モデル ID の凍結ルール

- **すべての run metadata に `model_id` と `revision` (git commit hash) を必ず記録する。**
  HF 側でモデルが差し替わっても比較が壊れないよう、`revision="..."` を config に書く。
  例: `Qwen/Qwen3-0.6B-Base @ da87bfb...` のように記録
- **モデル追加は §13 のプロンプト分離ルールに従う。** 「Qwen3.5 も試して」ではなく
  「@configs/model/qwen3-0.6b-base.yaml と同形で qwen35-0.8b-base.yaml を追加、`model_id` と `revision` のみ変更」
- **M3 の主結果テーブルに載せるモデルは最大 3 つまで**。追加はアブレーション章に閉じる
- **モデルのライセンスをコード内に明示する。** `configs/model/*.yaml` の各ファイルに
  `license` と `commercial_use_ok` フィールドを持たせ、CI で「未定義ならエラー」にする

---

### 2.5 ダウンロード URL 一覧 (2026-08-13 時点で実在確認済み)

**すべて本ドキュメント作成時に個別に web 検索し、モデルカード / ファイル一覧ページの実在を確認済み。**
ただし HF / Kaggle 側は今後もモデル追加・URL 変更がありうるため、実装時に各自 404 でないことを再確認すること。
ゲート付き (Gated) モデルは HF アカウントでライセンス同意が必要。

#### コンテキストエンコーダ (LLM)

| モデル | URL | 備考 |
|--------|-----|------|
| Qwen3-8B-Base | https://huggingface.co/Qwen/Qwen3-8B-Base | Apache 2.0、ゲートなし |
| Llama-3.1-8B | https://huggingface.co/meta-llama/Llama-3.1-8B | **ゲート付き**。Llama 3.1 Community License への同意が必要 (通常即時〜数時間で承認) |
| Qwen3-4B-Base | https://huggingface.co/Qwen/Qwen3-4B-Base | Apache 2.0、ゲートなし |
| Qwen3.5-4B-Base | https://huggingface.co/Qwen/Qwen3.5-4B-Base | Apache 2.0。§2.4.2 の注意書き参照 (ハイブリッドアーキ) |
| Qwen3-1.7B-Base | https://huggingface.co/Qwen/Qwen3-1.7B-Base | Apache 2.0、ゲートなし |
| SmolLM3-3B-Base | https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base | Apache 2.0、ゲートなし |
| Qwen3-0.6B-Base | https://huggingface.co/Qwen/Qwen3-0.6B-Base | Apache 2.0、ゲートなし (デバッグ専用) |
| Gemma 3 12B (pt) | https://huggingface.co/google/gemma-3-12b-pt | **ゲート付き**。Gemma 利用規約への同意が必要 |

#### アイテム埋め込み初期化

| モデル | URL | 備考 |
|--------|-----|------|
| Qwen3-Embedding-0.6B | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | Apache 2.0 |
| Qwen3-Embedding-4B | https://huggingface.co/Qwen/Qwen3-Embedding-4B | Apache 2.0 |
| BGE-M3 | https://huggingface.co/BAAI/bge-m3 | MIT |
| jina-embeddings-v3 | https://huggingface.co/jinaai/jina-embeddings-v3 | **CC BY-NC 4.0** (個人検証専用、商用不可) |
| multilingual-e5-base | https://huggingface.co/intfloat/multilingual-e5-base | MIT |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 |

#### データセット

| データセット | URL | 備考 |
|-------------|-----|------|
| Amazon Reviews 2023 (本体) | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | HF datasets 経由で `load_dataset` 可能 |
| Amazon Reviews 2023 (公式サイト・標準分割の説明) | https://amazon-reviews-2023.github.io/ | ベンチマーク分割の仕様はここ |
| Amazon Reviews 2023 処理スクリプト | https://github.com/hyp1231/AmazonReviews2023 | 5-core 化・標準分割生成用 |
| TaFeng | https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset | Kaggle アカウントでダウンロード |
| Dunnhumby The Complete Journey | https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey | Reality Check 論文の比較対象と同一データ |
| Instacart Market Basket Analysis | https://www.kaggle.com/c/instacart-market-basket-analysis | **コンペ規約により非商用限定**。個人検証のみ |
| H&M Personalized Fashion Recommendations | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations | **コンペ規約により非商用限定**。個人検証のみ |

#### 推奨ダウンロード方法

```bash
# huggingface-cli (推奨: レジューム可能、並列DL)
pip install -U "huggingface_hub[cli]" --break-system-packages
huggingface-cli download Qwen/Qwen3-8B-Base --local-dir ./models/qwen3-8b-base

# ゲート付きモデル (Llama, Gemma) は事前に https://huggingface.co/settings/tokens で
# token を発行し、`huggingface-cli login` してから同様に download する

# Kaggle データセット
pip install kaggle --break-system-packages
# ~/.kaggle/kaggle.json に API token を配置してから
kaggle datasets download -d chiranjivdas09/ta-feng-grocery-dataset
kaggle competitions download -c instacart-market-basket-analysis
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
```

**Unsloth 経由の事前量子化版について**: `unsloth/Qwen3-8B-unsloth-bnb-4bit` のような
事前 4bit 量子化済みチェックポイントも存在し、ダウンロード容量を抑えられる
(存在は確認済み: https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit)。
ただし **これは Instruct 版がベースであり、本設計が要求する Base 版の事前量子化チェックポイントは
本ドキュメント作成時点で存在確認が取れていない。** そのため既定手順は
「bf16 の Base 版をダウンロードし、bitsandbytes / Unsloth 側でロード時に 4bit 量子化する」
とする (§2.3 の VRAM 概算はこの前提)。事前量子化版の利用は任意の高速化オプションとし、
使う場合は M2 の最初に bit-identical 性を検証すること (§14.2 `test_encoder_determinism` 相当)。

---

## 3. データ

### 3.1 データセット選定

#### プライマリ: Amazon Reviews 2023 (McAuley Lab)

**ダウンロード URL は §2.5 に一元化。** ここでは選定理由のみ記す。
- HF: `McAuley-Lab/Amazon-Reviews-2023` (URL: §2.5 参照)
- 処理スクリプト: `hyp1231/AmazonReviews2023` (URL: §2.5 参照)

**選定理由:**
1. **商品テキストが豊富** (title, description, features, category, price) — verbalizer に不可欠
2. **標準分割が公式提供されている** (`benchmark/5core/last_out_w_his`, `timestamp_w_his` 等)
   → 自前の split で数値が跳ねるリスクを排除できる
3. 逐次推薦のベンチマーク文献が極めて多い (SASRec, BERT4Rec, S3Rec, UniSRec, RecFormer, P5 等)
4. カテゴリ単位で落とせるのでサイズをコントロールできる

**推奨カテゴリ (小→大の順に進む):**

| カテゴリ | 用途 |
|----------|------|
| `Musical_Instruments` | パイプライン疎通用。最小 |
| `Video_Games` | メイン実験。テキストが充実、規模が手頃 |
| `Office_Products` | 汎化確認 |
| `Beauty_and_Personal_Care` | リピート購買が多く、小売っぽい |

**5-core を既定とする** (ユーザー・アイテムとも最低5インタラクション)。

#### セカンダリ: TaFeng (次バスケット / repeat-explore 検証用)

- 台湾のグロサリー4ヶ月分。Kaggle で入手可 (URL: §2.5 参照)
- ベンチマーク: **"A Next Basket Recommendation Reality Check"** (Li et al., TOIS 2023, arXiv:2109.14233)
  - TaFeng / Dunnhumby / Instacart で 10手法を統一条件で比較した論文。**この論文の数値が我々の比較対象になる。**
  - 前処理条件: バスケットサイズ 3〜50 のユーザーをサンプリング
- **この論文の最重要知見:** Dunnhumby と Instacart では、
  **リピート推薦が全体性能の 97% 以上を占める。**
  つまり全体指標だけ見ていると手法の差が完全に埋もれる。
  → 我々の評価が repeat / explore を必ず分離しなければならない直接の根拠。

#### オプション: Instacart / H&M

- **Instacart** (Kaggle 2017、URL: §2.5 参照): 340万注文 / 20万ユーザー / 5万商品。
  商品テキストが `product_name + aisle + department` と薄いのが難点。
  Kaggle リーダーボードがベンチマークになる。**非商用ライセンスなので個人検証のみ。**
- **H&M Personalized Fashion Recommendations** (Kaggle 2022、URL: §2.5 参照): 3,100万トランザクション / 10.5万商品。
  商品説明文が豊富で cold-start が激しい = H2 の検証に理想的。ただし規模が大きく 3060 Ti にはやや重い。
  やるならユーザーを 5〜10万人にサブサンプルする。

> **ライセンス注意**: Instacart・H&M はコンペ規約に基づく非商用利用に限る。
> リポジトリに生データをコミットしないこと (`.gitignore` に `data/raw/` を入れる)。

### 3.2 データ契約 (Parquet スキーマ) — **変更禁止**

すべてのローダはこの形に正規化して `data/processed/{dataset}/` に出力する。

**`interactions.parquet`**
```
user_id      int32     # 0-indexed の連番
item_id      int32     # 0-indexed の連番
ts           int64     # UNIX秒
basket_id    int32     # バスケット構造がないデータセットでは行ごとにユニーク値
rating       float32   # 無い場合は NaN
event_type   int8      # 0=purchase, 1=view, 2=cart, 3=review  (Amazon は全て 3)
split        int8      # 0=train, 1=valid, 2=test
```

**`items.parquet`**
```
item_id            int32
raw_id             string    # ASIN / product_id
title              string
brand              string    # 無ければ ""
category_path      string    # "Electronics > Video Games > Accessories"
price              float32   # NaN 可
description        string    # 500文字で truncate 済み
first_seen_ts      int64     # 学習データ内での初出時刻
n_train_inter      int32     # 学習データ内の出現回数 (cold-start スライスの定義に使う)
```

**`users.parquet`**
```
user_id        int32
raw_id         string
n_inter        int32
first_ts       int64
last_ts        int64
repeat_ratio   float32   # 履歴中の再購入率。repeat/explore スライスに使う
```

**`samples.parquet`** (学習・評価の1件 = 「あるユーザーのある時点」)
```
sample_id      int64
user_id        int32
cutoff_ts      int64     # この時刻より前だけを履歴として使ってよい
target_item    int32     # 正解アイテム
history        list<int32>   # cutoff より前のアイテム列 (時系列昇順)
split          int8
is_repeat      bool      # target が history に含まれるか
target_is_cold bool      # items.n_train_inter < COLD_THRESHOLD (既定 5)
```

### 3.3 分割戦略

**2種類を両方実装し、両方で報告すること。**

| 名前 | 定義 | 用途 |
|------|------|------|
| `leave_one_out` | ユーザーごとに最後を test、その前を valid | 既存文献との比較用 (Amazon公式の `last_out` と一致させる) |
| `global_temporal` | 全体をある時刻 T で切り、T 以降を test | **こちらが主指標。** 現実の運用に近い |

> leave-one-out は「未来のユーザーの行動が他ユーザーの学習データに含まれる」リークがあり、
> 数値が楽観的に出ることが知られている (Meng et al., CIKM 2020)。
> 文献比較のために計算はするが、結論は `global_temporal` で出す。

---

## 4. アーキテクチャ

### 4.1 全体フロー

```
interactions.parquet
        │
        ▼
   [Verbalizer]  ── ユーザー履歴 → 自然文プロンプト (§5)
        │
        ▼
   [LLM prefill only]  ── decode しない。pooled hidden state h を取るだけ
        │                  (M2 では凍結してキャッシュ / M4 では LoRA で学習)
        ▼
      h ∈ R^d_llm
        │
        ▼
   [Projection]  ── Linear(d_llm → d_emb), d_emb=256 既定
        │
        ▼
      z ∈ R^d_emb ──┐
                     ├── s_i = z · e_i   (内積 or 小型MLP)
   [Item Embedding] ─┘
      E ∈ R^{|C| × d_emb}
        │
        ▼
   カタログ制約されたスコア {s_i}  → ランキング
```

**Netflix 構成との対応:**

| GenRec | 本リポジトリ |
|--------|-------------|
| Phase 1 (Netflix コーパスで継続事前学習) | **実装しない。** 個人のリソースでは非現実的。この省略の影響は「新作 cold-start での性能上限が Netflix の報告値より低くなる方向」に出ると予想される。汎用モデル同士 (§2.4.2) の比較で「モデル世代差」は測れるが、「ドメイン適応の効果」は本検証では測定範囲外とする |
| Phase 2 (推薦ランキング post-training) | M4 の QLoRA |
| Verbalizer | §5 |
| カタログ制約ランキングヘッド | `RankingHead` |
| Prefill-only 推論 | `encode/prefill.py` (`max_new_tokens=0`) |
| reward-weighted loss | M5 |

### 4.2 主要インターフェース

```python
# src/genrec_lite/verbalize/base.py
class Verbalizer(Protocol):
    def render(self, sample: Sample, items: ItemStore, budget: TokenBudget) -> str: ...
    def name(self) -> str: ...

# src/genrec_lite/encode/prefill.py
class PrefillEncoder:
    """LLM を prefill のみで走らせ、pooled hidden state を返す。decode は一切しない。"""
    def __init__(self, model_id: str, dtype: str = "bfloat16",
                 pooling: Literal["last", "mean", "eos"] = "last",
                 max_len: int = 512, quantize: str | None = None): ...
    def encode_batch(self, texts: list[str]) -> Tensor:  # [B, d_llm]
        ...

# src/genrec_lite/models/genrec_lite.py
class GenRecLite(nn.Module):
    """z = proj(h);  s_i = z @ e_i.T  (+ optional bias)"""
    def __init__(self, d_llm: int, d_emb: int, n_items: int,
                 scorer: Literal["dot", "mlp"] = "dot",
                 item_init: Tensor | None = None): ...
    def score(self, h: Tensor, candidate_ids: Tensor | None = None) -> Tensor: ...

# src/genrec_lite/eval/runner.py
def evaluate(scores_fn, samples, items, ks=(10, 20),
             slices=("all","repeat","explore","cold","warm","short_hist","long_hist")
            ) -> pd.DataFrame: ...
```

### 4.3 アイテム埋め込みの初期化 (重要)

`E` をランダム初期化するとアイテム数の多い設定で cold-start が壊滅する。**既定は текст初期化:**

1. 各アイテムの `title + brand + category_path` をテキストエンコーダ (§2.4.3 の順位1〜3) に通す
2. 得られた埋め込みを PCA or 学習可能 Linear で `d_emb` に落として `E` の初期値にする
3. `--item-init {random, text, text_frozen}` で切り替えられるようにする
   - `text_frozen` は E を凍結し projection だけ学習する = 純粋な content-based の上限確認

これは H2 (cold-start での優位) の主要な効き所なので、必ずアブレーションに含める。

---

## 5. Verbalizer 仕様

### 5.1 テンプレート (Amazon / v1)

```
You are a product recommendation ranker.

## User profile
Interactions: {n_inter} | Active since: {first_seen_relative} | Repeat ratio: {repeat_ratio:.2f}
Top categories: {top3_categories}

## Purchase history (most recent first)
1. [{days_ago}d ago] {title} | {category_leaf} | ${price} | rated {rating}/5
   {description_snippet}
2. [{days_ago}d ago] {title} | {category_leaf} | ${price}
...

## Current context
Time: {weekday} {hour}:00 | Season: {season}

## Task
Predict the next product this user will purchase.
```

### 5.2 コンテキスト圧縮ルール (Netflix の「コンテキストウィンドウ = 特徴量予算」の実装)

`compress.py` に、優先度順の削減ルールとして実装する。トークン予算を超えている間、上から順に適用:

| 優先度 | ルール | パラメータ |
|--------|--------|-----------|
| 1 | 古いイベントから削る (直近 N 件のみ残す) | `max_history` (既定 20) |
| 2 | 弱いシグナル (低評価 / 低価格 / 短い滞在) を削る | `min_rating`, `min_price` |
| 3 | 同一アイテムの反復を「×N回」に集約 | `collapse_repeats=True` |
| 4 | description を落とす (title と category は残す) | `desc_top_k` — 直近 k 件だけ description を付ける (既定 3) |
| 5 | title を先頭 M 文字で truncate | `title_max_chars` (既定 60) |

**設計上の要件:**
- `TokenBudget` はトークナイザ実測でカウントすること。文字数近似はしない
- **プロンプトの先頭 (system + task 説明) を全サンプルで完全一致させる。**
  vLLM の prefix caching が効く構造の検証になるし、M2 のバッチ処理も速くなる
- レンダリング結果を必ず `reports/verbalizer_samples.md` に 20件ダンプする
  (Cursor に書かせたあと人間が目で見る工程。ここを飛ばすと壊れたプロンプトで数時間溶かす)

### 5.3 バリエーション (アブレーション用)

| 名前 | 内容 |
|------|------|
| `v0_ids_only` | `item_1234, item_5678, ...` とIDだけ並べる (テキストの寄与を測る対照群) |
| `v1_full` | 上記フルテンプレート |
| `v2_compact` | title + category のみ、description なし |
| `v3_no_context` | 履歴のみ、時刻・プロファイルなし |

---

## 6. ベースライン (すべて実装必須)

| 手法 | 実装 | 理由 |
|------|------|------|
| `Pop` (G-TopFreq) | 自前 20行 | 下限。これに負けたらバグ |
| `P-TopFreq` | 自前 | **必須。** Reality Check 論文で多くの深層手法を上回った個人頻度ベース手法 |
| `GP-TopFreq` | 自前 | 個人頻度 + 全体頻度で穴埋め |
| `ItemKNN` | 自前 (共起行列 + cosine) | 古典的だが強い |
| `SASRec` | 自前 (200行程度) または RecBole | **ID系の主対抗馬。** 文献値との照合に使う |
| `TextKNN` | 履歴アイテムのテキスト埋め込み平均 → 全アイテムと cosine | LLM を使わない content-based の上限 |
| `GenRecLite` | 本命 | |

> `TextKNN` と `GenRecLite` の差が「LLM で履歴を読むこと」の正味の価値。
> ここが縮まらないなら LLM を使う意味がない。**最も重要な対照実験。**

---

## 7. 学習

### 7.1 損失

```python
# L_rank: カタログ制約 cross-entropy
#   全カタログ softmax は |C| が大きいと乗らないので sampled softmax を既定にする
#   in-batch negatives + popularity-proportional sampling + logQ correction
L_rank = -log( exp(s_y - logQ_y) / sum_{i in S} exp(s_i - logQ_i) )

# L_reward: 報酬重み付き (M5)
L_reward = sum_j r_j * L_rank_j

# r_j の定義 (小売文脈の模擬):
#   r_j = w1 * novelty(target)        # 人気度の逆数 → ロングテール露出
#       + w2 * is_cold(target)        # 新商品の露出
#       + w3 * category_expansion(j)  # ユーザーが未購入カテゴリなら加点
#   ※ 実データに粗利がないので、上記を「ビジネス目標の代理」として使う
```

**`L_LM` (言語モデル損失) は M4 でのみ実装。** 凍結エンコーダの M3 では不要。

### 7.2 ハイパーパラメータ既定値

```yaml
head:
  d_emb: 256
  scorer: dot
  dropout: 0.1
train_head:
  optimizer: adamw
  lr: 1e-3            # ヘッドのみなので大きめ
  item_emb_lr: 1e-3
  batch_size: 512
  n_negatives: 4096   # sampled softmax
  epochs: 50
  early_stop_patience: 5
  monitor: valid/ndcg@20
train_lora:
  # 8B クラス想定。Tier B (1.7B, 3B) では bs を上げても良い
  backend: unsloth       # 標準 PEFT よりVRAM 30%削減。フォールバックとして peft も残す
  lr: 1e-4
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # all linear
  batch_size: 1          # 8B なら 1、4B なら 2、1.7B なら 4 が目安
  grad_accum: 16         # 実効 batch 16 に揃える
  gradient_checkpointing: true
  quantize: nf4
  double_quant: true
  optim: adamw_8bit
  max_len: 512
  max_train_samples: 30000
  # ★ モデルサイズ別の実行時間目安 (30k サンプル / 1 epoch):
  #   Qwen3-1.7B ~ 2-3h  /  SmolLM3-3B ~ 3-5h  /  Qwen3-4B ~ 5-7h  /  Qwen3-8B ~ 10-15h
  # M4 は最初 1.7B で回路を確認し、動いたら 8B に上げて一晩回す運用
```

### 7.3 再現性

- `seed` を config に持ち、`torch`/`numpy`/`random` に伝播
- **すべての run は seed を 3つ (0,1,2) 振り、平均±標準偏差で報告する。**
  推薦系はシード差が手法差より大きいことが珍しくない
- run ごとに `reports/runs/{run_id}/{config.yaml, metrics.json, git_sha.txt}` を出力

---

## 8. 評価

### 8.1 プロトコル

- **full-catalog ranking (全アイテムをスコアリング) を必須とする。**
  「正解 + ランダム負例100件」のサンプリング評価は禁止。
  サンプリング評価は手法間の順位を歪めることが知られている (Krichene & Rendle, KDD 2020)。
- 学習データで既に購入済みのアイテムは **除外しない** (repeat 予測を評価対象にするため)。
  ただし repeat / explore スライスで必ず分離して報告する。

### 8.2 指標

**精度**: `Recall@{10,20}`, `NDCG@{10,20}`, `MRR@20`, `HitRate@{10,20}`
**beyond-accuracy**: `Coverage@20` (推薦されたユニークアイテム数 / |C|), `Gini@20`, `AvgPopularity@20`, `Novelty@20`

### 8.3 スライス (**これが本設計の中核**)

`eval/slices.py` で以下を必ず出す。全体指標だけの表は作らないこと。

| スライス | 定義 | なぜ見るか |
|----------|------|-----------|
| `repeat` | target が履歴に含まれる | Reality Check 論文より、グロサリー系ではここが全体性能の97%以上を占める。ここに埋もれると手法差が見えない |
| `explore` | target が履歴にない | **LLM が効くべき領域。H2 の主戦場** |
| `cold` | `items.n_train_inter < 5` | **cold-start。H2 の主戦場** |
| `warm` | `n_train_inter >= 20` | 対照 |
| `short_hist` | 履歴 < 5 件 | 疎なユーザー |
| `long_hist` | 履歴 >= 20 件 | 長文脈が効くか |
| `pop_decile_{0..9}` | target の人気度十分位 | 人気バイアスの可視化 |

出力は必ず `手法 × スライス` のマトリクスで `reports/results.md` に。

### 8.4 文献照合 (M1 の受け入れ条件)

SASRec を Amazon の公式 `5core/last_out` 分割 + full ranking で回し、
公表されている同条件の数値と **相対 ±20% 以内** に入ること。
入らない場合、以下を疑う: 分割の不一致 / サンプリング評価との混同 / 学習不足 / パディング方向。

---

## 9. マイルストーンと受け入れ条件

### M0: データパイプライン (1〜2日)
- [ ] `python -m genrec_lite data prepare --dataset amazon_video_games` で §3.2 の parquet 一式が出る
- [ ] `data stats` がユーザー数 / アイテム数 / 密度 / repeat_ratio / 期間を出力
- **AC**: 5-core 適用後の統計値が Amazon 公式サイトの公表値と一致する
- **AC**: `global_temporal` 分割で、test に train 期間のイベントが1件も混入していないことをテストで保証

### M1: ベースライン + 評価ハーネス (2〜3日)
- [ ] §6 の `Pop` / `P-TopFreq` / `GP-TopFreq` / `ItemKNN` / `SASRec` / `TextKNN`
- [ ] §8 の指標とスライス
- **AC**: §8.4 の文献照合をパス
- **AC**: `Pop` が全手法中最下位であること (逆転していたらどこかがバグ)
- **AC**: `results.md` が 手法 × スライス のマトリクスで自動生成される

### M2: Verbalizer + 凍結 prefill エンコード (2〜3日)
- [ ] `scripts/bench_prefill.py` でスループット実測 → §2.3 の表を更新
- [ ] `verbalize render --n 20` でサンプルダンプ → **人間が目視確認**
- [ ] hidden state を `float16` memmap + index parquet でキャッシュ
- **AC**: 3060 Ti で OOM せずに train+valid+test 全サンプルをエンコードできる
- **AC**: キャッシュサイズが `N × d_llm × 2 bytes` と一致
- **AC**: 同一入力を2回エンコードして bit-identical (決定性の確認)

### M3: GenRec-lite ヘッド学習 (2〜3日) ★ H1/H2 の検証
- [ ] `GenRecLite` + sampled softmax
- [ ] `--item-init {random, text, text_frozen}`
- **AC**: `TextKNN` を全体 NDCG@20 で上回る
- **AC**: `cold` スライスで `SASRec` を上回る ← **ここが本題**
- **AC**: `v0_ids_only` verbalizer より `v1_full` が良い (テキストが効いている証明)

### M4: QLoRA post-training (3〜5日、うち学習は一晩×数回) ★ H5
- [ ] 4-bit + LoRA + gradient checkpointing。まず 0.6B で
- [ ] `L_rank` と `L_LM` の重み付き合成
- **AC**: 8GB に収まり、OOM せず 1 epoch 完走する
- **AC**: M3 の凍結版を上回る
- **AC**: `max_train_samples` を {3k, 10k, 30k} と振ったデータ効率カーブを出す
      → **SASRec がフルデータで到達する性能に、何分の1のデータで届くか**が H5 の答え

### M5: アブレーション (3〜4日) ★ H3/H4
- [ ] コンテキスト長 elbow: `max_history` ∈ {3,5,10,20,50} で NDCG とトークン数を両方プロット
- [ ] verbalizer バリエーション v0〜v3 の比較
- [ ] 報酬重み付き損失: 精度 vs Coverage/Novelty のトレードオフ曲線
- [ ] プーリング方式 {last, mean, eos} の比較
- **AC**: 品質-コスト平面のプロットが出る (`n* = argmax [ Q(n) - λ·Cost(n) ]` の可視化)
- **AC**: `reports/FINDINGS.md` に H1〜H5 の判定 (支持 / 不支持 / 判定不能) が書かれている

### M6 (任意): TaFeng で次バスケットに拡張
- 問題設定が「次の1件」から「次のバスケット」に変わる。損失を multi-label BCE に変更
- Reality Check 論文の公表値と比較

---

## 10. 落とし穴 (実装前に必読)

1. **decoder-only モデルのプーリングは left padding が必須。**
   right padding のまま「最後のトークンの hidden state」を取ると、パディング位置の出力を掴んで全部壊れる。
   `tokenizer.padding_side = "left"` を設定し、単体テストで確認すること。

2. **サンプリング評価をしない。** §8.1 の通り full ranking。
   文献の数値をコピーしてくるときも、それが sampled か full かを必ず確認する。両者は比較不能。

3. **leave-one-out のリーク。** 主結論は `global_temporal` で出す (§3.3)。

4. **repeat に埋もれる。** 全体指標だけ見て「勝った/負けた」を判断しない。
   Reality Check 論文はグロサリー系で repeat が性能の97%超を占めることを示している。

5. **人気バイアス。** NDCG が上がってもカバレッジが崩壊しているだけ、が頻発する。
   beyond-accuracy 指標を必ず併記。

6. **fp16 のオーバーフロー。** Ampere なので **bf16 を使う**。fp16 は避ける。

7. **アイテム埋め込みテーブルのサイズ。**
   `|C|=100k, d=256, fp32` で 100MB。勾配 + Adam の状態で ×4 = 400MB。ここまでは平気。
   `|C|` が 100万を超えるデータセット (H&M フル) では sparse embedding か候補集合制限が必須。

8. **hidden state キャッシュの整合性。**
   verbalizer やモデルを変えたら**キャッシュを必ず無効化する。**
   キャッシュキーに `hash(model_id + verbalizer_name + verbalizer_config + max_len)` を含めること。
   ここを手抜きすると「なぜか結果が変わらない」で半日溶ける。

9. **seed 1本で結論を出さない。** §7.3。

10. **タイムスタンプの単位。** Amazon Reviews 2023 はミリ秒精度のフィールドがある。秒に正規化してから使う。

11. **`description` の長さ。** Amazon の description は数千文字あることがある。
    parquet 化の時点で truncate しておかないとメモリを食い潰す。

---

## 11. リポジトリ構成

```
genrec-lite/
├── DESIGN.md                  # このファイル
├── README.md                  # クイックスタートのみ
├── pyproject.toml
├── Makefile                   # make data / make baselines / make encode / make train / make report
├── .gitignore                 # data/raw, data/processed, cache/, reports/runs/
├── configs/
│   ├── base.yaml
│   ├── data/{amazon_video_games,amazon_beauty,tafeng}.yaml
│   ├── model/llm/{qwen3-1.7b-base,qwen3-4b-base,qwen3-8b-base,llama-3.1-8b,smollm3-3b-base}.yaml
│   ├── model/embed/{qwen3-emb-0.6b,bge-m3,e5-multilingual-base}.yaml
│   ├── verbalizer/{v0_ids_only,v1_full,v2_compact,v3_no_context}.yaml
│   └── exp/{m1_baselines,m3_frozen,m4_lora,m5_ablation}.yaml
├── src/genrec_lite/
│   ├── cli.py                 # typer。data / verbalize / encode / train / eval / report
│   ├── config.py              # pydantic でスキーマ検証
│   ├── data/
│   │   ├── schema.py          # §3.2 の dataclass + parquet 検証
│   │   ├── loaders/{amazon.py,tafeng.py,instacart.py,hm.py}
│   │   ├── split.py           # leave_one_out / global_temporal
│   │   └── stats.py
│   ├── verbalize/
│   │   ├── base.py
│   │   ├── templates.py       # jinja2
│   │   ├── compress.py        # §5.2 の優先度付き削減
│   │   └── budget.py          # トークナイザ実測カウント
│   ├── encode/
│   │   ├── prefill.py         # PrefillEncoder
│   │   ├── cache.py           # memmap + キャッシュキー管理
│   │   └── text_embed.py      # §2.4.3 のモデルでアイテム埋め込み初期化
│   ├── models/
│   │   ├── genrec_lite.py
│   │   ├── ranking_head.py
│   │   └── baselines/{pop.py,topfreq.py,itemknn.py,sasrec.py,textknn.py}
│   ├── train/
│   │   ├── losses.py          # sampled_softmax_with_logq / reward_weighted
│   │   ├── head_trainer.py
│   │   └── lora_trainer.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── slices.py
│   │   └── runner.py
│   └── report/
│       └── build.py           # results.md / プロット生成
├── scripts/
│   ├── bench_prefill.py       # M2 冒頭で実行。スループット実測
│   └── check_vram.py
├── tests/
│   ├── test_schema.py
│   ├── test_split_no_leak.py  # ★ 必須
│   ├── test_padding_side.py   # ★ 必須
│   ├── test_metrics.py        # 手計算した小さなケースで NDCG/Recall を検証
│   └── test_cache_key.py
└── reports/
    ├── verbalizer_samples.md
    ├── results.md
    ├── FINDINGS.md
    └── runs/
```

### 依存パッケージ

```toml
[project.dependencies]
torch, transformers, accelerate, peft, bitsandbytes, unsloth,
sentence-transformers, datasets,
polars, pyarrow, numpy, scipy,
typer, pydantic, pyyaml, jinja2, rich,
matplotlib, pandas
[dev]
pytest, ruff, mypy
```

> `polars` を既定にする (pandas より省メモリで、32GB 未満の環境で効く)。

---

## 12. 実装順序の指示 (Cursor向け)

1. `pyproject.toml`, `Makefile`, `.gitignore`, `configs/base.yaml` の骨組み
2. `data/schema.py` → `tests/test_schema.py` が通るまで
3. `data/loaders/amazon.py` + `data/split.py` → `tests/test_split_no_leak.py` が通るまで
4. `eval/metrics.py` → `tests/test_metrics.py` (手計算ケース) が通るまで
5. `models/baselines/pop.py`, `topfreq.py` → まず一番簡単な手法で評価ハーネスを一周させる
6. `models/baselines/sasrec.py` → §8.4 の文献照合
7. `verbalize/` 一式 → サンプルダンプして目視
8. `encode/prefill.py` + `scripts/bench_prefill.py` → 実測してから本番エンコード
9. `models/genrec_lite.py` + `train/head_trainer.py`
10. `train/lora_trainer.py`
11. `report/build.py`

**各ステップで `make report` が動く状態を保つこと。** 大きく作ってから繋ぐのではなく、
最小の縦串を通してから横に太らせる。

---

## 13. Cursor への実装依頼の粒度

### 13.1 基本原則

1. **1 プロンプト = 1 ファイル (目安 300 行以内) または 1 クラス + そのユニットテスト。**
   これを超える依頼は必ず分割する。Cursor は 500 行を超えると型ヒントを黙って落としたり、
   既存の関数シグネチャを勝手に変えたりし始める。
2. **周辺コンテキスト (依存する schema や隣接インターフェース) は先に open してから依頼する。**
   `@DESIGN.md §3.2` `@src/genrec_lite/data/schema.py` のように参照を明示する。
3. **「実装して」と「良くして」は必ず別プロンプト。**
   最初は「テストを通す最小実装」、次に「ここを速く/正確に」の順にする。
4. **コンフィグ変更とコード変更を混ぜない。** 混ざると diff レビューが破綻する。
5. **新規ファイルの追加は明示的に許可する。** 「このファイル以外変更しない」と書かないと、
   関係ない config や util を勝手に触りにいく。

### 13.2 プロンプトのテンプレート

すべての実装依頼は以下の 5 要素を必ず含める。

```
【参照】 @DESIGN.md §X.Y, @src/genrec_lite/xxx/yyy.py
【やること】 一文で目的
【インターフェース】 関数/クラスのシグネチャを完全一致で提示 (§4.2 からコピペ)
【テスト】 tests/test_zzz.py の以下ケースが通ること: [ケース列挙]
【制約】 - 新しい依存を追加しない
         - 300 行以内
         - このファイル以外変更しない
         - print ではなく logging を使う
```

**悪いプロンプト例**: 「M3 の GenRecLite を実装して」
**良いプロンプト例**: 「@DESIGN.md §4.1, §4.3, §7.1 を参照。
`src/genrec_lite/models/genrec_lite.py` に `GenRecLite` クラスだけを実装する。
シグネチャは §4.2 の通り。`tests/test_genrec_lite.py` の
`test_score_shape`, `test_dot_equivalence`, `test_text_init_shape` が通ること。
sampled softmax は losses.py に既にあるので import して使う。」

### 13.3 マイルストーンごとの分割目安

| MS | プロンプト数 | 各プロンプトのスコープ |
|----|-------------|----------------------|
| M0 | 3〜4 | (a) schema.py + test_schema (b) amazon loader (c) split.py + test_split_no_leak (d) stats + CLI |
| M1 | 6〜8 | metrics.py + test → Pop → P/GP-TopFreq → ItemKNN → TextKNN → SASRec → 文献照合スクリプト。**手法1つ = 1プロンプト** |
| M2 | 4 | (a) verbalize/templates + compress + budget (b) verbalizer サンプルダンプ CLI (c) prefill.py + bench (d) cache.py + text_embed.py |
| M3 | 3 | (a) models/genrec_lite.py + ranking_head.py (b) train/losses.py + head_trainer.py (c) M3 の実験ランナー |
| M4 | 2〜3 | (a) train/lora_trainer.py (b) LoRA + LM loss 統合 (c) データ効率カーブのスクリプト |
| M5 | 5 | アブレーション項目 (コンテキスト長 / verbalizer / pooling / 報酬重み / item_init) を **1項目 = 1プロンプト** |

「M1 を全部やって」は禁止。手法ごとに区切ると、途中で数値がおかしい時に犯人が特定できる。

### 13.4 やってはいけない依頼

- 「全部実装して」「M3 を実装して」など、スコープが 3 ファイル以上に及ぶ依頼
- 「性能を改善して」(何を、何の指標で、どの程度改善するかを明示せず)
- 「バグを直して」(期待動作と実際の動作を明示せず)
- 「テストを書いて」(§14 のケースを指定せず)
- 「リファクタして」(意図と受け入れ条件を明示せず)
- 実装とレビューを同じプロンプトで頼む (別セッションで人間がレビューする)

### 13.5 コミット単位

**1 プロンプト = 1 コミット** を原則とする。プロンプト → 差分確認 → テスト実行 → コミット → 次のプロンプト。
コミットメッセージには対応する DESIGN.md のセクションと MS 番号を書く (例: `M3: implement GenRecLite (§4.1)`)。
Cursor が複数目的を混ぜて実装してきたら、コミットせずに分割し直しを依頼する。

### 13.6 ペースの目安

M1〜M2 は 1 日に 3〜5 プロンプト、レビュー含めて 4〜6 時間。M3 以降は 1 日 2〜3 プロンプト。
一度に長時間走らせず、テストが通るたびに人間が結果を目視する。特に §5.2 の verbalizer 出力と
§8.3 のスライス表は数字だけ見ていると壊れているのに気づけないので、毎回目で見る。

---

## 14. 自動テスト

### 14.1 方針

- **フィクスチャは超小型のダミーデータで用意する。**
  ユーザー 10 人、アイテム 20 件、インタラクション 50 件程度の parquet を `tests/fixtures/mini/`
  に置き、全ユニットテストはこれで走る。**実データをテストで使わない** (CI 不能・遅い・再現性ゼロ)。
- **重い E2E テストは 1 本だけ** (`tests/test_smoke_end_to_end.py`)。マーカー `@pytest.mark.slow` を付け、
  通常の `pytest` からは除外。手動で `pytest -m slow` で走らせる。
- **数値の期待値は手計算で導出し、コメントに式を書く。** マジックナンバー禁止。
- **すべてのテストは 5 秒以内に完了する** (slow を除く)。超えるものは fixture を小さくする。
- pytest + pytest-xdist で並列実行。`ruff` と `mypy --strict` も CI に含める。

### 14.2 テストファイル一覧と各ケース

#### `tests/conftest.py`
- `mini_dataset` フィクスチャ: §3.2 スキーマに準拠した最小 parquet セットをtmp_path に作って返す
- `tiny_model_id`: `"sshleifer/tiny-gpt2"` (テスト用の超小型モデル、実 LLM の代わり)
- `deterministic_seeds`: torch/numpy/random を 42 で固定するフィクスチャ

#### `tests/test_schema.py` (M0)
| ケース | 期待 |
|--------|------|
| `test_interactions_columns_and_dtypes` | §3.2 の列名と dtype に完全一致 |
| `test_items_columns_and_dtypes` | 同上 |
| `test_users_columns_and_dtypes` | 同上 |
| `test_user_id_is_zero_indexed_contiguous` | `set(user_id) == set(range(n_users))` |
| `test_item_id_is_zero_indexed_contiguous` | 同上 |
| `test_split_values_in_012` | `interactions.split.unique() ⊆ {0,1,2}` |
| `test_no_negative_ts` | ts >= 0 |
| `test_items_first_seen_le_ts` | `items.first_seen_ts <= interactions.ts.min()` per item |
| `test_users_n_inter_matches` | `users.n_inter == interactions.groupby(user_id).size()` |

#### `tests/test_split_no_leak.py` (M0、**最重要テストの一つ**)
| ケース | 期待 |
|--------|------|
| `test_global_temporal_no_future_leak` | `interactions[split==0].ts.max() < interactions[split==2].ts.min()` |
| `test_leave_one_out_last_per_user_in_test` | 各ユーザーの test は 1 件、そのユーザーの最終 ts と一致 |
| `test_leave_one_out_valid_is_second_last` | 同様に valid は 2 番目に新しい 1 件 |
| `test_all_users_have_train` | すべての user_id が train (split=0) に少なくとも 1 件出現 |
| `test_target_item_id_in_train_vocab_or_flagged_cold` | test の target が train に無いなら `target_is_cold=True` になっている |
| `test_history_only_from_before_cutoff` | 各 sample の history 内 ts が すべて `cutoff_ts` 未満 |

#### `tests/test_padding_side.py` (M2、**最重要テストの一つ**)
| ケース | 期待 |
|--------|------|
| `test_tokenizer_padding_side_is_left` | `PrefillEncoder` が使う tokenizer の `padding_side == "left"` |
| `test_last_token_pooling_ignores_pad` | 長さ違いの 2 文をバッチと単体でエンコードし、pooled ベクトルが `atol=1e-4` で一致 |
| `test_batch_invariance` | 同じ 4 文を bs=1,2,4 で流し、結果が一致 |
| `test_eos_pooling_finds_correct_position` | eos プーリング時に attention_mask の最終 True 位置の hidden を取れているか、既知位置のトークンで検証 |

**このテストが落ちるまま先に進むと M3 以降の数値がすべて無効になる。CI で必須マーク。**

#### `tests/test_metrics.py` (M1)
手計算ケースをコメントに残す。例:

```python
def test_ndcg_at_3_single_hit_position_2():
    # ranking = [item0, item1, TARGET, item3, item4]  (0-indexed で target が 2 番目)
    # NDCG@3 = DCG@3 / IDCG@3
    # DCG@3 = 1 / log2(1 + 2) = 1 / log2(3) ≈ 0.6309
    # IDCG@3 = 1 / log2(1 + 0) = 1.0
    # → NDCG@3 ≈ 0.6309
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    target = 2
    assert ndcg_at_k(scores, target, k=3) == pytest.approx(0.6309, abs=1e-4)
```

| ケース | 期待 |
|--------|------|
| `test_recall_at_k_hit` | target が top-k に入る → 1.0 |
| `test_recall_at_k_miss` | 入らない → 0.0 |
| `test_ndcg_at_k_position_0` | target が 1 位 → 1.0 |
| `test_ndcg_at_3_single_hit_position_2` | 上記 |
| `test_mrr_multiple_users` | 3 ユーザーで順位 1, 2, 5 → MRR = (1 + 0.5 + 0.2) / 3 |
| `test_coverage_at_k` | 全ユーザーに同じアイテムを推薦 → coverage = 1 / n_items |
| `test_gini_uniform_is_zero` | 全アイテム同確率 → gini ≈ 0 |
| `test_metrics_batch_matches_loop` | バッチ計算と loop 計算が一致 |

#### `tests/test_full_ranking.py` (M1)
| ケース | 期待 |
|--------|------|
| `test_evaluator_scores_all_items` | 呼び出しごとに `scores.shape[-1] == n_items` (サンプリング評価になっていないこと) |
| `test_evaluator_does_not_hide_train_items` | 学習済みアイテムがスコア対象から除外されていない |

#### `tests/test_verbalizer.py` (M2)
| ケース | 期待 |
|--------|------|
| `test_render_deterministic` | 同じ入力 → 同じ出力 |
| `test_compression_reduces_below_budget` | 履歴を過剰に含む sample でも、budget=256 なら 256 トークン以内 |
| `test_compression_priority_order` | max_history 削減より前に、weak signal 削減が起きていない (§5.2 の順序保持) |
| `test_id_only_verbalizer_contains_no_titles` | v0 テンプレートは title を含まない |
| `test_prompt_prefix_is_shared` | 異なるユーザーで、先頭 100 トークンが完全一致 (prefix cache 前提の担保) |

#### `tests/test_encoder_cache.py` (M2)
| ケース | 期待 |
|--------|------|
| `test_key_changes_with_model_id` | model_id を変えるとキャッシュキーが変わる |
| `test_key_changes_with_verbalizer_name` | verbalizer 名を変えると変わる |
| `test_key_changes_with_verbalizer_config` | max_history などを変えると変わる |
| `test_key_stable_across_process` | 同じ設定なら別プロセスでも同じキー |
| `test_cache_hit_returns_bit_identical` | 保存 → 再読み込みで bit-identical (`torch.equal`) |
| `test_cache_miss_recomputes` | キーが違えば再計算される |

#### `tests/test_encoder_determinism.py` (M2)
| ケース | 期待 |
|--------|------|
| `test_same_input_same_output` | 同じ入力 x を 2 回 encode → `torch.allclose(rtol=0, atol=1e-5)` (bf16 だと若干緩めても可) |
| `test_no_grad_by_default` | 凍結モードで `.requires_grad == False` |

#### `tests/test_genrec_lite.py` (M3)
| ケース | 期待 |
|--------|------|
| `test_score_shape_full_catalog` | 出力 shape が `(B, n_items)` |
| `test_score_shape_candidate_set` | candidate_ids 指定時は `(B, K)` |
| `test_dot_equivalence` | `scorer='dot'` 時、`score(h) == h @ E.T` (手計算一致) |
| `test_text_init_uses_provided_matrix` | `item_init` に渡した行列が `E.weight.data` に反映されている |
| `test_text_init_shape_projection` | 入力 dim が d_emb と違うとき、projection で正しく落ちる |
| `test_backward_updates_head_only_when_llm_frozen` | LLM 側 grad が None、head の grad が非 None |

#### `tests/test_losses.py` (M3)
| ケース | 期待 |
|--------|------|
| `test_sampled_softmax_matches_full_when_all_sampled` | 全アイテムを負例に取ると full softmax と一致 |
| `test_logq_correction_unbiased_gradient` | 一様サンプリング時、log Q 補正が定数となる (勾配同一) |
| `test_reward_weight_scales_loss` | 全 r_j を 2 倍にすると loss も 2 倍 |
| `test_reward_zero_masks_sample` | r_j=0 の sample は勾配に寄与しない |

#### `tests/test_slices.py` (M1〜M3)
| ケース | 期待 |
|--------|------|
| `test_repeat_flag_correctness` | history に target が含まれる sample のみ `is_repeat=True` |
| `test_cold_flag_uses_threshold` | `n_train_inter < 5` のアイテムのみ cold |
| `test_all_slice_counts_sum_to_total` | repeat + explore = 全体、cold + warm ≤ 全体 (未分類あり) |
| `test_slice_metrics_never_nan_when_nonempty` | 空でないスライスの指標は NaN にならない |

#### `tests/test_baselines.py` (M1)
| ケース | 期待 |
|--------|------|
| `test_pop_recommends_most_frequent` | mini fixture で最頻アイテムが 1 位 |
| `test_p_topfreq_prefers_user_history` | 個人頻度が全体頻度に優先される |
| `test_gp_topfreq_falls_back_to_global_when_short_hist` | 履歴 1 件のユーザーで G-TopFreq に近い挙動 |
| `test_itemknn_symmetric_similarity` | cosine 類似度が対称 |
| `test_sasrec_forward_shape` | `(B, L)` 入力 → `(B, L, d)` 出力 |
| `test_sasrec_causal_mask` | 位置 t の出力が t+1 以降のトークンに依存していない (位置ずらしテスト) |

#### `tests/test_smoke_end_to_end.py` (`@pytest.mark.slow`)
mini fixture + `tiny-gpt2` で以下を通す:

1. `data prepare` で parquet 一式が出る
2. `verbalize render` でプロンプト文字列が返る
3. `encode` で hidden state cache が作られる
4. `train head` で 1 epoch が完走する
5. `eval` で results 表が生成される
6. **5 分以内に完走する**

このテストが通れば、パイプラインの継ぎ目のバグは全部潰れている。
M3 完了時点で必ずこれを通す。

### 14.3 CI (ローカルでもよい) の実行順

```makefile
test-fast:
	ruff check src tests
	mypy --strict src
	pytest -x -n auto --timeout=10 tests/ -m "not slow"

test-slow:
	pytest tests/ -m slow --timeout=600
```

`test-fast` は毎コミット、`test-slow` は M3 と M4 の完了時に手動で。

### 14.4 テストが落ちたときの対応

1. **テストを緩めない。** テスト側を書き換えて通すのは禁止 (fixture が明らかに間違いの場合を除く)。
2. **失敗が「境界での丸め」なら atol を明示的に緩める。** rtol/atol を「なんとなく」変えない。
3. **flaky (時々落ちる) テストは即座に issue 化。** 特に決定性系 (`test_encoder_determinism`) が
   flaky なら、GPU の非決定性演算が混じっているサイン。`torch.use_deterministic_algorithms(True)` を確認。
4. **統計的テスト (数値実験の受け入れ条件、§9 の AC) は seed 3 本の中央値で判定する。**
   単一 seed で判定してはならない。

---

## 15. 免責

- 個人の学習・技術検証目的。商用利用しない。
- 利用データセットはそれぞれのライセンス / コンペ規約に従う (Instacart・H&M は非商用限定)。
- 生データおよび派生成果物をリポジトリにコミットしない。
- 本設計は Netflix Tech Blog "GenRec: Towards LLM-native Recommendation at Netflix" の
  公開情報に基づく**独立した再解釈**であり、Netflix の実装の再現ではない。
