# sample-llm-eval

LLM の出力品質を **定量的に比較評価** するパイプラインです。
複数の候補モデル（例: tsuzumi2 vs Azure OpenAI）の出力を、LLM-as-a-Judge 方式でスコアリングし、レポートを自動生成します。

## パイプライン概要

```
testcases.jsonl ─┐
                 ├─▶ Stage 1: Inference ──▶ inference-{run_id}.jsonl
run-config.yaml ─┘         │
                           ├─▶ Stage 2: Autocheck ──▶ autocheck-{run_id}.jsonl
                           │
                           ├─▶ Stage 3: Judge ──▶ judgements-{run_id}.jsonl
                           │
                           └─▶ Stage 4: Compare ──▶ comparison-report-{run_id}.json / .md
```

| Stage | 内容 |
|-------|------|
| **Inference** | テストケース × 候補モデルごとに API を呼び出し、出力を記録 |
| **Autocheck** | 出力の形式チェック（JSON パース、スキーマ検証など）を自動実行 |
| **Judge** | LLM-as-a-Judge がルーブリックに基づいて 1/3/5 スコアで採点 |
| **Compare** | 勝率・平均スコア・タスク別集計をまとめてレポート出力 |

## セットアップ

### 前提条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャ)

### インストール

```bash
# 依存パッケージのインストール
uv sync
```

### 環境変数の設定

```bash
cp .env.example .env
# .env を編集して API キーとエンドポイントを設定
```

`.env` に設定する項目:

| 変数名 | 説明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI (本家) の API キー |
| `OPENAI_ENDPOINT` | OpenAI API 互換のエンドポイント URL (任意) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI の API キー |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI のエンドポイント URL |
| `TSUZUMI2_API_KEY` | tsuzumi2 の API キー |
| `TSUZUMI2_ENDPOINT` | tsuzumi2 のエンドポイント URL |

> **規約**: vendor 名を大文字化・ハイフンをアンダースコアに変換し、`_API_KEY` / `_ENDPOINT` を付与した環境変数名で解決します。

## 使い方

### 全ステージを一括実行

```bash
uv run llm-eval run-all --config configs/run-config.yaml
```

### ステージ個別実行

```bash
# Stage 1: 推論
uv run llm-eval infer --config configs/run-config.yaml

# Stage 2: 形式チェック
uv run llm-eval autocheck --config configs/run-config.yaml

# Stage 3: Judge 採点
uv run llm-eval judge --config configs/run-config.yaml

# Stage 4: 集計・レポート
uv run llm-eval compare --config configs/run-config.yaml
```

### 共通オプション

| オプション | 短縮 | 説明 |
|-----------|------|------|
| `--config` | `-c` | run-config YAML のパス（デフォルト: `configs/run-config.yaml`） |
| `--output` | `-o` | 出力ファイルのパス |

`autocheck`, `judge` は `--inference` (`-i`) で推論結果ファイルを明示指定可能。
`compare` は `--judgements` (`-j`) で判定結果ファイルを指定可能。

## 設定ファイル

### `configs/run-config.yaml`

評価プロトコルの全設定を定義します。

```yaml
run_id: "quality-eval-20260213"

candidates:            # 比較する候補モデル (1つ以上)
  - candidate_id: "tsuzumi2-28b"
    vendor: "tsuzumi2"
    model_id: "tsuzumi2-28b"
    generation_params:
      temperature: 0
      max_tokens: 1024

judges:                # Judge モデル (1つ以上)
  - judge_id: "aoai-judge-1"
    vendor: "azure-openai"
    model_id: "gpt-4o"
    rubric_version: "v1"

protocol:
  evaluation_mode: "pairwise"   # pairwise / absolute / hybrid
  scoring_scale: [1, 3, 5]
  blinding:
    enabled: true
    random_seed: 12345
  repeats:
    judge_repeats: 3            # 分散確認用に複数回実行
  metrics: [accuracy, completeness, ...]
  aggregation:
    method: "majority_vote"
```

設定のスキーマ: `.claude/skills/evaluating-llm-quality/schemas/run-config.schema.json`

### `data/testcases.jsonl`

テストケースを JSONL 形式で記述します。1行1ケース。

```json
{
  "testcase_id": "preprocess-001",
  "task_type": "preprocessing",
  "input": {"raw_text": "..."},
  "constraints": {
    "output_format": {"type": "json", "json_schema_ref": "schemas/preprocess-output.schema.json"}
  },
  "metadata": {"difficulty": 1, "input_length_bucket": "S"}
}
```

対応タスク種別: `preprocessing` / `report_generation` / `report_qa`

推奨フォーマット運用:

- `report_generation`: JSON + `json_schema_ref`（厳格Schema検証）

## 出力ファイル

| ファイル | 形式 | 内容 |
|---------|------|------|
| `data/inference-{run_id}.jsonl` | JSONL | 各候補モデルの生成結果 |
| `data/autocheck-{run_id}.jsonl` | JSONL | 形式チェック結果 |
| `data/judgements-{run_id}.jsonl` | JSONL | Judge の採点結果（`critical_issue` フラグ含む） |
| `data/comparison-report-{run_id}.json` | JSON | 集計レポート（機械可読） |
| `data/comparison-report-{run_id}.md` | Markdown | 集計レポート（人間可読） |

各出力のパイプラインスキーマは `.claude/skills/evaluating-llm-quality/schemas/` に定義されています。
タスク出力の検証用スキーマは `schemas/` に配置します。

## 評価指標 (Metrics)

12 種類の評価指標を run-config で選択できます。

| 指標 | 説明 | 対象タスク |
|------|------|-----------|
| accuracy | 事実の正確性 | 全タスク |
| completeness | 必須要件の網羅性 | 全タスク |
| relevance | 指示への適合度 | 全タスク |
| coherence | 論理的一貫性 | 全タスク |
| conciseness | 簡潔さ | 全タスク |
| clarity | 明確さ | 全タスク |
| reasoning | 推論・分析力 | report_generation, report_qa |
| harmlessness | 安全性 | 全タスク |
| format_compliance | 形式準拠 | 全タスク |
| citation_quality | 引用品質 | report_generation, report_qa |
| actionability | 実用性 | report_generation |
| japanese_writing_style | 日本語の流暢さ | 全タスク |

各指標は **1 (Poor) / 3 (OK) / 5 (Excellent)** の 3 段階で採点されます。
ルーブリック詳細: `.claude/skills/evaluating-llm-quality/rubrics/v1.md`

## ディレクトリ構成

```
sample-llm-eval/
├── configs/                  # 評価プロトコル設定
│   └── run-config.yaml
├── data/                     # テストケース & 出力データ
│   └── testcases.jsonl
├── schemas/                  # タスク出力検証用 JSON Schema
├── src/llm_eval/             # Python パッケージ
│   ├── cli.py                # CLI コマンド定義
│   ├── models.py             # Pydantic データモデル
│   ├── config.py             # 設定読込 & 環境変数
│   ├── llm_client.py         # OpenAI SDK ラッパー
│   ├── prompts.py            # プロンプト構築
│   ├── utils.py              # JSONL I/O, ハッシュ, 統計
│   └── stages/               # 4 ステージの実装
│       ├── inference.py
│       ├── autocheck.py
│       ├── judge.py
│       └── compare.py
├── .env.example              # 環境変数テンプレート
├── pyproject.toml            # プロジェクト設定
├── ARCHITECTURE.md           # アーキテクチャ設計書
└── README.md                 # このファイル
```

詳細な設計については [ARCHITECTURE.md](./ARCHITECTURE.md) を参照してください。
