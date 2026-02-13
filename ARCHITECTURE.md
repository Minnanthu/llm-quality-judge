# ARCHITECTURE.md

本ドキュメントでは、LLM 評価パイプラインの内部設計を説明します。

---

## 設計原則

### 安定する部分（変更しない）

- **成果物スキーマ**: `schemas/` 配下の JSON Schema はパイプライン間の契約。容易に変更しない。
- **4 ステージ構成**: Inference → Autocheck → Judge → Compare の役割分離は固定。
- **トレーサビリティ**: `run_id`, `prompt_version`, `input_hash` 等を必ず記録し、再現可能にする。

### 可変な部分（run-config で制御）

- 評価モード（pairwise / absolute / hybrid）
- ブラインド化の有無・ランダムシード
- Judge 構成（単一 / マルチ Judge）
- 繰り返し回数（推論・Judge それぞれ）
- 評価指標の選択と重み
- 集約方式（mean / majority_vote / worst_case / custom）

> **方式の改善は run-config とルーブリックの更新で行い、コード変更を最小化する。**

---

## モジュール構成

```
src/llm_eval/
├── cli.py            ─── Typer CLI。各ステージのエントリポイント
├── models.py         ─── 全 JSON Schema に対応する Pydantic モデル
├── config.py         ─── RunConfig YAML 読込 + 環境変数解決
├── llm_client.py     ─── OpenAI SDK ラッパー + リトライ制御
├── prompts.py        ─── 候補推論 / Judge 用プロンプト構築
├── utils.py          ─── JSONL I/O, JSON 出力, ハッシュ, 統計
└── stages/
    ├── inference.py  ─── Stage 1
    ├── autocheck.py  ─── Stage 2
    ├── judge.py      ─── Stage 3
    └── compare.py    ─── Stage 4
```

---

## データフロー

```
                    run-config.yaml
                         │
                         ▼
┌─────────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│ testcases   │───▶│ Inference │───▶│ Autocheck │    │  Judge   │
│   .jsonl    │    │ (Stage 1) │    │ (Stage 2) │    │(Stage 3) │
└─────────────┘    └─────┬─────┘    └─────┬─────┘    └────┬─────┘
                         │                │               │
                  inference-{run_id}.jsonl   autocheck-{run_id}.jsonl  judgements-{run_id}.jsonl
                         │                │               │
                         └────────────────┴───────────────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │  Compare   │
                                   │ (Stage 4)  │
                                   └──────┬─────┘
                                          │
                              comparison-report-{run_id}.json / .md
```

各ステージは **前ステージの出力ファイル (JSONL/JSON)** のみに依存します。
途中から再実行したい場合は、対応する入力ファイルを `--inference` や `--judgements` オプションで指定できます。

---

## 各ステージの詳細

### Stage 1: Inference (`stages/inference.py`)

**入力**: run-config + testcases.jsonl
**出力**: `data/inference-{run_id}.jsonl`

処理の流れ:

1. テストケースを読み込み、各ケースに対してプロンプトを構築（`prompts.py`）
2. testcase × candidate × inference_repeats の全組み合わせで API 呼び出し
3. 出力テキスト・トークン使用量・レイテンシ・ステータスを記録
4. JSON 形式が期待される場合はパース結果も `json` フィールドに格納

**プロンプト構築ロジック** (`prompts.py`):
- タスク種別ごとのテンプレート（`templates/preprocessing.md` 等）を読み込む
- `constraints` の出力形式・必須要件・禁止事項をシステムメッセージに埋め込む
- テストケースの `input` をユーザメッセージとして整形

**エラーハンドリング**: API 呼び出し失敗時は `status.ok=false` で記録し、パイプラインは止めない。

### Stage 2: Autocheck (`stages/autocheck.py`)

**入力**: run-config + inference.jsonl
**出力**: `data/autocheck-{run_id}.jsonl`

実行するチェック:

| チェック | 内容 |
|---------|------|
| `format_compliance` | 期待形式（JSON/markdown/free_text）と実出力を比較 |
| `json_schema_validation` | `json_schema_ref` で参照されるスキーマでバリデーション |

- JSON 出力が期待される場合: パース可否を検査
- Markdown 出力が期待される場合: 見出し (`#`) の存在を検査
- スキーマ参照がある場合: `jsonschema` ライブラリで Draft 2020-12 バリデーション

> Autocheck の結果は Stage 4 (Compare) の Notable Failures に反映されます。

### Stage 3: Judge (`stages/judge.py`)

**入力**: run-config + inference.jsonl
**出力**: `data/judgements-{run_id}.jsonl`

評価モード別の動作:

#### Pairwise モード

1. 同一テストケースの候補ペアを全組み合わせ (C(n,2)) で列挙
2. **ブラインド化**: 提示順をランダム化（`blinding.random_seed` で再現可能）
3. Judge モデルに Answer A / Answer B として提示し、JSON 形式で採点を要求
4. Judge の回答をパースし、ラベル (A/B) を candidate_id にマッピング
5. `judge_repeats` 回繰り返し、分散を確認

#### Absolute モード

1. 各候補出力を単独で Judge に提示
2. 各指標について 1/3/5 で採点 + 全体スコア

#### Hybrid モード

Pairwise と Absolute の両方を実行します。

**Judge プロンプト**:
- ルーブリック (`rubrics/v1.md`) を全文埋め込み
- 有効な評価指標リストとスコアリングスケールを提示
- `response_format: json_object` で JSON 応答を強制
- `critical_issue` 判定を要求（重大な事実捏造、安全性違反、形式不全、指示無視）

### Stage 4: Compare (`stages/compare.py`)

**入力**: run-config + judgements.jsonl（+ autocheck.jsonl）
**出力**: `data/comparison-report-{run_id}.json` / `.md`

集計内容:

| 集計 | 説明 |
|------|------|
| **overall.win_rate** | 候補ごとの勝率（pairwise）、tie 率を含む |
| **overall.mean_score** | 評価指標ごとの平均スコア |
| **by_task** | task_type 別の集計 |
| **by_bucket** | input_length_bucket (S/M/L) 別の集計 |
| **critical_issue_count** | 候補ごとの重大品質欠陥 (`critical_issue=true`) の件数 |
| **judge_agreement** | Judge 間の一致率（同一ケース・同一ペアの判定一致度） |
| **notable_failures** | Autocheck で検出された形式エラー・スキーマ不整合 |

Markdown レポートには上記を表形式で可視化します。

---

## LLM クライアント (`llm_client.py`)

```
vendor 名 → create_client() → OpenAI or AzureOpenAI インスタンス
                                    │
                         chat_completion() ← tenacity リトライ
                                    │
                              API レスポンス
```

- **vendor 解決**: vendor 名を大文字化し `{VENDOR}_API_KEY` / `{VENDOR}_ENDPOINT` 環境変数を参照
- `azure-openai` → `AzureOpenAI` クライアント
- その他（`tsuzumi2` 等） → `OpenAI` 互換クライアント（`base_url` にエンドポイントを指定）
- **リトライ**: `tenacity` で最大 3 回、指数バックオフ（2〜30 秒）

---

## データモデル (`models.py`)

すべての JSON Schema に 1:1 対応する Pydantic v2 モデルを定義しています。

| Pydantic モデル | 対応スキーマ | 用途 |
|----------------|-------------|------|
| `Testcase` | `testcases.schema.json` | テストケース定義 |
| `RunConfig` | `run-config.schema.json` | 評価プロトコル設定 |
| `InferenceRecord` | `inference-record.schema.json` | 推論結果レコード |
| `AutoCheckRecord` | `autocheck-record.schema.json` | 自動チェック結果 |
| `JudgementRecord` | `judgements.schema.json` | Judge 採点結果 |
| `ComparisonReport` | `comparison-report.schema.json` | 比較集計レポート |

バリデーションは入出力の両方で行われ、スキーマ不整合を早期に検出します。

---

## 設定の読み込み (`config.py`)

```
.env ファイル
     │
     ▼
EnvConfig (pydantic-settings)  ← 環境変数を自動読込
     │
     ▼
resolve_vendor_env("azure-openai")
  → AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

run-config.yaml
     │
     ▼
load_run_config() → RunConfig (Pydantic バリデーション済み)
```

---

## 拡張ポイント

### 新しい vendor を追加する

1. `.env` に `{VENDOR}_API_KEY` と `{VENDOR}_ENDPOINT` を追加
2. `llm_client.py` の `create_client()` に分岐を追加（OpenAI 互換なら不要）
3. `run-config.yaml` の candidates/judges に追加

### 新しいタスク種別を追加する

1. `schemas/` にタスク出力用のスキーマを追加
2. `templates/` にプロンプトテンプレートを追加
3. `data/testcases.jsonl` に `task_type` を指定したケースを追加

### 評価プロトコルを変更する

- **コード変更不要**: `configs/run-config.yaml` を編集するだけ
- 新しいルーブリックが必要なら `rubrics/v2.md` を追加し、`rubric_version: "v2"` を指定

---

## スキーマ一覧

| スキーマファイル | 概要 |
|-----------------|------|
| `run-config.schema.json` | 評価プロトコル設定 |
| `testcases.schema.json` | テストケース定義 |
| `inference-record.schema.json` | 推論結果 |
| `autocheck-record.schema.json` | 自動チェック結果 |
| `judgements.schema.json` | Judge 採点結果 |
| `comparison-report.schema.json` | 比較集計レポート |
| `preprocess-output.schema.json` | 前処理タスク出力形式 |
| `report-generation-output.schema.json` | 深掘り分析（レポート生成）タスク出力形式 |

すべて JSON Schema Draft 2020-12 に準拠しています。
