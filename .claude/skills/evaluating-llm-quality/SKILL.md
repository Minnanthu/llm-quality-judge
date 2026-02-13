---
name: evaluating-llm-quality
description: Evaluates and compares LLM output quality by producing inference artifacts, running automated checks, running one or more judges, and generating comparative reports for multiple task types.
# Claude Code optional fields (safe defaults):
disable-model-invocation: true
argument-hint: "[run-config.yaml]"
# allowed-tools: Read, Write, Grep  # adjust for your environment
---

# Evaluating LLM Quality (A/B + Multi-Judge)

## Goal
品質（Quality）比較の検証フレームワークを、**測定方式（プロトコル）を固定せず**に改善できる形で運用する。
そのために、処理を以下の役割に分離し、**中間成果物（artifact）を標準化**する。

- Inference（候補モデルごとの生成）
- Auto-check（形式準拠などの機械検査）
- Judge（Judgeモデルごとの採点。複数Judge対応）
- Compare/Aggregate（採点結果の集約・比較レポート生成）

## Non-goals
- 単一の測定方式（例：ペア比較だけ）にロックしない。
- 評価の“正解”を押し付けない。代わりに、プロトコル差し替えと再現性を優先する。

## Inputs
- `configs/run-config.yaml`（プロトコル/モデル/Judge/繰り返し/集約方法など）
- `data/testcases.jsonl`（テストケース。タスク種別、入力、制約、期待形式など）

### Required testcase fields (Quality)
`data/testcases.jsonl` は少なくとも以下を持つこと（`schemas/testcases.schema.json` 準拠）。

- `testcase_id`: ケースの一意ID
- `task_type`: `preprocessing` / `report_generation` / `report_qa`
- `input`: 評価対象入力（オブジェクト）
- `constraints`: 制約（`required_points`, `forbidden_points`, `output_format`, `citation_policy`）
- `metadata`: 難易度 (`difficulty`), 入力長バケット (`input_length_bucket`: S/M/L), `tags`

`json_schema_ref` は `constraints.output_format` に記載する。

- 例: `constraints.output_format.type: json`
- 例: `constraints.output_format.json_schema_ref: schemas/report-generation-output.schema.json`

タスク別の推奨運用:

- `report_generation`: JSON出力 + `json_schema_ref` 指定（厳格検証）

## Outputs (artifacts)
- `data/inference-{run_id}.jsonl` : 候補モデルの生成結果（1行=1レコード）
- `data/autocheck-{run_id}.jsonl` : 自動検査結果（形式準拠、schema検証など）
- `data/judgements-{run_id}.jsonl` : Judgeごとの採点結果（1行=1判定）
- `data/comparison-report-{run_id}.json` / `.md` : 集約結果

各artifactは `schemas/*.schema.json` に準拠すること。

## Protocol is configurable
測定方式は `configs/run-config.yaml` で制御する（例：ペア比較/絶対採点/マルチJudge集約/反復回数/重み）。
方式の改善は、原則として **run-config と judge rubric** の更新で行い、artifactスキーマは安定させる。

### Required run-config fields (Quality)
`run-config.yaml` は `schemas/run-config.schema.json` に準拠し、以下のトップレベル構造を持つ。

- `run_id`: 実行の一意ID
- `candidates`: 候補モデル配列（`candidate_id`, `vendor`, `model_id`, `generation_params` 等）
- `judges`: Judgeモデル配列（`judge_id`, `vendor`, `model_id`, `rubric_version` 等）
- `dataset`: テストケースファイルのパスとバージョン
- `protocol`: 評価プロトコル設定
  - `evaluation_mode`: `pairwise` / `absolute` / `hybrid`
  - `blinding`: `enabled` (bool), `random_seed` (int)
  - `repeats`: `inference_repeats`, `judge_repeats`
  - `metrics`: 評価指標リスト
  - `aggregation`: `method` (`mean`/`majority_vote`/`worst_case`/`custom`), `weights`

### Required metrics (default)
`protocol.metrics` には、特段の理由がない限り以下を含める。

**Core 9（全タスク共通）:**
`accuracy`, `completeness`, `relevance`, `coherence`, `conciseness`, `clarity`, `harmlessness`, `format_compliance`, `reasoning`

**Optional 3:**
`citation_quality`, `actionability`, `japanese_writing_style`

## Judge policy (for fairness)
- 同一run内で Judgeモデルと Judgeプロンプトは固定する。
- 候補出力はブラインド化し、`answer_a` / `answer_b` として提示する。
- Judge出力は「勝者/同点」「軸別スコア」「理由」を必須とする。
- 可能な限り自動判定を優先し、Judge裁量のみで決まる項目を減らす。
- `format_compliance` は、`protocol.metrics` に含まれ、かつ testcase が `output_format.type=json` と `json_schema_ref` を持つ場合、LLM JudgeではなくJSON Schema厳格検証結果を採用する。

## Safety / Reproducibility
- ブラインド（回答1/回答2）と順序ランダム化をサポートする。
- run_id / prompt_version / params_hash を必ず記録し、再実行可能にする。
- 可能な限り“自動判定（format_compliance等）”を優先し、Judge裁量を減らす。

## Qualitative review (human gate)
定量判定に加え、以下の定性チェックを運用手順に含める。

- 内容妥当性（要件への適合）
- 形式妥当性（フォーマット崩れ有無）
- 不適切出力有無（安全性）
- セキュリティ観点（注入耐性、機微情報混入）

重大バグ（致命誤り、断定的ハルシネーション、機微漏洩、重大な形式崩れ）は、
`judgements.jsonl` または別途レビュー記録で `critical_issue=true` として機械可読に残す。

## Comparison report minimums
`data/comparison-report-{run_id}.json` / `.md` は最低限以下を出力する。

- ケース総数、有効判定数、除外数
- モデル別の勝率/敗率/同点率
- 軸別平均スコアと分散（または信頼区間）
- 反復実行の安定性指標（ケース内ばらつき）
- `critical_issue` 件数（タスク別内訳含む）

## References
- 詳細設計・運用ルール：`REFERENCE.md`
- スキーマ：`schemas/*.schema.json`
- ルーブリック：`rubrics/v1.md`
