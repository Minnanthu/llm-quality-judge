# REFERENCE.md — LLM Quality Evaluation Framework（プロトコル改善前提）

このドキュメントは、品質評価フレームワークの「運用ルール / 設計詳細」です。
SKILL.md は薄く保ち、詳細はここに集約します（progressive disclosure）。

---

## 0. このスキルの設計原則（固定するもの / 固定しないもの）

### 0.1 固定するもの（Stable Contract）
測定方式を改善しても壊れないよう、以下は原則固定（互換維持）します。

- **artifact（中間成果物）のフォーマット**
  - `data/inference-{run_id}.jsonl` / `data/autocheck-{run_id}.jsonl` / `data/judgements-{run_id}.jsonl` / `data/comparison-report-{run_id}.json`
  - それぞれ `schemas/*.schema.json` に準拠
- **バージョニングと追跡可能性**
  - `run_id`, `dataset_version`, `prompt_version`, `rubric_version`, `params_hash`, `input_hash`
- **役割分離**
  - Inference → Auto-check → Judge → Compare/Aggregate

### 0.2 固定しないもの（Configurable / Evolving）
改善余地として “run-config” 側に逃がす対象です。

- 評価モード（pairwise / absolute / hybrid）
- ブラインド方式（提示順ランダム、seed、tieの扱い）
- Judge構成（単一 / マルチJudge / 第三者Judge）
- 反復回数（inference/judge repeats）
- メトリクスの重み、集約法（平均/多数決/最悪値/カスタム）
- 自動検査ルール（schema、regex、禁止語、長さ制約…）
- 監査（人手スポットチェック）の対象範囲

---

## 1. AnthropicのSKILL.md作法（公式準拠のための注意）

### 1.1 ファイル/フォルダ命名の前提
- スキル本体は **`SKILL.md`（完全一致）**。
- frontmatter（YAML）は “スキルをロードするかどうか” に影響するため重要。

### 1.2 frontmatter の必須扱い（標準 vs Claude Code）
- **Agent Skills（標準）**: `name` と `description` を必須として扱うのが安全。
- **Claude Code**: frontmatter は任意（ただし `description` 推奨、`name` は省略時にディレクトリ名）。

> 運用推奨：両対応のため、SKILL.md には常に `name` と `description` を入れる。

### 1.3 “薄いSKILL.md + 参照ファイル” の推奨
- SKILL.md を肥大化させず、**REFERENCE.md / examples.md 等に分割**する。
- 参照のネストは浅く保つ（SKILL.md → 参照ファイル、程度）。

---

## 2. 全体フロー（言語非依存）

### 2.1 ステージと責務
1) **Inference（生成）**
- 入力：`data/testcases.jsonl`, `configs/run-config.yaml`
- 出力：`data/inference-{run_id}.jsonl`（候補モデル×テストケース×反復）

2) **Auto-check（機械検査）**
- 入力：`data/inference-{run_id}.jsonl` + testcase constraints
- 出力：`data/autocheck-{run_id}.jsonl`
- 目的：`format_compliance` 等を **Judgeから切り離し**、不公平を減らす

3) **Judge（採点）**
- 入力：testcase + inference outputs +（必要なら）autocheck結果
- 出力：`data/judgements-{run_id}.jsonl`（Judgeごとに独立）
- 重要：A/Bのブラインド提示・提示順ランダムをサポート

4) **Compare/Aggregate（集約・比較）**
- 入力：`data/judgements-{run_id}.jsonl`（複数Judgeを含む）+ `configs/run-config.yaml`
- 出力：`data/comparison-report-{run_id}.json` +（任意で）`.md`

---

## 3. プロトコル（測定方式）を改善し続けるためのルール

### 3.1 プロトコル変更は “Config差し替え” を基本にする
- 実装を変える前に、まず `configs/run-config.yaml` の変更で吸収できないか検討
- 変更のたびに `protocol_version`（run-config内 or ファイル名）を更新

### 3.2 互換性ルール（破壊的変更を避ける）
- artifactスキーマを変える場合は「新スキーマファイル追加（v2）」にし、既存を壊さない
- `candidate_id` / `judge_id` は固定IDとして扱い、表示名は別に持つ

### 3.3 推奨初期設定（固定ではない）
- 生成パラメータは比較のため揃える（temperature/top_p/max_tokens 等）
- seed が揃わない場合は **反復回数で吸収**する（inference repeats を増やす）
- 入力長は S/M/L などバケットで分解（品質差の傾向が見えやすい）

---

## 4. Judge設計（不公平を抑えつつ、取り回し良く）

### 4.1 Judgeは「A用/B用に分ける」より「judge_idで差し替え」が楽
- Judgeの実体を `judge_id` として増やすだけで、同じパイプラインで回せる
- `judgements.jsonl` は Judgeごとに追記・分離しても良い（後でマージ可能）

### 4.2 ブラインド（必須級）
- A/Bを “回答1/回答2” として提示し、提示順はランダム
- `presented_order` と `random_seed` を記録し、後から復元できるようにする

### 4.3 マルチJudge運用（推奨）
- バイアスが出やすい項目だけでもマルチJudge化する
  - 高リスク：`harmlessness`, `japanese_writing_style`, `clarity`, `conciseness`, `reasoning`
- 集約は run-config で切替可能にする（多数決/平均/最悪値）

### 4.4 Rubricの置き場所（推奨）
- `rubrics/<rubric_version>.md`
- Rubricには以下を必ず含める：
  - メトリクス定義（9+3）
  - 1/3/5 のアンカー
  - tie の許容条件
  - “根拠（evidence）推奨” の指示

---

## 5. Auto-check（機械検査）を先に作る理由
- `format_compliance` は可能な限り schema/regex等で自動判定し、Judge裁量を減らす
- citationの形式（参照ID/URL/箇所）も可能なら機械検査を混ぜる
- Auto-checkは「落とす」のではなく、Judge/Reportに **失敗理由を運ぶ**のが運用しやすい

### 5.1 `format_compliance` の厳格判定ルール
- testcase の `constraints.output_format` に `type: json` と `json_schema_ref` がある場合、`format_compliance` は JSON Schema 検証結果を一次判定とする。
- この条件では Judge は `format_compliance` を採点しない（Judgeへの評価指標から除外）。
- `json_schema_ref` の記載場所は testcase レコード内の `constraints.output_format.json_schema_ref`。
- 例: `schemas/preprocess-output.schema.json`, `schemas/report-generation-output.schema.json`

---

## 6. データセット設計（品質比較の土台）

### 6.1 テストケースに必ず入れるもの
- `task_type`（preprocessing / report_generation / report_qa）
- `input_length_bucket`（S/M/L）
- 制約：必須項目、禁止事項、期待形式（JSON schema参照など）

### 6.2 分割（リーク防止）
- `dev`：rubricやプロンプト調整に使う
- `holdout`：最終比較に使う（調整に使わない）

### 6.3 件数の目安（運用ガイド）
- スモーク：各セル（task×bucket）10件
- 標準：各セル20件
- 意思決定：各セル40件
※差が僅差なら必要数は増える（run-configで調整）。

---

## 7. Reproducibility（再現性）とログ

### 7.1 すべてのrunで残すもの
- `configs/run-config.yaml` のコピー（結果フォルダに固定保存）
- `data/inference-{run_id}.jsonl` / `data/autocheck-{run_id}.jsonl` / `data/judgements-{run_id}.jsonl` / `data/comparison-report-{run_id}.json`
- 主要hash：`prompt_hash`, `params_hash`, `input_hash`

### 7.2 エラー取り扱い
- 生成失敗（timeout/429/5xx）やパース失敗は欠損として記録し、成功率・欠損率を必ず出す

---

## 8. レポーティング（説明責任）

### 8.1 最低限出す集計
- 勝率（pairwise）/ 平均スコア（absolute）
- タスク別・長さ別（S/M/L）で分解
- Judge間一致率（マルチJudge時）
- 重大失敗例（リンクと理由）

### 8.2 改善ループ
- failure taxonomy として分類し、プロンプト改善・自動検査ルール追加・rubric明確化につなげる（holdoutは汚さない）

---

## 9. ファイル参照（ナビ）
- `schemas/run-config.schema.json`
- `schemas/testcases.schema.json`
- `schemas/inference-record.schema.json`
- `schemas/autocheck-record.schema.json`
- `schemas/judgements.schema.json`
- `schemas/comparison-report.schema.json`
- `schemas/preprocess-output.schema.json`
- `schemas/report-generation-output.schema.json`
- `rubrics/v1.md`
- `templates/*.md`
