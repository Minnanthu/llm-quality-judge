# スキル・ドキュメント・実装 整合性チェック結果

- 対象リポジトリ: `sample-llm-eval`
- 実施日: 2026-02-16
- 結論: **齟齬あり**

## 前提の更新（反映済み）

- `1,3,5` は固定値ではなくアンカー
- LLM採点で中間値 `2,4` を許容
- 集計結果は小数を許容

この前提により、`judgements` に `2/4` が入ること自体は不具合ではなく、仕様記述・スキーマ側の不一致として扱う。

## 確認された齟齬（重大度順）

### 1. [高] run-config の「スキーマ準拠必須」方針と実装が不一致

- スキル文書は `run-config.yaml` の JSON Schema 準拠を明示しているが、実装は Pydantic での読み込みのみで JSON Schema 検証を行っていない。
- そのため、schema違反の設定でも処理が進む。

参照:
- `.claude/skills/evaluating-llm-quality/SKILL.md:63`
- `src/llm_eval/config.py:29`
- `.claude/skills/evaluating-llm-quality/schemas/run-config.schema.json:64`

### 2. [高] 実サンプル設定が schema 非準拠

- `configs/run-config.local-openai.yaml` の `judge_id: openai-judge-5.2` は `^[a-z0-9][a-z0-9_-]{0,63}$` に非準拠（`.` を含む）。
- `.claude/skills/.../run-config.hybrid.yaml` の `policy_notes` は `protocol.additionalProperties: false` に反する。

参照:
- `configs/run-config.local-openai.yaml:29`
- `.claude/skills/evaluating-llm-quality/configs/run-config.hybrid.yaml:81`
- `.claude/skills/evaluating-llm-quality/schemas/run-config.schema.json:64`
- `.claude/skills/evaluating-llm-quality/schemas/run-config.schema.json:197`

### 3. [高] 採点仕様（アンカー+中間値許容）と文書/スキーマが不一致

- 実運用方針は `2/4` 許容だが、README表現は「3段階」に見え、judgement schema は `1/3/5` のみ許容。
- 現行データには `4` が含まれており、schema上は不正扱いになる。

参照:
- `README.md:186`
- `.claude/skills/evaluating-llm-quality/schemas/judgement-record.schema.json`
- `data/judgements-local-openai-4o-mini.jsonl:6`

### 4. [中] スキーマファイル名の記述ゆれ

- 文書内の表記 `testcases.schema.json` / `judgements.schema.json` と、実ファイル名 `testcase.schema.json` / `judgement-record.schema.json` が一致していない。

参照:
- `.claude/skills/evaluating-llm-quality/SKILL.md:30`
- `.claude/skills/evaluating-llm-quality/REFERENCE.md:176`
- `.claude/skills/evaluating-llm-quality/REFERENCE.md:179`
- `ARCHITECTURE.md:248`
- `ARCHITECTURE.md:251`

### 5. [中] 再現性メタデータ要件（params_hash）と実装が不一致

- スキル文書は `params_hash` 記録を要求しているが、実装・モデルに該当フィールドがない（`input_hash` はあり）。

参照:
- `.claude/skills/evaluating-llm-quality/SKILL.md:99`
- `.claude/skills/evaluating-llm-quality/REFERENCE.md:17`
- `src/llm_eval/models.py:143`
- `src/llm_eval/stages/inference.py:127`

### 6. [中] Compare 最低要件と出力項目の差

- 文書は「勝率/敗率/同点率」「分散または信頼区間」を最低要件にしているが、実装は `loss_rate` を出力せず、`confidence_intervals` は実質未計算（空）。

参照:
- `.claude/skills/evaluating-llm-quality/SKILL.md:117`
- `.claude/skills/evaluating-llm-quality/SKILL.md:118`
- `src/llm_eval/stages/compare.py:142`
- `src/llm_eval/models.py:256`
- `data/comparison-report-local-openai-4o-mini.json`

## 補足（整合している点）

- 「集計結果が小数になり得る」点は実装と整合している。
- `mean_score` と `weighted_overall` は小数出力を許容するモデル・出力になっている。

