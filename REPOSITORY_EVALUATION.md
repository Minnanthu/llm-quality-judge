# リポジトリ評価レポート

対象: `sample-llm-eval`  
評価日: 2026-02-16

## 総評

設計は明確で実行可能ですが、評価結果の信頼性に影響するロジック上の不整合があり、厳密な比較評価基盤としては改善が必要です。  
体感評価: **6.5 / 10**

## 主要な指摘（重大度順）

### 1. 高: aggregation 設定が実装で未使用

- `protocol.aggregation.method` / `weights` を設定しても集計結果に反映されません。
- 影響: 設計意図と実際の集計が乖離し、設定変更の意味が失われます。

参照:
- `src/llm_eval/models.py:75`
- `src/llm_eval/models.py:77`
- `src/llm_eval/stages/compare.py:57`

### 2. 高: JSONスキーマ制約の与え方が弱い

- 推論プロンプトではスキーマ内容ではなくパス文字列のみを提示しています。
- 影響: モデルが期待キーを推定しづらく、スキーマ不一致が増えます。

参照:
- `src/llm_eval/prompts.py:105`

### 3. 高: pairwise の mean_score 集計ロジックが候補比較として不正確

- pairwise 判定の `per_metric` を各候補に同値で加算しており、候補間の差分が失われます。
- 影響: 指標表が比較結果を適切に表現しません。

参照:
- `src/llm_eval/prompts.py:177`
- `src/llm_eval/stages/compare.py:145`

### 4. 中: inference_repeats が pairwise 判定では実質無視

- pairwise 前に候補ごとに1件へ間引く実装になっています。
- 影響: 反復実行による分散確認という設計意図とずれます。

参照:
- `src/llm_eval/stages/judge.py:54`
- `src/llm_eval/stages/judge.py:57`
- `src/llm_eval/stages/inference.py:52`

### 5. 中: fenced JSON を不正JSON扱いしやすい

- 出力が ```json ... ``` 形式だと JSON パース失敗になります。
- 影響: Autocheck で偽陰性が増えます。

参照:
- `src/llm_eval/stages/inference.py:111`
- `src/llm_eval/schema_validation.py:55`

### 6. 中: ローカル設定の候補が同一モデル

- `run-config.local-openai.yaml` の2候補がどちらも `gpt-4o` です。
- 影響: 比較評価の妥当性が下がります。

参照:
- `configs/run-config.local-openai.yaml:12`
- `configs/run-config.local-openai.yaml:21`

## 良い点

- 4ステージ分離と CLI 設計が明快で、部分再実行が容易。
- Pydantic モデルとスキーマの対応が整理されており拡張しやすい。
- 進捗表示・失敗時継続処理など、運用面の実用性が高い。

参照:
- `src/llm_eval/cli.py:19`
- `src/llm_eval/models.py:95`
- `src/llm_eval/stages/inference.py:41`

## 実行確認

- `uv run llm-eval --help` 正常
- `uv run python -m compileall src` 正常
- `uv run llm-eval autocheck --config configs/run-config.local-openai.yaml --inference data/inference-local-openai-4o-mini.jsonl --output /tmp/autocheck-check.jsonl` 正常
- `uv run llm-eval compare --config configs/run-config.local-openai.yaml --judgements data/judgements-local-openai-4o-mini.jsonl --output /tmp/comparison-check.json` 正常

## 改善優先度

1. aggregation 実装反映（`method` / `weights`）
2. pairwise の `mean_score` 定義・集計方法の修正
3. 推論時のスキーマ提示方法改善（内容埋め込みや JSON mode 強制）
4. repeats の扱い（pairwise と absolute で意図を明確化）
