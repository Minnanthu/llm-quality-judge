# 依頼内容: UC1出力を Structured Outputs 化しつつ、System B 互換を維持する

## 背景
- UC1の現行出力は「Python辞書型を指示した自由生成」で、LLM変更時にブレる。
- 今後は `response_format=json_schema` を使って構造を強制したい。
- ただし System B（他社）は改修困難。受け取り互換を壊せない。

## 目的
1. UC1のLLM出力を `response_format=json_schema (strict)` で生成する。
2. System B向けペイロードは「Python辞書パースでも通る互換性」を維持する。
3. 既存の非UC1ケースの挙動は変えない。

## 実装要件（必須）
1. UC1 (`testcase_id=uc1-car-001`) の推論呼び出しで `response_format` を指定すること。
2. スキーマは `/Users/kegasawa/git/llm-quality-judge/schemas/uc1-report-output.schema.json` を使用すること。
3. 受信後は必ず JSON としてパースし、スキーマ適合を確認すること。
4. System B向けに出す最終文字列は、以下を満たすこと。
   - `json.loads` でパース可能
   - `ast.literal_eval` でもパース可能（UC1スキーマ範囲内）
5. 文字列内改行は `\\n` エスケープで保持し、`バックスラッシュ + 実改行` は禁止すること。
6. 失敗時は明確なエラーログを出し、壊れたペイロードを下流へ流さないこと。

## 実装方針（推奨）
1. 推論処理に UC1専用の `response_format` 生成関数を追加。
2. LLM応答を `json.loads` → Pydantic/JSON Schema検証。
3. System B向け整形関数を追加（決定的シリアライズ）。
4. 送信前に `json.loads` と `ast.literal_eval` の両方で自己検証。

## 変更対象の目安
1. `/Users/kegasawa/git/llm-quality-judge/src/llm_judge/stages/inference.py`
2. 必要なら新規ユーティリティ（例: structured output/互換整形）
3. テスト追加（新規 `tests/` 作成可）

## 作成する schema
以下を `/Users/kegasawa/git/llm-quality-judge/schemas/uc1-report-output.schema.json` として保存すること。

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.invalid/schemas/uc1-report-output.schema.json",
  "title": "UC1ReportOutput",
  "type": "object",
  "required": ["tag_data", "action_plan", "summary"],
  "additionalProperties": false,
  "properties": {
    "tag_data": {
      "type": "array",
      "minItems": 5,
      "maxItems": 5,
      "items": {
        "type": "object",
        "required": ["labels", "add_infos", "counts", "dialogue_samples"],
        "additionalProperties": false,
        "properties": {
          "labels": { "type": "string", "minLength": 1, "maxLength": 30 },
          "add_infos": { "type": "string", "minLength": 1, "maxLength": 140 },
          "counts": { "type": "integer", "minimum": 0, "maximum": 71 },
          "dialogue_samples": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": { "type": "integer", "minimum": 0, "maximum": 70 }
          }
        }
      }
    },
    "action_plan": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["title", "summary"],
        "additionalProperties": false,
        "properties": {
          "title": { "type": "string", "minLength": 1 },
          "summary": { "type": "string", "minLength": 1 }
        }
      }
    },
    "summary": { "type": "string", "minLength": 1, "maxLength": 200 }
  }
}
```

## 推論 API 呼び出しで使う response_format
UC1 の場合、以下を `chat.completions.create(...)` に渡すこと。

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "uc1_report_output",
        "strict": True,
        "schema": json.loads(
            Path("/Users/kegasawa/git/llm-quality-judge/schemas/uc1-report-output.schema.json")
            .read_text(encoding="utf-8")
        ),
    },
}
```

## 受け入れ条件
1. UC1出力で `response_format=json_schema` が実際に使われている。
2. 生成結果がスキーマ適合する。
3. System B向け最終文字列が `json.loads` と `ast.literal_eval` の双方で成功する。
4. `data/uc1-sample.txt` で問題だった `Invalid \escape` 系を再発させない。
5. 非UC1ケースの既存挙動に回帰がない。

## 成果物
1. 変更コード
2. 追加テスト
3. 実行ログ（最低限: 対象テストPASS）
4. 変更点サマリ（何を、なぜ変えたか）
