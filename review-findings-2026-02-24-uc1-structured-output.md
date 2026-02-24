# レビュー指摘（2026-02-24）UC1 Structured Outputs 対応

## 重要指摘

1. `[P1]` UC1 で `response_format=json_schema` を無条件適用しており、ベンダー非対応時に推論が全落ちするリスクがあります。  
   - 根拠:
     - `/Users/kegasawa/git/llm-quality-judge/src/llm_judge/stages/inference.py:165`  
       UC1 の場合に常に `response_format` を付与している。
     - `/Users/kegasawa/git/llm-quality-judge/configs/run-config.yaml:10`  
       デフォルト構成に `tsuzumi2` 候補が含まれている。
   - 影響:
     - `json_schema` 非対応エンドポイントでは API エラーになり、`status.ok=false` + 空出力が増える。
     - 以降ステージ（autocheck / judge / compare）の品質が劣化する。
   - 対応案:
     - UC1 の `response_format` 適用を vendor/model capability で分岐する。
     - もしくは run-config で UC1 対象候補を JSON Schema 対応モデルに限定する。

2. `[P2]` UC1 だけ `input_hash` が実送信プロンプトと一致しません。  
   - 根拠:
     - `/Users/kegasawa/git/llm-quality-judge/src/llm_judge/stages/inference.py:119`  
       ハッシュは元メッセージで計算。
     - `/Users/kegasawa/git/llm-quality-judge/src/llm_judge/stages/inference.py:167`  
       送信直前に system メッセージを差し替え。
   - 影響:
     - 再現性・監査性の指標として `input_hash` が不正確になる。
   - 対応案:
     - 実際に送信する `actual_messages` でハッシュを計算する。
     - あるいは `prompt_hash_original` と `prompt_hash_sent` を分けて記録する。

## 確認事項

1. テスト実行結果  
   - コマンド: `uv run pytest -q`  
   - 結果: `16 passed, 1 warning`

2. 実装方針自体の妥当性  
   - `uc1-report-output.schema.json` による UC1 構造固定は実装済み。  
   - System B 互換を意識した `json.loads` / `ast.literal_eval` 両立確認も実装済み。  
   - 上記 2 点の指摘を解消すれば、運用上の安定性はさらに上げられる。
