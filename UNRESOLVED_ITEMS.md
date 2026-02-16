# 未対応項目メモ

対象: `sample-llm-eval`  
作成日: 2026-02-16

## 概要

前回対応後の再レビューで、以下 2 点が未対応でした。

## 1. [高] `aggregation.method` が未使用

- `RunConfig` では `method` を保持しているが、`compare` 集計処理で分岐利用されていません。
- 現状は `majority_vote` / `worst_case` / `custom` を設定しても実質挙動が変わりません。

参照:
- `src/llm_eval/models.py:76`
- `src/llm_eval/stages/compare.py:57`
- `src/llm_eval/stages/compare.py:119`

## 2. [中] `weighted_overall` が `by_task` / `by_bucket` で未計算

- `overall` には `weights` を渡しているが、グループ集計側では未伝播のため空になります。
- レポート上で全体と内訳の一貫性が崩れます。

参照:
- `src/llm_eval/stages/compare.py:62`
- `src/llm_eval/stages/compare.py:66`
- `src/llm_eval/stages/compare.py:250`

## 推奨対応順

1. `aggregation.method` に応じた集計ロジック分岐を実装
2. `_compute_by_group()` から `_compute_aggregate()` へ `weights` を渡す
