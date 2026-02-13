"""Prompt builders for candidate inference and LLM-as-a-Judge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_eval.models import Constraints, JudgeRef, Testcase

# ── Template loading ──────────────────────────────────────

TEMPLATE_DIRS = [
    Path(".claude/skills/evaluating-llm-quality/templates"),
    Path("templates"),
]

RUBRIC_DIRS = [
    Path(".claude/skills/evaluating-llm-quality/rubrics"),
    Path("rubrics"),
]


def _find_file(name: str, dirs: list[Path]) -> str | None:
    """Search for a file in multiple directories."""
    for d in dirs:
        path = d / name
        if path.exists():
            return path.read_text()
    return None


def load_template(task_type: str) -> str:
    """Load the prompt template for a task type."""
    mapping = {
        "preprocessing": "preprocessing.md",
        "report_generation": "report.md",
        "report_qa": "report_qa.md",
    }
    filename = mapping.get(task_type, f"{task_type}.md")
    content = _find_file(filename, TEMPLATE_DIRS)
    return content or ""


def load_rubric(version: str) -> str:
    """Load a rubric by version name."""
    content = _find_file(f"{version}.md", RUBRIC_DIRS)
    return content or ""


# ── Candidate inference prompt ────────────────────────────


def build_inference_prompt(testcase: Testcase) -> list[dict[str, str]]:
    """Build chat messages for candidate model inference."""
    template = load_template(testcase.task_type)
    constraints = testcase.constraints or Constraints()

    system_parts = ["あなたは高品質な出力を生成するアシスタントです。"]

    if template:
        system_parts.append(f"## 出力テンプレート\n{template}")

    if constraints.output_format:
        fmt = constraints.output_format
        if fmt.type == "json":
            system_parts.append("出力はJSON形式で返してください。")
            if fmt.json_schema_ref:
                system_parts.append(f"JSONスキーマ: {fmt.json_schema_ref}")
        elif fmt.type == "markdown":
            system_parts.append("出力はMarkdown形式で返してください。")
            if fmt.template_ref:
                system_parts.append(f"テンプレート構造: {fmt.template_ref}")

    if constraints.required_points:
        points = "\n".join(f"- {p}" for p in constraints.required_points)
        system_parts.append(f"## 必須要件\n{points}")

    if constraints.forbidden_points:
        points = "\n".join(f"- {p}" for p in constraints.forbidden_points)
        system_parts.append(f"## 禁止事項\n{points}")

    system_msg = "\n\n".join(system_parts)

    user_msg = _format_input(testcase.input)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _format_input(input_data: dict[str, Any]) -> str:
    """Format testcase input dict into a user message string."""
    parts = []
    for key, value in input_data.items():
        if isinstance(value, str):
            parts.append(f"## {key}\n{value}")
        else:
            parts.append(f"## {key}\n{json.dumps(value, ensure_ascii=False)}")
    return "\n\n".join(parts)


# ── Judge prompts ─────────────────────────────────────────


def build_pairwise_judge_prompt(
    testcase: Testcase,
    output_a: str,
    output_b: str,
    label_a: str,
    label_b: str,
    metrics: list[str],
    rubric_version: str,
    scoring_scale: list[int],
) -> list[dict[str, str]]:
    """Build prompt for pairwise judge evaluation."""
    rubric = load_rubric(rubric_version)
    scale_str = "/".join(str(s) for s in scoring_scale)
    metrics_str = ", ".join(metrics)

    system_msg = f"""あなたはLLM出力の品質評価を行う公平なJudgeです。
2つの回答（Answer A / Answer B）を比較し、各指標について{scale_str}で採点してください。

## 評価指標
{metrics_str}

## ルーブリック
{rubric}

## critical_issue 判定
以下のいずれかに該当する回答には `critical_issue: true` を設定してください:
- 重大な事実捏造（高リスク領域で確信を持って誤った主張）
- 安全性違反（有害・差別的内容、機微情報の生成）
- 出力形式が完全に使用不可能
- 指示を大幅に無視（別のタスクへの回答など）

## 出力形式
以下のJSON形式で出力してください:
{{
  "per_metric": {{"metric_id": score, ...}},
  "overall_winner": "A" or "B" or "tie",
  "critical_issue_a": true/false,
  "critical_issue_b": true/false,
  "rationale": "判定理由（日本語）"
}}
"""

    user_msg = f"""## タスク情報
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## 入力
{_format_input(testcase.input)}

## Answer {label_a}
{output_a}

## Answer {label_b}
{output_b}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_absolute_judge_prompt(
    testcase: Testcase,
    output_text: str,
    candidate_label: str,
    metrics: list[str],
    rubric_version: str,
    scoring_scale: list[int],
) -> list[dict[str, str]]:
    """Build prompt for absolute (single-answer) judge evaluation."""
    rubric = load_rubric(rubric_version)
    scale_str = "/".join(str(s) for s in scoring_scale)
    metrics_str = ", ".join(metrics)

    system_msg = f"""あなたはLLM出力の品質評価を行う公平なJudgeです。
回答を各指標について{scale_str}で採点してください。

## 評価指標
{metrics_str}

## ルーブリック
{rubric}

## critical_issue 判定
以下のいずれかに該当する場合は `critical_issue: true` を設定してください:
- 重大な事実捏造（高リスク領域で確信を持って誤った主張）
- 安全性違反（有害・差別的内容、機微情報の生成）
- 出力形式が完全に使用不可能
- 指示を大幅に無視（別のタスクへの回答など）

## 出力形式
以下のJSON形式で出力してください:
{{
  "per_metric": {{"metric_id": score, ...}},
  "overall_score": 平均スコア,
  "critical_issue": true/false,
  "rationale": "判定理由（日本語）"
}}
"""

    user_msg = f"""## タスク情報
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## 入力
{_format_input(testcase.input)}

## 回答
{output_text}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
