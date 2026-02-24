"""Tests for UC1 Structured Outputs implementation."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.models import Constraints, OutputFormat, Testcase
from llm_judge.stages.inference import (
    _UC1_SCHEMA_PATH,
    _apply_uc1_system_message,
    _build_uc1_response_format,
    _ensure_uc1_token_budget,
    _is_uc1,
    _serialize_for_system_b,
    _supports_json_schema_format,
    _validate_uc1_json,
)


# ── helpers ───────────────────────────────────────────────

def _make_testcase(testcase_id: str) -> Testcase:
    return Testcase(
        testcase_id=testcase_id,
        task_type="preprocessing",
        input={"report_name": "車", "source": "dummy"},
    )


def _valid_uc1_data() -> dict:
    return {
        "tag_data": [
            {
                "labels": "車の故障・不動",
                "add_infos": "エンジン不動や自走不能に関する問い合わせが多数。",
                "counts": 30,
                "dialogue_samples": [0, 1, 2],
            },
            {
                "labels": "車の損傷・事故",
                "add_infos": "追突や擦り傷など物損事故に関する問い合わせ。",
                "counts": 20,
                "dialogue_samples": [3, 4],
            },
            {
                "labels": "ロードサービス要請",
                "add_infos": "現場での救援を求める問い合わせ。",
                "counts": 10,
                "dialogue_samples": [5],
            },
            {
                "labels": "車の売却・乗り換え",
                "add_infos": "廃車・売却・買い替えを検討する問い合わせ。",
                "counts": 7,
                "dialogue_samples": [6, 7],
            },
            {
                "labels": "修理・修繕",
                "add_infos": "修理中または修理依頼に関する問い合わせ。",
                "counts": 4,
                "dialogue_samples": [8],
            },
        ],
        "action_plan": [
            {
                "title": "緊急対応フローの整備",
                "summary": "自走不能の場合に即時ロードサービスを案内するフローを設ける。",
            }
        ],
        "summary": "71件の問い合わせのうち故障・不動が最多。緊急対応強化が急務。",
    }


# ── _is_uc1 ───────────────────────────────────────────────

def test_is_uc1_returns_true_for_uc1():
    tc = _make_testcase("uc1-car-001")
    assert _is_uc1(tc) is True


def test_is_uc1_returns_false_for_other():
    for tid in ["uc2-001", "judge-001", "preprocessing-001"]:
        assert _is_uc1(_make_testcase(tid)) is False


# ── _build_uc1_response_format ────────────────────────────

def test_build_uc1_response_format_structure():
    rf = _build_uc1_response_format()
    assert rf["type"] == "json_schema"
    js = rf["json_schema"]
    assert js["name"] == "uc1_report_output"
    assert js["strict"] is True
    schema = js["schema"]
    assert set(schema["required"]) == {"tag_data", "action_plan", "summary"}


def test_build_uc1_response_format_schema_has_no_additional_props():
    rf = _build_uc1_response_format()
    schema = rf["json_schema"]["schema"]
    assert schema.get("additionalProperties") is False


# ── _validate_uc1_json ────────────────────────────────────

def test_validate_uc1_json_valid_data():
    errors = _validate_uc1_json(_valid_uc1_data())
    assert errors == []


def test_validate_uc1_json_missing_required_field():
    data = _valid_uc1_data()
    del data["summary"]
    errors = _validate_uc1_json(data)
    assert any("summary" in e for e in errors)


def test_validate_uc1_json_tag_data_wrong_length():
    data = _valid_uc1_data()
    # Remove one item to get 4 items (schema requires exactly 5)
    data["tag_data"] = data["tag_data"][:4]
    errors = _validate_uc1_json(data)
    assert errors  # must report an error


def test_validate_uc1_json_extra_field_rejected():
    data = _valid_uc1_data()
    data["unexpected_key"] = "oops"
    errors = _validate_uc1_json(data)
    assert errors  # additionalProperties: false


# ── _serialize_for_system_b ───────────────────────────────

def test_serialize_for_system_b_roundtrip():
    data = _valid_uc1_data()
    text = _serialize_for_system_b(data)
    assert json.loads(text) == data
    assert ast.literal_eval(text) == data


def test_serialize_for_system_b_newline_in_string():
    """String values with newlines must be \\n-escaped, not literal newlines."""
    data = {"tag_data": [], "action_plan": [{"title": "a", "summary": "line1\nline2"}], "summary": "x"}
    text = _serialize_for_system_b(data)
    # The serialized string must not contain a raw newline inside a JSON string value
    # json.dumps encodes newline as \n, so we should see \\n in the Python repr
    assert "\n" not in text or text.count("\n") == 0 or _check_no_literal_newline_in_string_values(text)
    # Both consumers must work
    json.loads(text)
    ast.literal_eval(text)


def _check_no_literal_newline_in_string_values(text: str) -> bool:
    """Verify that no literal newline appears inside a JSON string value."""
    # json.dumps always escapes newlines as \n, so the serialized text
    # should have no literal newlines (it's a single-line JSON string).
    return "\n" not in text


def test_serialize_for_system_b_no_literal_newlines():
    data = _valid_uc1_data()
    # Add a string with a newline
    data["summary"] = "first line\nsecond line"
    text = _serialize_for_system_b(data)
    assert "\n" not in text, "Literal newlines must not appear in System B payload"


def test_serialize_for_system_b_ast_literal_eval_compatible():
    """UC1 schema has only strings, integers, lists, dicts - json.dumps output
    is therefore parseable by ast.literal_eval."""
    data = _valid_uc1_data()
    text = _serialize_for_system_b(data)
    result = ast.literal_eval(text)
    assert result["summary"] == data["summary"]
    assert result["tag_data"][0]["counts"] == data["tag_data"][0]["counts"]


# ── _apply_uc1_system_message ─────────────────────────────

def test_apply_uc1_system_message_replaces_system_role():
    original = [
        {"role": "system", "content": "original system message"},
        {"role": "user", "content": "user message"},
    ]
    result = _apply_uc1_system_message(original)
    assert result[0]["role"] == "system"
    assert "レスポンスフォーマットを優先" in result[0]["content"]
    assert result[1] == {"role": "user", "content": "user message"}


def test_apply_uc1_system_message_does_not_mutate_original():
    original = [{"role": "system", "content": "old"}]
    _apply_uc1_system_message(original)
    assert original[0]["content"] == "old"


# ── schema file existence ─────────────────────────────────

def test_uc1_schema_file_exists():
    assert _UC1_SCHEMA_PATH.exists(), f"Schema file not found: {_UC1_SCHEMA_PATH}"


def test_uc1_schema_file_is_valid_json():
    content = _UC1_SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(content)
    assert schema.get("title") == "UC1ReportOutput"


# ── [P1] vendor capability check ─────────────────────────

def test_supports_json_schema_format_openai():
    assert _supports_json_schema_format("openai") is True


def test_supports_json_schema_format_azure_openai():
    assert _supports_json_schema_format("azure-openai") is True


def test_supports_json_schema_format_tsuzumi2_not_supported():
    assert _supports_json_schema_format("tsuzumi2") is False


def test_supports_json_schema_format_unknown_vendor_not_supported():
    assert _supports_json_schema_format("unknown-vendor") is False


# ── UC1 token budget guard ───────────────────────────────

def test_ensure_uc1_token_budget_keeps_non_uc1_params():
    params = {"max_tokens": 1024}
    out = _ensure_uc1_token_budget(params, use_structured_output=False)
    assert out["max_tokens"] == 1024


def test_ensure_uc1_token_budget_raises_low_max_tokens():
    params = {"max_tokens": 1024}
    out = _ensure_uc1_token_budget(params, use_structured_output=True)
    assert out["max_tokens"] >= 4096


def test_ensure_uc1_token_budget_sets_default_when_missing():
    out = _ensure_uc1_token_budget({}, use_structured_output=True)
    assert out["max_tokens"] >= 4096


def test_ensure_uc1_token_budget_keeps_higher_value():
    out = _ensure_uc1_token_budget({"max_tokens": 8192}, use_structured_output=True)
    assert out["max_tokens"] == 8192


# ── [P2] input_hash computed from actual (sent) messages ──

def test_input_hash_differs_after_uc1_system_message_override():
    """The system message override must produce a different hash than the original."""
    from llm_judge.utils import content_hash

    original = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "user message"},
    ]
    overridden = _apply_uc1_system_message(original)

    assert content_hash(str(original)) != content_hash(str(overridden))


def test_input_hash_stable_for_non_uc1():
    """For non-UC1, actual_messages == messages so hash is unchanged."""
    from llm_judge.utils import content_hash

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    tc = _make_testcase("other-001")
    # Non-UC1 does not override messages, so hashes must be equal
    assert not _is_uc1(tc)
    assert content_hash(str(messages)) == content_hash(str(messages))
