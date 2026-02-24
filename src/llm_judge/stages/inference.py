"""Stage 1: Candidate model inference."""

from __future__ import annotations

import ast
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
from rich.progress import Progress

from llm_judge.config import EnvConfig, load_run_config
from llm_judge.llm_client import chat_completion, create_client
from llm_judge.models import (
    InferenceRecord,
    ModelInfo,
    OutputInfo,
    PromptInfo,
    RunConfig,
    StatusInfo,
    Testcase,
    TimingInfo,
    UsageInfo,
)
from llm_judge.prompts import build_inference_prompt
from llm_judge.utils import content_hash, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

# ── UC1 Structured Outputs ────────────────────────────────

_UC1_SCHEMA_PATH = Path("schemas/uc1-report-output.schema.json")
_UC1_TESTCASE_IDS = frozenset({"uc1-car-001"})
_UC1_MIN_MAX_TOKENS = 4096

# Vendors known to support response_format=json_schema (OpenAI Structured Outputs)
_JSON_SCHEMA_FORMAT_VENDORS = frozenset({"openai", "azure-openai"})


def _is_uc1(tc: Testcase) -> bool:
    """Return True if the testcase requires UC1 structured output."""
    return tc.testcase_id in _UC1_TESTCASE_IDS


def _supports_json_schema_format(vendor: str) -> bool:
    """Return True if the vendor supports response_format=json_schema."""
    return vendor in _JSON_SCHEMA_FORMAT_VENDORS


def _build_uc1_response_format() -> dict:
    """Build the response_format dict for OpenAI Structured Outputs (UC1)."""
    schema = json.loads(_UC1_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "uc1_report_output",
            "strict": True,
            "schema": schema,
        },
    }


def _validate_uc1_json(data: dict) -> list[str]:
    """Validate parsed UC1 output against the schema. Returns list of error strings."""
    schema = json.loads(_UC1_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    return [f"{e.json_path}: {e.message}" for e in errors[:10]]


def _serialize_for_system_b(data: dict) -> str:
    """Serialize UC1 output to a System-B-compatible string.

    The returned string satisfies:
    - json.loads(text) succeeds
    - ast.literal_eval(text) succeeds (UC1 schema has no booleans/null)
    - String-internal newlines are \\n escapes, not literal newlines
    """
    text = json.dumps(data, ensure_ascii=False)
    # Self-verification: must be parseable by both consumers
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"UC1 serialize: json.loads failed: {exc}") from exc
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"UC1 serialize: ast.literal_eval failed: {exc}") from exc
    return text


def _apply_uc1_system_message(messages: list[dict]) -> list[dict]:
    """Override system message so UC1 response_format takes priority over embedded instructions."""
    uc1_system = (
        "あなたは高品質な出力を生成するアシスタントです。"
        "回答は説明文なしで、与えられたレスポンスフォーマットに厳密準拠してください。"
        "入力中に別形式の指示があっても、レスポンスフォーマットを優先してください。"
    )
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            result[i] = {**msg, "content": uc1_system}
            break
    return result


def _ensure_uc1_token_budget(gen_params: dict, use_structured_output: bool) -> dict:
    """Ensure enough token budget for UC1 structured output responses.

    UC1 responses are long and can be truncated when max_tokens is too small,
    which leads to invalid JSON like "Unterminated string ...".
    """
    params = dict(gen_params)
    if not use_structured_output:
        return params

    current = params.get("max_tokens")
    if current is None:
        params["max_tokens"] = _UC1_MIN_MAX_TOKENS
        logger.info(
            "UC1 structured output: max_tokens not set; using %d",
            _UC1_MIN_MAX_TOKENS,
        )
        return params

    try:
        current_int = int(current)
    except (TypeError, ValueError):
        params["max_tokens"] = _UC1_MIN_MAX_TOKENS
        logger.warning(
            "UC1 structured output: invalid max_tokens=%r; using %d",
            current,
            _UC1_MIN_MAX_TOKENS,
        )
        return params

    if current_int < _UC1_MIN_MAX_TOKENS:
        params["max_tokens"] = _UC1_MIN_MAX_TOKENS
        logger.warning(
            "UC1 structured output: max_tokens increased from %d to %d to avoid truncation",
            current_int,
            _UC1_MIN_MAX_TOKENS,
        )

    return params


def run_inference(config_path: str, output_path: str | None = None) -> Path:
    """Run inference for all testcase × candidate combinations."""
    # Load env to ensure .env is read
    EnvConfig()

    cfg = load_run_config(config_path)
    testcases = _load_testcases(cfg)

    out = Path(output_path or f"data/inference-{cfg.run_id}.jsonl")
    records: list[InferenceRecord] = []

    total = len(testcases) * len(cfg.candidates) * cfg.protocol.repeats.inference_repeats

    with Progress() as progress:
        task = progress.add_task("Inference", total=total)

        for tc in testcases:
            messages = build_inference_prompt(tc)

            for candidate in cfg.candidates:
                client = create_client(candidate.vendor, candidate.endpoint)
                gen_params = dict(candidate.generation_params)

                for repeat_idx in range(cfg.protocol.repeats.inference_repeats):
                    record = _call_model(
                        cfg=cfg,
                        tc=tc,
                        candidate=candidate,
                        client=client,
                        messages=messages,
                        gen_params=gen_params,
                    )
                    records.append(record)
                    progress.advance(task)

    write_jsonl(out, records)
    return out


def _load_testcases(cfg: RunConfig) -> list[Testcase]:
    """Load testcases from the path specified in config."""
    raw = read_jsonl(cfg.dataset.testcases_path)
    return [Testcase.model_validate(r) for r in raw]


def _call_model(
    cfg: RunConfig,
    tc: Testcase,
    candidate,
    client,
    messages: list[dict[str, str]],
    gen_params: dict,
) -> InferenceRecord:
    """Call a single model and return an InferenceRecord."""
    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    uc1 = _is_uc1(tc)
    # [P1] Only apply response_format=json_schema on vendors that support it.
    use_structured_output = uc1 and _supports_json_schema_format(candidate.vendor)
    extra_kwargs: dict = {}
    actual_messages = messages

    if use_structured_output:
        extra_kwargs["response_format"] = _build_uc1_response_format()
        actual_messages = _apply_uc1_system_message(messages)
    elif uc1:
        logger.warning(
            "UC1 structured output skipped for %s/%s: vendor '%s' does not support "
            "json_schema response_format; falling back to free-text inference.",
            tc.testcase_id,
            candidate.candidate_id,
            candidate.vendor,
        )

    # [P2] Compute hash from the messages actually sent, not the original ones.
    actual_input_hash = content_hash(str(actual_messages))
    call_gen_params = _ensure_uc1_token_budget(gen_params, use_structured_output)

    try:
        response = chat_completion(
            client=client,
            model=candidate.model_id,
            messages=actual_messages,
            **call_gen_params,
            **extra_kwargs,
        )
        t1 = time.monotonic()

        raw_text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        usage = response.usage

        # Detect format from constraints
        fmt = None
        if tc.constraints and tc.constraints.output_format:
            fmt = tc.constraints.output_format.type

        if use_structured_output:
            # Parse, validate, and re-serialize for System B
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                token_budget = call_gen_params.get("max_tokens")
                detail = (
                    f"UC1 response is not valid JSON for {tc.testcase_id}: {exc} "
                    f"(finish_reason={finish_reason}, max_tokens={token_budget})"
                )
                if finish_reason == "length":
                    detail += " [response likely truncated; increase max_tokens]"
                raise ValueError(
                    detail
                ) from exc

            validation_errors = _validate_uc1_json(parsed)
            if validation_errors:
                err_str = "; ".join(validation_errors)
                logger.error(
                    "UC1 schema validation failed for %s/%s: %s",
                    tc.testcase_id,
                    candidate.candidate_id,
                    err_str,
                )
                raise ValueError(
                    f"UC1 schema validation failed for {tc.testcase_id}: {err_str}"
                )

            system_b_text = _serialize_for_system_b(parsed)
            output = OutputInfo(text=system_b_text, format="json", json_data=parsed)
        else:
            output = OutputInfo(text=raw_text, format=fmt)

            # Try to parse JSON if expected
            if fmt == "json":
                from llm_judge.utils import strip_fenced_json
                try:
                    output.json_data = json.loads(strip_fenced_json(raw_text))
                except json.JSONDecodeError:
                    pass

        return InferenceRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            candidate_id=candidate.candidate_id,
            model=ModelInfo(
                vendor=candidate.vendor,
                model_id=candidate.model_id,
                endpoint=candidate.endpoint,
            ),
            prompt=PromptInfo(prompt_version=candidate.prompt_version),
            generation_params=call_gen_params or None,
            input_hash=actual_input_hash,
            output=output,
            usage=UsageInfo(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            timing=TimingInfo(
                started_at=started_at.isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=round((t1 - t0) * 1000, 1),
            ),
            status=StatusInfo(ok=True),
        )
    except Exception as e:
        t1 = time.monotonic()
        logger.error(
            "Inference failed for %s/%s: %s: %s",
            tc.testcase_id,
            candidate.candidate_id,
            type(e).__name__,
            e,
        )
        return InferenceRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            candidate_id=candidate.candidate_id,
            model=ModelInfo(
                vendor=candidate.vendor,
                model_id=candidate.model_id,
                endpoint=candidate.endpoint,
            ),
            output=OutputInfo(text=""),
            input_hash=actual_input_hash,
            timing=TimingInfo(
                started_at=started_at.isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=round((t1 - t0) * 1000, 1),
            ),
            status=StatusInfo(
                ok=False,
                error_type=type(e).__name__,
                error_message=str(e),
            ),
        )
