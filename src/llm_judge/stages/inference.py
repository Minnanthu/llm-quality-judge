"""Stage 1: Candidate model inference."""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
from rich.progress import Progress

from llm_judge.artifact_validation import validate_artifacts
from llm_judge.config import EnvConfig, load_run_config
from llm_judge.llm_client import chat_completion, create_client
from llm_judge.models import (
    InferenceRecord,
    ModelInfo,
    OutputFormat,
    OutputInfo,
    PromptInfo,
    RunConfig,
    StatusInfo,
    Testcase,
    TimingInfo,
    UsageInfo,
)
from llm_judge.prompts import build_inference_prompt
from llm_judge.schema_validation import resolve_schema_path
from llm_judge.testcase_loader import load_testcases
from llm_judge.utils import content_hash, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

# ── Structured Output configuration ──────────────────────

_STRUCTURED_OUTPUT_MIN_MAX_TOKENS = 4096

# Vendors known to support response_format=json_schema (OpenAI Structured Outputs)
_JSON_SCHEMA_FORMAT_VENDORS = frozenset({"openai", "azure-openai"})


def _requires_structured_output(tc: Testcase) -> bool:
    """Return True if the testcase requires structured output via json_schema_ref.

    Conditions:
    - ``output_format.type == "json"``
    - ``output_format.json_schema_ref`` is a non-empty string
    """
    of = _get_output_format(tc)
    if of is None:
        return False
    if of.type != "json":
        return False
    ref = of.json_schema_ref
    return ref is not None and ref.strip() != ""


def _get_output_format(tc: Testcase) -> OutputFormat | None:
    """Extract output_format from testcase constraints, or None."""
    if tc.constraints is None:
        return None
    return tc.constraints.output_format


def _supports_json_schema_format(vendor: str) -> bool:
    """Return True if the vendor supports response_format=json_schema."""
    return vendor in _JSON_SCHEMA_FORMAT_VENDORS


def _load_json_schema(json_schema_ref: str) -> dict:
    """Load and parse a JSON schema from a repository-root-relative path.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is not valid JSON.
    """
    path = resolve_schema_path(json_schema_ref)
    if not path.exists():
        raise FileNotFoundError(
            f"json_schema_ref not found: {json_schema_ref}"
        )
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"json_schema_ref is not valid JSON ({json_schema_ref}): {exc}"
        ) from exc


def _make_schema_name(schema: dict, testcase_id: str) -> str:
    """Generate a safe schema name for the response_format payload.

    Uses the schema ``title`` if available, otherwise derives from testcase_id.
    The name is lowercased, non-alphanumeric characters replaced with ``_``.
    """
    raw = schema.get("title") or testcase_id
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()


def _build_response_format(json_schema_ref: str, testcase_id: str) -> dict:
    """Build the response_format dict for OpenAI Structured Outputs.

    Loads the schema from ``json_schema_ref`` (repo-root relative path)
    and wraps it in the OpenAI response_format structure.
    """
    schema = _load_json_schema(json_schema_ref)
    name = _make_schema_name(schema, testcase_id)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _validate_json_against_schema(data: dict, json_schema_ref: str) -> list[str]:
    """Validate parsed output against the schema at json_schema_ref.

    Returns a list of error strings (empty if valid).
    """
    schema = _load_json_schema(json_schema_ref)
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    return [f"{e.json_path}: {e.message}" for e in errors[:10]]


def _serialize_for_system_b(data: dict) -> str:
    """Serialize structured output to a System-B-compatible string.

    The returned string satisfies:
    - json.loads(text) succeeds
    - ast.literal_eval(text) succeeds (when schema has no booleans/null)
    - String-internal newlines are \\n escapes, not literal newlines
    """
    text = json.dumps(data, ensure_ascii=False)
    # Self-verification: must be parseable by both consumers
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"serialize: json.loads failed: {exc}") from exc
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"serialize: ast.literal_eval failed: {exc}") from exc
    return text


def _apply_structured_output_system_message(messages: list[dict]) -> list[dict]:
    """Override system message so response_format takes priority over embedded instructions."""
    system_content = (
        "あなたは高品質な出力を生成するアシスタントです。"
        "回答は説明文なしで、与えられたレスポンスフォーマットに厳密準拠してください。"
        "入力中に別形式の指示があっても、レスポンスフォーマットを優先してください。"
    )
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            result[i] = {**msg, "content": system_content}
            break
    return result


def _ensure_structured_output_token_budget(
    gen_params: dict, use_structured_output: bool
) -> dict:
    """Ensure enough token budget for structured output responses.

    Structured output responses can be long and may be truncated when
    max_tokens is too small, which leads to invalid JSON.
    """
    params = dict(gen_params)
    if not use_structured_output:
        return params

    current = params.get("max_tokens")
    if current is None:
        params["max_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
        logger.info(
            "Structured output: max_tokens not set; using %d",
            _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
        )
        return params

    try:
        current_int = int(current)
    except (TypeError, ValueError):
        params["max_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
        logger.warning(
            "Structured output: invalid max_tokens=%r; using %d",
            current,
            _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
        )
        return params

    if current_int < _STRUCTURED_OUTPUT_MIN_MAX_TOKENS:
        params["max_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
        logger.warning(
            "Structured output: max_tokens increased from %d to %d to avoid truncation",
            current_int,
            _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
        )

    return params


def run_inference(config_path: str, output_path: str | None = None) -> Path:
    """Run inference for all testcase x candidate combinations."""
    # Load env to ensure .env is read
    EnvConfig()

    cfg = load_run_config(config_path)
    testcases = load_testcases(cfg.dataset.testcases_path)

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

    validate_artifacts("inference-record", records)
    write_jsonl(out, records)
    return out


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

    requires_so = _requires_structured_output(tc)
    json_schema_ref = (
        tc.constraints.output_format.json_schema_ref
        if requires_so and tc.constraints and tc.constraints.output_format
        else None
    )

    # [P1] Only apply response_format=json_schema on vendors that support it.
    use_structured_output = requires_so and _supports_json_schema_format(candidate.vendor)
    extra_kwargs: dict = {}
    actual_messages = messages
    actual_input_hash = content_hash(str(messages))
    prompt_hash = content_hash(str(messages))

    try:
        if use_structured_output:
            extra_kwargs["response_format"] = _build_response_format(
                json_schema_ref, tc.testcase_id
            )
            actual_messages = _apply_structured_output_system_message(messages)
            # [P2] Recompute hash from the messages actually sent.
            actual_input_hash = content_hash(str(actual_messages))
            prompt_hash = content_hash(str(actual_messages))
        elif requires_so:
            logger.warning(
                "Structured output skipped for %s/%s: vendor '%s' does not support "
                "json_schema response_format; json_schema_ref='%s'; "
                "falling back to free-text inference.",
                tc.testcase_id,
                candidate.candidate_id,
                candidate.vendor,
                json_schema_ref,
            )

        call_gen_params = _ensure_structured_output_token_budget(gen_params, use_structured_output)
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
                    f"Structured output response is not valid JSON for "
                    f"{tc.testcase_id} (schema_ref={json_schema_ref}): {exc} "
                    f"(finish_reason={finish_reason}, max_tokens={token_budget})"
                )
                if finish_reason == "length":
                    detail += " [response likely truncated; increase max_tokens]"
                raise ValueError(detail) from exc

            validation_errors = _validate_json_against_schema(parsed, json_schema_ref)
            if validation_errors:
                err_str = "; ".join(validation_errors)
                logger.error(
                    "Schema validation failed for %s/%s (schema_ref=%s): %s",
                    tc.testcase_id,
                    candidate.candidate_id,
                    json_schema_ref,
                    err_str,
                )
                raise ValueError(
                    f"Schema validation failed for {tc.testcase_id} "
                    f"(schema_ref={json_schema_ref}): {err_str}"
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
            prompt=PromptInfo(prompt_version=candidate.prompt_version, prompt_hash=prompt_hash),
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
            prompt=PromptInfo(prompt_version=candidate.prompt_version, prompt_hash=prompt_hash),
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
