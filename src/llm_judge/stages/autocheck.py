"""Stage 2: Automated format and schema checks."""

from __future__ import annotations

import json
from pathlib import Path
from rich.progress import Progress

from llm_judge.config import load_run_config
from llm_judge.models import (
    AutoCheckRecord,
    Checks,
    FormatCompliance,
    InferenceRecord,
    JsonSchemaValidation,
    Testcase,
)
from llm_judge.schema_validation import (
    SchemaValidationResult,
    validate_output_against_testcase_schema,
)
from llm_judge.utils import read_jsonl, write_jsonl


def run_autocheck(
    config_path: str,
    inference_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Run automated checks on inference outputs."""
    cfg = load_run_config(config_path)

    inf_path = inference_path or f"data/inference-{cfg.run_id}.jsonl"
    raw_inferences = read_jsonl(inf_path)
    inferences = [InferenceRecord.model_validate(r) for r in raw_inferences]

    raw_testcases = read_jsonl(cfg.dataset.testcases_path)
    tc_map = {
        tc["testcase_id"]: Testcase.model_validate(tc) for tc in raw_testcases
    }

    out = Path(output_path or f"data/autocheck-{cfg.run_id}.jsonl")
    records: list[AutoCheckRecord] = []

    with Progress() as progress:
        task = progress.add_task("Autocheck", total=len(inferences))

        for inf in inferences:
            tc = tc_map.get(inf.testcase_id)
            checks = _run_checks(inf, tc)
            records.append(
                AutoCheckRecord(
                    run_id=cfg.run_id,
                    testcase_id=inf.testcase_id,
                    candidate_id=inf.candidate_id,
                    checks=checks,
                )
            )
            progress.advance(task)

    write_jsonl(out, records)
    return out


def _run_checks(inf: InferenceRecord, tc: Testcase | None) -> Checks:
    """Run all applicable checks on a single inference record."""
    schema_result = validate_output_against_testcase_schema(tc, inf.output.text)

    fmt_check = _check_format_compliance(inf, tc, schema_result)
    schema_check = _check_json_schema(schema_result)

    return Checks(
        format_compliance=fmt_check,
        json_schema_validation=schema_check,
    )


def _check_format_compliance(
    inf: InferenceRecord,
    tc: Testcase | None,
    schema_result: SchemaValidationResult | None,
) -> FormatCompliance:
    """Check if output matches the expected format."""
    if not tc or not tc.constraints or not tc.constraints.output_format:
        return FormatCompliance(passed=True, details="No format constraint specified")

    expected = tc.constraints.output_format.type
    output_text = inf.output.text.strip()

    if not output_text:
        return FormatCompliance(passed=False, details="Empty output")

    if expected == "json":
        if schema_result is not None:
            if schema_result.passed:
                return FormatCompliance(passed=True, details="Valid JSON and schema compliant")
            return FormatCompliance(
                passed=False,
                details=f"Schema validation failed: {'; '.join(schema_result.errors[:3])}",
            )
        try:
            json.loads(output_text)
            return FormatCompliance(passed=True, details="Valid JSON")
        except json.JSONDecodeError as e:
            return FormatCompliance(passed=False, details=f"Invalid JSON: {e}")

    if expected == "markdown":
        has_heading = output_text.startswith("#") or "\n#" in output_text
        if has_heading:
            return FormatCompliance(passed=True, details="Contains markdown headings")
        return FormatCompliance(
            passed=False, details="No markdown headings found"
        )

    # free_text: always passes
    return FormatCompliance(passed=True, details=f"Format: {expected}")


def _check_json_schema(
    schema_result: SchemaValidationResult | None,
) -> JsonSchemaValidation | None:
    """Validate output against a referenced JSON schema if applicable."""
    if schema_result is None:
        return None

    return JsonSchemaValidation(
        schema_ref=schema_result.schema_ref,
        passed=schema_result.passed,
        errors=schema_result.errors,
    )
