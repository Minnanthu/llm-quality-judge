"""Stage 2: Automated format and schema checks."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from rich.progress import Progress

from llm_eval.config import load_run_config
from llm_eval.models import (
    AutoCheckRecord,
    Checks,
    FormatCompliance,
    InferenceRecord,
    JsonSchemaValidation,
    Testcase,
)
from llm_eval.utils import read_jsonl, write_jsonl


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
    fmt_check = _check_format_compliance(inf, tc)
    schema_check = _check_json_schema(inf, tc)

    return Checks(
        format_compliance=fmt_check,
        json_schema_validation=schema_check,
    )


def _check_format_compliance(
    inf: InferenceRecord, tc: Testcase | None
) -> FormatCompliance:
    """Check if output matches the expected format."""
    if not tc or not tc.constraints or not tc.constraints.output_format:
        return FormatCompliance(passed=True, details="No format constraint specified")

    expected = tc.constraints.output_format.type
    output_text = inf.output.text.strip()

    if not output_text:
        return FormatCompliance(passed=False, details="Empty output")

    if expected == "json":
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
    inf: InferenceRecord, tc: Testcase | None
) -> JsonSchemaValidation | None:
    """Validate output against a referenced JSON schema if applicable."""
    if not tc or not tc.constraints or not tc.constraints.output_format:
        return None

    schema_ref = tc.constraints.output_format.json_schema_ref
    if not schema_ref:
        return None

    # Try to load the schema file
    schema_path = Path(schema_ref)
    if not schema_path.exists():
        return JsonSchemaValidation(
            schema_ref=schema_ref,
            passed=False,
            errors=[f"Schema file not found: {schema_ref}"],
        )

    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as e:
        return JsonSchemaValidation(
            schema_ref=schema_ref,
            passed=False,
            errors=[f"Invalid schema file: {e}"],
        )

    # Parse the output as JSON
    try:
        data = json.loads(inf.output.text.strip())
    except json.JSONDecodeError:
        return JsonSchemaValidation(
            schema_ref=schema_ref,
            passed=False,
            errors=["Output is not valid JSON"],
        )

    # Validate against schema
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))

    if not errors:
        return JsonSchemaValidation(
            schema_ref=schema_ref, passed=True, errors=[]
        )

    return JsonSchemaValidation(
        schema_ref=schema_ref,
        passed=False,
        errors=[f"{e.json_path}: {e.message}" for e in errors[:10]],
    )
