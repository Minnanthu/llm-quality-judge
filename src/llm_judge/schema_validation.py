"""Shared JSON schema validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from llm_judge.models import Testcase


class SchemaValidationResult:
    def __init__(self, schema_ref: str, passed: bool, errors: list[str]) -> None:
        self.schema_ref = schema_ref
        self.passed = passed
        self.errors = errors


def get_json_schema_ref(testcase: Testcase | None) -> str | None:
    if not testcase or not testcase.constraints or not testcase.constraints.output_format:
        return None
    output_format = testcase.constraints.output_format
    if output_format.type != "json":
        return None
    return output_format.json_schema_ref


def validate_output_against_testcase_schema(
    testcase: Testcase | None,
    output_text: str,
) -> SchemaValidationResult | None:
    schema_ref = get_json_schema_ref(testcase)
    if not schema_ref:
        return None

    schema_path = Path(schema_ref)
    if not schema_path.exists():
        return SchemaValidationResult(
            schema_ref=schema_ref,
            passed=False,
            errors=[f"Schema file not found: {schema_ref}"],
        )

    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as error:
        return SchemaValidationResult(
            schema_ref=schema_ref,
            passed=False,
            errors=[f"Invalid schema file: {error}"],
        )

    try:
        from llm_judge.utils import strip_fenced_json
        data = json.loads(strip_fenced_json(output_text))
    except json.JSONDecodeError:
        return SchemaValidationResult(
            schema_ref=schema_ref,
            passed=False,
            errors=["Output is not valid JSON"],
        )

    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    if not errors:
        return SchemaValidationResult(schema_ref=schema_ref, passed=True, errors=[])

    return SchemaValidationResult(
        schema_ref=schema_ref,
        passed=False,
        errors=[f"{error.json_path}: {error.message}" for error in errors[:10]],
    )
