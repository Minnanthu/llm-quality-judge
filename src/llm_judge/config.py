"""RunConfig loader and environment variable resolution."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from llm_judge.models import RunConfig

# Load .env into os.environ so resolve_vendor_env() can access all keys
load_dotenv()


class EnvConfig(BaseSettings):
    """API keys and endpoints resolved from environment / .env file."""

    model_config = {"env_file": ".env", "extra": "ignore"}

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    tsuzumi2_api_key: str = ""
    tsuzumi2_endpoint: str = ""


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run-config YAML file.

    Validates against the JSON Schema first, then against the Pydantic model.
    """
    import json

    import jsonschema

    with open(path) as f:
        raw = yaml.safe_load(f)

    # JSON Schema validation (SKILL.md requirement)
    schema_path = (
        Path(__file__).resolve().parents[2]
        / ".claude"
        / "skills"
        / "evaluating-llm-quality"
        / "schemas"
        / "run-config.schema.json"
    )
    if schema_path.exists():
        with open(schema_path) as sf:
            schema = json.load(sf)
        jsonschema.validate(instance=raw, schema=schema)

    return RunConfig.model_validate(raw)


def resolve_vendor_env(vendor: str) -> tuple[str, str]:
    """Return (api_key, endpoint) for a vendor by environment variable convention.

    Convention: VENDOR_API_KEY, VENDOR_ENDPOINT where vendor is upper-cased
    and hyphens are replaced with underscores.
    """
    prefix = vendor.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    endpoint = os.environ.get(f"{prefix}_ENDPOINT", "")
    return api_key, endpoint
