"""RunConfig loader and environment variable resolution."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

from llm_eval.models import RunConfig


class EnvConfig(BaseSettings):
    """API keys and endpoints resolved from environment / .env file."""

    model_config = {"env_file": ".env", "extra": "ignore"}

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    tsuzumi2_api_key: str = ""
    tsuzumi2_endpoint: str = ""


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run-config YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
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
