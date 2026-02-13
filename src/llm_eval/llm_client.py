"""OpenAI SDK wrapper with retry logic for multiple vendors."""

from __future__ import annotations

from typing import Any

from openai import AzureOpenAI, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_eval.config import resolve_vendor_env


def create_client(vendor: str, endpoint: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client based on vendor name."""
    api_key, env_endpoint = resolve_vendor_env(vendor)
    endpoint = endpoint or env_endpoint

    if vendor == "azure-openai":
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-12-01-preview",
        )
    else:
        # Generic OpenAI-compatible endpoint (tsuzumi2 etc.)
        return OpenAI(
            api_key=api_key,
            base_url=endpoint or None,
        )


@retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> Any:
    """Call chat completions with automatic retry on transient errors."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
