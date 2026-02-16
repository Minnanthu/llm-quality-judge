"""JSONL I/O, hashing, and statistics helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any] | BaseModel]) -> None:
    """Write records (dicts or Pydantic models) to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            if isinstance(record, BaseModel):
                f.write(record.model_dump_json(exclude_none=True, by_alias=True) + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any] | BaseModel) -> None:
    """Write a single JSON object to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2, exclude_none=True, by_alias=True)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    with open(path, "w") as f:
        f.write(text + "\n")


def content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def mean(values: list[float | int]) -> float:
    """Compute arithmetic mean. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: list[float | int]) -> float:
    """Compute population variance. Returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


import re

_FENCED_RE = re.compile(r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fenced_json(text: str) -> str:
    """Remove ```json ... ``` fencing if present."""
    stripped = text.strip()
    m = _FENCED_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped
