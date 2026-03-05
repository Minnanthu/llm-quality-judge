"""Tests for candidate count validation by evaluation_mode."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_judge.config import load_run_config
from llm_judge.models import RunConfig


def _candidate(cid: str) -> dict:
    return {
        "candidate_id": cid,
        "vendor": "openai",
        "model_id": "gpt-4o",
    }


def _base_config(mode: str, num_candidates: int) -> dict:
    candidates = [_candidate(f"c{i}") for i in range(num_candidates)]
    return {
        "run_id": "test-run",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": candidates,
        "judges": [
            {
                "judge_id": "j1",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "rubric_version": "v1",
            }
        ],
        "protocol": {
            "evaluation_mode": mode,
            "scoring_scale": [1, 3, 5],
            "aggregation": {"method": "mean"},
        },
    }


# ── Pydantic (model_validate) tests ──


class TestRunConfigPydanticValidation:
    def test_absolute_1_candidate_ok(self):
        cfg = _base_config("absolute", 1)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 1

    def test_absolute_2_candidates_ok(self):
        cfg = _base_config("absolute", 2)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 2

    def test_absolute_0_candidates_fails(self):
        cfg = _base_config("absolute", 0)
        with pytest.raises(Exception):
            RunConfig.model_validate(cfg)

    def test_pairwise_1_candidate_fails(self):
        cfg = _base_config("pairwise", 1)
        with pytest.raises(Exception, match="at least 2 candidates"):
            RunConfig.model_validate(cfg)

    def test_pairwise_2_candidates_ok(self):
        cfg = _base_config("pairwise", 2)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 2

    def test_hybrid_1_candidate_fails(self):
        cfg = _base_config("hybrid", 1)
        with pytest.raises(Exception, match="at least 2 candidates"):
            RunConfig.model_validate(cfg)

    def test_hybrid_2_candidates_ok(self):
        cfg = _base_config("hybrid", 2)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 2


# ── load_run_config (YAML + JSON Schema + Pydantic) tests ──


class TestLoadRunConfigValidation:
    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "run-config.yaml"
        p.write_text(yaml.dump(data, allow_unicode=True))
        return p

    def test_absolute_1_candidate_loads(self, tmp_path):
        cfg = _base_config("absolute", 1)
        path = self._write_yaml(tmp_path, cfg)
        rc = load_run_config(path)
        assert len(rc.candidates) == 1

    def test_pairwise_1_candidate_rejected(self, tmp_path):
        cfg = _base_config("pairwise", 1)
        path = self._write_yaml(tmp_path, cfg)
        with pytest.raises(Exception):
            load_run_config(path)

    def test_hybrid_1_candidate_rejected(self, tmp_path):
        cfg = _base_config("hybrid", 1)
        path = self._write_yaml(tmp_path, cfg)
        with pytest.raises(Exception):
            load_run_config(path)

    def test_pairwise_2_candidates_loads(self, tmp_path):
        cfg = _base_config("pairwise", 2)
        path = self._write_yaml(tmp_path, cfg)
        rc = load_run_config(path)
        assert len(rc.candidates) == 2
