"""Stage 1: Candidate model inference."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import Progress

from llm_eval.config import EnvConfig, load_run_config
from llm_eval.llm_client import chat_completion, create_client
from llm_eval.models import (
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
from llm_eval.prompts import build_inference_prompt
from llm_eval.utils import content_hash, read_jsonl, write_jsonl


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
            input_hash_val = content_hash(str(messages))

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
                        input_hash_val=input_hash_val,
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
    input_hash_val: str,
) -> InferenceRecord:
    """Call a single model and return an InferenceRecord."""
    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    try:
        response = chat_completion(
            client=client,
            model=candidate.model_id,
            messages=messages,
            **gen_params,
        )
        t1 = time.monotonic()

        text = response.choices[0].message.content or ""
        usage = response.usage

        # Detect format from constraints
        fmt = None
        if tc.constraints and tc.constraints.output_format:
            fmt = tc.constraints.output_format.type

        output = OutputInfo(text=text, format=fmt)

        # Try to parse JSON if expected
        if fmt == "json":
            import json as _json
            from llm_eval.utils import strip_fenced_json
            try:
                output.json_data = _json.loads(strip_fenced_json(text))
            except _json.JSONDecodeError:
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
            generation_params=gen_params or None,
            input_hash=input_hash_val,
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
            input_hash=input_hash_val,
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
