"""Stage 3b: Inference consistency evaluation (LLM-as-a-Judge).

For each (testcase_id, candidate_id) group that has >= 2 inference records,
a Judge model is asked to score how consistently the candidate answered the
same prompt across all repeated outputs (1 = incoherent/contradicting,
5 = perfectly consistent).

This stage is only meaningful when inference_repeats >= 2.
Results are written to data/consistency-<run_id>.jsonl.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.progress import Progress

from llm_judge.config import EnvConfig, load_run_config
from llm_judge.llm_client import chat_completion, create_client
from llm_judge.models import (
    ConsistencyRecord,
    ConsistencyScores,
    InferenceRecord,
    StatusInfo,
    Testcase,
)
from llm_judge.prompts import build_consistency_judge_prompt
from llm_judge.testcase_loader import load_testcase_map
from llm_judge.utils import read_jsonl, write_jsonl


def run_consistency(
    config_path: str,
    inference_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Evaluate consistency of repeated inference outputs via LLM-as-a-Judge."""
    EnvConfig()

    cfg = load_run_config(config_path)
    inf_path = inference_path or f"data/inference-{cfg.run_id}.jsonl"
    raw_inferences = read_jsonl(inf_path)
    inferences = [InferenceRecord.model_validate(r) for r in raw_inferences]

    tc_map = load_testcase_map(cfg.dataset.testcases_path)

    # Group inferences by (testcase_id, candidate_id)
    groups: dict[tuple[str, str], list[InferenceRecord]] = defaultdict(list)
    for inf in inferences:
        if inf.status.ok and inf.output.text:
            groups[(inf.testcase_id, inf.candidate_id)].append(inf)

    # Only evaluate groups with >= 2 repeats
    eligible = {k: v for k, v in groups.items() if len(v) >= 2}

    if not eligible:
        import warnings
        warnings.warn(
            "No groups with inference_repeats >= 2 found. "
            "Set inference_repeats >= 2 in run config to enable consistency evaluation.",
            RuntimeWarning,
        )

    out = Path(output_path or f"data/consistency-{cfg.run_id}.jsonl")
    records: list[ConsistencyRecord] = []

    total = len(eligible) * len(cfg.judges)

    with Progress() as progress:
        task_bar = progress.add_task("Consistency", total=total)

        for (tc_id, candidate_id), inf_list in eligible.items():
            tc = tc_map.get(tc_id)
            if not tc:
                progress.advance(task_bar, len(cfg.judges))
                continue

            outputs = [inf.output.text for inf in inf_list]

            for judge_ref in cfg.judges:
                client = create_client(judge_ref.vendor)
                messages = build_consistency_judge_prompt(
                    testcase=tc,
                    outputs=outputs,
                    rubric_version=judge_ref.rubric_version,
                )
                rec = _call_consistency_judge(
                    cfg_run_id=cfg.run_id,
                    tc_id=tc_id,
                    candidate_id=candidate_id,
                    judge_id=judge_ref.judge_id,
                    judge_model=judge_ref.model_id,
                    messages=messages,
                    client=client,
                    repeat_count=len(outputs),
                )
                records.append(rec)
                progress.advance(task_bar)

    write_jsonl(out, records)
    return out


def _call_consistency_judge(
    cfg_run_id: str,
    tc_id: str,
    candidate_id: str,
    judge_id: str,
    judge_model: str,
    messages: list[dict[str, str]],
    client,
    repeat_count: int,
) -> ConsistencyRecord:
    try:
        response = chat_completion(
            client=client,
            model=judge_model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)

        overall = result.get("overall")
        if overall is not None:
            overall = float(overall)
        rationale = result.get("rationale")

        return ConsistencyRecord(
            run_id=cfg_run_id,
            testcase_id=tc_id,
            candidate_id=candidate_id,
            judge_id=judge_id,
            repeat_count=repeat_count,
            scores=ConsistencyScores(overall=overall, rationale=rationale),
            status=StatusInfo(ok=True),
        )

    except Exception as e:
        return ConsistencyRecord(
            run_id=cfg_run_id,
            testcase_id=tc_id,
            candidate_id=candidate_id,
            judge_id=judge_id,
            repeat_count=repeat_count,
            scores=ConsistencyScores(),
            status=StatusInfo(
                ok=False,
                error_type=type(e).__name__,
                error_message=str(e),
            ),
        )
