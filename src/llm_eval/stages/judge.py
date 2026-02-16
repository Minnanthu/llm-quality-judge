"""Stage 3: LLM-as-a-Judge evaluation."""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

from rich.progress import Progress

from llm_eval.config import EnvConfig, load_run_config
from llm_eval.llm_client import chat_completion, create_client
from llm_eval.models import (
    BlindingInfo,
    InferenceRecord,
    InferenceRef,
    JudgeInfo,
    JudgeTarget,
    JudgementRecord,
    RunConfig,
    Scores,
    Testcase,
)
from llm_eval.prompts import build_absolute_judge_prompt, build_pairwise_judge_prompt
from llm_eval.schema_validation import (
    get_json_schema_ref,
    validate_output_against_testcase_schema,
)
from llm_eval.utils import read_jsonl, write_jsonl


def run_judge(
    config_path: str,
    inference_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Run LLM-as-a-Judge evaluation on inference outputs."""
    EnvConfig()

    cfg = load_run_config(config_path)
    inf_path = inference_path or f"data/inference-{cfg.run_id}.jsonl"
    raw_inferences = read_jsonl(inf_path)
    inferences = [InferenceRecord.model_validate(r) for r in raw_inferences]

    raw_testcases = read_jsonl(cfg.dataset.testcases_path)
    tc_map = {tc["testcase_id"]: Testcase.model_validate(tc) for tc in raw_testcases}

    # Group inferences by testcase_id
    inf_by_tc: dict[str, list[tuple[int, InferenceRecord]]] = {}
    for idx, inf in enumerate(inferences):
        inf_by_tc.setdefault(inf.testcase_id, []).append((idx, inf))

    # For pairwise判定は候補ペアごとに一度に絞り、リピート同士の二乗爆発を防ぐ
    pairwise_inf_by_tc: dict[str, list[tuple[int, InferenceRecord]]] = {}
    for tc_id, inf_list in inf_by_tc.items():
        seen_candidates: set[str] = set()
        unique_list: list[tuple[int, InferenceRecord]] = []
        for idx, inf in inf_list:
            if inf.candidate_id in seen_candidates:
                continue
            seen_candidates.add(inf.candidate_id)
            unique_list.append((idx, inf))
        pairwise_inf_by_tc[tc_id] = unique_list

    out = Path(output_path or f"data/judgements-{cfg.run_id}.jsonl")
    records: list[JudgementRecord] = []

    mode = cfg.protocol.evaluation_mode
    rng = random.Random(cfg.protocol.blinding.random_seed or 42)

    # Calculate total work
    total = _estimate_total(cfg, pairwise_inf_by_tc, inf_by_tc)

    with Progress() as progress:
        task = progress.add_task("Judging", total=total)

        for tc_id, inf_list in inf_by_tc.items():
            tc = tc_map.get(tc_id)
            if not tc:
                continue

            for judge_ref in cfg.judges:
                client = create_client(judge_ref.vendor)

                for repeat in range(cfg.protocol.repeats.judge_repeats):
                    if mode in ("pairwise", "hybrid"):
                        # All pairs
                        pairwise_list = pairwise_inf_by_tc.get(tc_id, [])
                        for (idx_a, inf_a), (idx_b, inf_b) in combinations(pairwise_list, 2):
                            rec = _judge_pairwise(
                                cfg=cfg,
                                tc=tc,
                                inf_a=inf_a,
                                inf_b=inf_b,
                                idx_a=idx_a,
                                idx_b=idx_b,
                                judge_ref=judge_ref,
                                client=client,
                                rng=rng,
                                inf_path=inf_path,
                            )
                            records.append(rec)
                            progress.advance(task)

                    if mode in ("absolute", "hybrid"):
                        for idx, inf in inf_list:
                            rec = _judge_absolute(
                                cfg=cfg,
                                tc=tc,
                                inf=inf,
                                idx=idx,
                                judge_ref=judge_ref,
                                client=client,
                                inf_path=inf_path,
                            )
                            records.append(rec)
                            progress.advance(task)

    write_jsonl(out, records)
    return out


def _estimate_total(
    cfg: RunConfig,
    pairwise_inf_by_tc: dict,
    absolute_inf_by_tc: dict,
) -> int:
    """Estimate total number of judge calls (pairwise uses deduped candidates)."""
    total = 0
    mode = cfg.protocol.evaluation_mode
    n_judges = len(cfg.judges)
    repeats = cfg.protocol.repeats.judge_repeats

    for inf_list in pairwise_inf_by_tc.values():
        n = len(inf_list)
        if mode in ("pairwise", "hybrid"):
            total += n * (n - 1) // 2 * n_judges * repeats

    for inf_list in absolute_inf_by_tc.values():
        n = len(inf_list)
        if mode in ("absolute", "hybrid"):
            total += n * n_judges * repeats

    return total


def _judge_pairwise(
    cfg: RunConfig,
    tc: Testcase,
    inf_a: InferenceRecord,
    inf_b: InferenceRecord,
    idx_a: int,
    idx_b: int,
    judge_ref,
    client,
    rng: random.Random,
    inf_path: str,
) -> JudgementRecord:
    """Run a pairwise judge evaluation."""
    strict_format = _should_use_strict_format_compliance(cfg.protocol.metrics, tc)
    judge_metrics = _filter_llm_metrics(cfg.protocol.metrics, strict_format)

    # Blinding: randomize presentation order
    blinding_enabled = cfg.protocol.blinding.enabled
    if blinding_enabled and rng.random() < 0.5:
        presented = [inf_b.candidate_id, inf_a.candidate_id]
        out_first, out_second = inf_b.output.text, inf_a.output.text
        label_first, label_second = "A", "B"
        cand_map = {"A": inf_b.candidate_id, "B": inf_a.candidate_id}
    else:
        presented = [inf_a.candidate_id, inf_b.candidate_id]
        out_first, out_second = inf_a.output.text, inf_b.output.text
        label_first, label_second = "A", "B"
        cand_map = {"A": inf_a.candidate_id, "B": inf_b.candidate_id}

    messages = build_pairwise_judge_prompt(
        testcase=tc,
        output_a=out_first,
        output_b=out_second,
        label_a=label_first,
        label_b=label_second,
        metrics=judge_metrics,
        rubric_version=judge_ref.rubric_version,
        scoring_scale=cfg.protocol.scoring_scale,
    )

    try:
        response = chat_completion(
            client=client,
            model=judge_ref.model_id,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)

        raw_per_metric = result.get("per_metric", {})
        # Normalize: LLM may return {"A": 5, "B": 3} dicts instead of int
        per_metric: dict[str, int] = {}
        for k, v in raw_per_metric.items():
            if isinstance(v, dict):
                vals = [x for x in v.values() if isinstance(x, (int, float))]
                per_metric[k] = round(sum(vals) / len(vals)) if vals else 0
            else:
                per_metric[k] = int(v)
        raw_winner = result.get("overall_winner", "tie")
        # Map label back to candidate_id
        if raw_winner in cand_map:
            winner = cand_map[raw_winner]
        elif raw_winner == "tie":
            winner = "tie"
        else:
            winner = raw_winner
        rationale = result.get("rationale")
        ci_a = bool(result.get("critical_issue_a", False))
        ci_b = bool(result.get("critical_issue_b", False))
        critical_issue = ci_a or ci_b
        ci_candidates: list[str] = []
        if ci_a:
            ci_candidates.append(cand_map["A"])
        if ci_b:
            ci_candidates.append(cand_map["B"])

    except Exception as e:
        per_metric = {}
        winner = "tie"
        rationale = f"Judge error: {e}"
        critical_issue = False
        ci_candidates = []

    if strict_format:
        strict_score, strict_winner_override = _pairwise_format_compliance_score(
            tc,
            inf_a,
            inf_b,
        )
        per_metric["format_compliance"] = strict_score
        if strict_winner_override is not None:
            winner = strict_winner_override

    return JudgementRecord(
        run_id=cfg.run_id,
        testcase_id=tc.testcase_id,
        judge=JudgeInfo(
            judge_id=judge_ref.judge_id,
            vendor=judge_ref.vendor,
            model_id=judge_ref.model_id,
            rubric_version=judge_ref.rubric_version,
            prompt_version=judge_ref.prompt_version,
        ),
        mode="pairwise",
        targets=[
            JudgeTarget(
                candidate_id=inf_a.candidate_id,
                inference_ref=InferenceRef(path=inf_path, line_index=idx_a),
            ),
            JudgeTarget(
                candidate_id=inf_b.candidate_id,
                inference_ref=InferenceRef(path=inf_path, line_index=idx_b),
            ),
        ],
        blinding=BlindingInfo(
            enabled=blinding_enabled,
            presented_order=presented,
            random_seed=cfg.protocol.blinding.random_seed,
        ),
        scores=Scores(per_metric=per_metric, overall_winner=winner),
        critical_issue=critical_issue,
        critical_issue_candidates=ci_candidates,
        rationale=rationale,
    )


def _judge_absolute(
    cfg: RunConfig,
    tc: Testcase,
    inf: InferenceRecord,
    idx: int,
    judge_ref,
    client,
    inf_path: str,
) -> JudgementRecord:
    """Run an absolute (single-answer) judge evaluation."""
    strict_format = _should_use_strict_format_compliance(cfg.protocol.metrics, tc)
    judge_metrics = _filter_llm_metrics(cfg.protocol.metrics, strict_format)

    messages = build_absolute_judge_prompt(
        testcase=tc,
        output_text=inf.output.text,
        candidate_label=inf.candidate_id,
        metrics=judge_metrics,
        rubric_version=judge_ref.rubric_version,
        scoring_scale=cfg.protocol.scoring_scale,
    )

    try:
        response = chat_completion(
            client=client,
            model=judge_ref.model_id,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)

        per_metric = result.get("per_metric", {})
        overall_score = result.get("overall_score")
        rationale = result.get("rationale")
        critical_issue = bool(result.get("critical_issue", False))

    except Exception as e:
        per_metric = {}
        overall_score = None
        rationale = f"Judge error: {e}"
        critical_issue = False

    if strict_format:
        per_metric["format_compliance"] = _absolute_format_compliance_score(tc, inf.output.text)

    return JudgementRecord(
        run_id=cfg.run_id,
        testcase_id=tc.testcase_id,
        judge=JudgeInfo(
            judge_id=judge_ref.judge_id,
            vendor=judge_ref.vendor,
            model_id=judge_ref.model_id,
            rubric_version=judge_ref.rubric_version,
            prompt_version=judge_ref.prompt_version,
        ),
        mode="absolute",
        targets=[
            JudgeTarget(
                candidate_id=inf.candidate_id,
                inference_ref=InferenceRef(path=inf_path, line_index=idx),
            ),
        ],
        scores=Scores(per_metric=per_metric, overall_score=overall_score),
        critical_issue=critical_issue,
        critical_issue_candidates=[inf.candidate_id] if critical_issue else [],
        rationale=rationale,
    )


def _filter_llm_metrics(metrics: list[str], strict_format_compliance: bool) -> list[str]:
    if not strict_format_compliance:
        return metrics
    return [m for m in metrics if m != "format_compliance"]


def _should_use_strict_format_compliance(metrics: list[str], tc: Testcase) -> bool:
    if "format_compliance" not in metrics:
        return False
    return get_json_schema_ref(tc) is not None


def _absolute_format_compliance_score(tc: Testcase, output_text: str) -> int:
    result = validate_output_against_testcase_schema(tc, output_text)
    if result is None:
        return 1
    return 5 if result.passed else 1


def _pairwise_format_compliance_score(
    tc: Testcase,
    inf_a: InferenceRecord,
    inf_b: InferenceRecord,
) -> tuple[int, str | None]:
    result_a = validate_output_against_testcase_schema(tc, inf_a.output.text)
    result_b = validate_output_against_testcase_schema(tc, inf_b.output.text)

    passed_a = bool(result_a and result_a.passed)
    passed_b = bool(result_b and result_b.passed)

    if passed_a and passed_b:
        return 5, None
    if passed_a and not passed_b:
        return 3, inf_a.candidate_id
    if passed_b and not passed_a:
        return 3, inf_b.candidate_id
    if (not passed_a) and (not passed_b):
        return 1, None

    return 1, None
