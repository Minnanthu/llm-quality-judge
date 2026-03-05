"""Stage 4: Aggregation and comparison report generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from llm_judge.artifact_validation import validate_single_artifact
from llm_judge.config import load_run_config
from llm_judge.models import (
    AggregateBlock,
    AutoCheckRecord,
    CandidateInfo,
    ComparisonReport,
    ConsistencyRecord,
    InferenceRecord,
    JudgeAgreement,
    JudgeSummary,
    JudgementRecord,
    NotableFailure,
    ReportDataset,
    ReportSummary,
    Results,
    Testcase,
)
from llm_judge.testcase_loader import load_testcases as _load_testcases
from llm_judge.utils import mean, read_jsonl, write_json, write_jsonl


def run_compare(
    config_path: str,
    judgements_path: str | None = None,
    autocheck_path: str | None = None,
    inference_path: str | None = None,
    output_path: str | None = None,
    consistency_path: str | None = None,
) -> Path:
    """Aggregate judgements and produce comparison report."""
    cfg = load_run_config(config_path)

    jdg_path = judgements_path or f"data/judgements-{cfg.run_id}.jsonl"
    raw_judgements = read_jsonl(jdg_path)
    judgements = [JudgementRecord.model_validate(r) for r in raw_judgements]

    # Load autocheck if available
    ack_path = autocheck_path or f"data/autocheck-{cfg.run_id}.jsonl"
    autochecks: list[AutoCheckRecord] = []
    if Path(ack_path).exists():
        autochecks = [
            AutoCheckRecord.model_validate(r) for r in read_jsonl(ack_path)
        ]

    # Load consistency if available
    con_path = consistency_path or f"data/consistency-{cfg.run_id}.jsonl"
    consistencies: list[ConsistencyRecord] = []
    if Path(con_path).exists():
        consistencies = [
            ConsistencyRecord.model_validate(r) for r in read_jsonl(con_path)
        ]

    # Load testcases for metadata
    testcases = _load_testcases(cfg.dataset.testcases_path)
    tc_map = {tc.testcase_id: tc for tc in testcases}

    # Build report
    candidate_ids = [c.candidate_id for c in cfg.candidates]

    agg_method = cfg.protocol.aggregation.method
    agg_weights = cfg.protocol.aggregation.weights

    overall = _compute_aggregate(
        judgements, candidate_ids, autochecks, consistencies,
        weights=agg_weights, method=agg_method,
    )

    by_task = _compute_by_group(
        judgements, candidate_ids, autochecks, tc_map, consistencies, group_by="task_type",
        weights=agg_weights, method=agg_method,
    )
    by_bucket = _compute_by_group(
        judgements, candidate_ids, autochecks, tc_map, consistencies, group_by="bucket",
        weights=agg_weights, method=agg_method,
    )

    agreement = _compute_judge_agreement(judgements)
    summary = _compute_summary(judgements)

    report = ComparisonReport(
        run_id=cfg.run_id,
        dataset=ReportDataset(
            dataset_version=cfg.dataset.dataset_version,
            testcase_count=len(testcases),
        ),
        candidates=[
            CandidateInfo(
                candidate_id=c.candidate_id,
                vendor=c.vendor,
                model_id=c.model_id,
            )
            for c in cfg.candidates
        ],
        judges=[
            JudgeSummary(
                judge_id=j.judge_id,
                vendor=j.vendor,
                model_id=j.model_id,
                rubric_version=j.rubric_version,
            )
            for j in cfg.judges
        ],
        protocol=cfg.protocol.model_dump(),
        summary=summary,
        results=Results(
            overall=overall,
            by_task=by_task,
            by_bucket=by_bucket,
            judge_agreement=agreement,
        ),
    )

    json_out = Path(output_path or f"data/comparison-report-{cfg.run_id}.json")
    validate_single_artifact("comparison-report", report)
    write_json(json_out, report)

    # Also write markdown summary
    md_out = json_out.with_suffix(".md")
    _write_markdown_report(report, md_out)

    return json_out


def _mode_value(values: list[float | int]) -> float:
    """Return the mode (most frequent value). Tie → min (conservative)."""
    from collections import Counter

    if not values:
        return 0.0
    counts = Counter(values)
    max_count = max(counts.values())
    modes = [v for v, c in counts.items() if c == max_count]
    return float(min(modes))


def _aggregate_scores(
    method: str,
    metric_scores: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    """Dispatch metric-score aggregation by method name."""
    if method == "mean":
        return {
            m: {cid: round(mean(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    if method == "worst_case":
        return {
            m: {cid: round(min(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    if method == "majority_vote":
        # After _reduce_absolute_scores_by_majority, each value is already
        # a per-group mode.  Final aggregation is mean over those representatives.
        return {
            m: {cid: round(mean(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    if method == "custom":
        # custom uses mean per metric; weighting is applied in weighted_overall
        return {
            m: {cid: round(mean(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    raise ValueError(f"Unknown aggregation method: '{method}'")


def _compute_weighted_overall(
    method: str,
    weights: dict[str, float] | None,
    metric_scores: dict[str, dict[str, list[float]]],
    candidate_ids: list[str],
) -> dict[str, float]:
    """Compute weighted overall score per candidate."""
    if method == "custom" and not weights:
        raise ValueError(
            "Aggregation method 'custom' requires non-empty weights"
        )

    if not weights or not metric_scores:
        return {}

    weighted_overall: dict[str, float] = {}
    for cid in candidate_ids:
        w_sum = 0.0
        score_sum = 0.0
        for metric_id, w in weights.items():
            if metric_id in metric_scores and cid in metric_scores[metric_id]:
                score_sum += mean(metric_scores[metric_id][cid]) * w
                w_sum += w
        if w_sum > 0:
            weighted_overall[cid] = round(score_sum / w_sum, 2)
    return weighted_overall


def _count_pairwise_majority(
    judgements: list[JudgementRecord],
) -> dict[str, int]:
    """Count majority-vote winners across (testcase, pair, judge) groups.

    Returns a dict of winner_id -> count, including "tie".
    """
    # Group by (testcase_id, target_pair, judge_id)
    groups: dict[str, list[str]] = defaultdict(list)
    for jdg in judgements:
        if jdg.mode != "pairwise" or not jdg.scores.overall_winner:
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{target_key}:{jdg.judge.judge_id}"
        groups[key].append(jdg.scores.overall_winner)

    from collections import Counter

    counts: dict[str, int] = defaultdict(int)
    for winners in groups.values():
        c = Counter(winners)
        max_count = max(c.values())
        modes = [w for w, cnt in c.items() if cnt == max_count]
        majority_winner = "tie" if len(modes) > 1 else modes[0]
        counts[majority_winner] += 1

    return dict(counts)


def _aggregate_majority_vote_pairwise(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute win/loss rates using majority vote per (testcase, pair, judge) group."""
    counts = _count_pairwise_majority(judgements)
    total = sum(counts.values())
    if total == 0:
        return {}, {}

    # Collect all target candidates per group to track losses
    groups: dict[str, list[str]] = defaultdict(list)
    group_targets: dict[str, set[str]] = {}
    for jdg in judgements:
        if jdg.mode != "pairwise" or not jdg.scores.overall_winner:
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{target_key}:{jdg.judge.judge_id}"
        groups[key].append(jdg.scores.overall_winner)
        if key not in group_targets:
            group_targets[key] = {t.candidate_id for t in jdg.targets}

    from collections import Counter

    win_counts: dict[str, int] = defaultdict(int)
    loss_counts: dict[str, int] = defaultdict(int)
    for key, winners in groups.items():
        c = Counter(winners)
        max_count = max(c.values())
        modes = [w for w, cnt in c.items() if cnt == max_count]
        majority_winner = "tie" if len(modes) > 1 else modes[0]
        if majority_winner != "tie":
            win_counts[majority_winner] += 1
            for cid in group_targets[key]:
                if cid != majority_winner:
                    loss_counts[cid] += 1

    win_rate: dict[str, float] = {}
    loss_rate: dict[str, float] = {}
    for cid in candidate_ids:
        win_rate[cid] = round(win_counts[cid] / total, 4)
        loss_rate[cid] = round(loss_counts[cid] / total, 4)

    return win_rate, loss_rate


def _reduce_absolute_scores_by_majority(
    judgements: list[JudgementRecord],
) -> dict[str, dict[str, list[float]]]:
    """Reduce absolute scores by majority vote within each (testcase, candidate, judge, metric) group.

    For each group of repeat scores, the mode is computed (tie → min).
    Returns ``metric_scores`` in the same shape as the raw collector:
    ``{metric_id: {candidate_id: [reduced_value_per_group, ...]}}``.
    """
    # Collect raw scores per (testcase_id, candidate_id, judge_id, metric_id)
    raw: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for jdg in judgements:
        if jdg.mode == "pairwise":
            continue
        for t in jdg.targets:
            cid = t.candidate_id
            for metric_id, score in jdg.scores.per_metric.items():
                key = (jdg.testcase_id, cid, jdg.judge.judge_id, metric_id)
                raw[key].append(score)

    # Reduce each group to its mode, then reorganise by metric → candidate
    reduced: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (tc_id, cid, judge_id, metric_id), scores in raw.items():
        representative = _mode_value(scores)
        reduced[metric_id][cid].append(representative)

    return reduced


def _compute_aggregate(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
    autochecks: list[AutoCheckRecord],
    consistencies: list[ConsistencyRecord] | None = None,
    weights: dict[str, float] | None = None,
    method: str = "mean",
) -> AggregateBlock:
    """Compute aggregate stats across all judgements."""
    import math

    from llm_judge.utils import variance as _variance

    # Win rate (pairwise only)
    win_counts: dict[str, int] = defaultdict(int)
    loss_counts: dict[str, int] = defaultdict(int)
    tie_count = 0
    pairwise_total = 0

    for jdg in judgements:
        if jdg.mode == "pairwise" and jdg.scores.overall_winner:
            pairwise_total += 1
            winner = jdg.scores.overall_winner
            if winner == "tie":
                tie_count += 1
            else:
                win_counts[winner] += 1
                # Count losses for all other candidates in this judgement
                for t in jdg.targets:
                    if t.candidate_id != winner:
                        loss_counts[t.candidate_id] += 1

    win_rate: dict[str, float] = {}
    loss_rate: dict[str, float] = {}
    if pairwise_total > 0:
        for cid in candidate_ids:
            win_rate[cid] = round(win_counts[cid] / pairwise_total, 4)
            loss_rate[cid] = round(loss_counts[cid] / pairwise_total, 4)
        win_rate["tie"] = round(tie_count / pairwise_total, 4)

    # Mean score per metric per candidate (absolute judgements only)
    # Pairwise per_metric scores represent a comparative evaluation,
    # not individual candidate scores, so they are excluded here.
    metric_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for jdg in judgements:
        if jdg.mode == "pairwise":
            continue  # skip pairwise – scores are not per-candidate
        for t in jdg.targets:
            cid = t.candidate_id
            for metric_id, score in jdg.scores.per_metric.items():
                metric_scores[metric_id][cid].append(score)

    # For majority_vote, reduce scores per (testcase, candidate, judge, metric)
    # group first, then use the reduced data for all downstream computations.
    if method == "majority_vote":
        metric_scores = _reduce_absolute_scores_by_majority(judgements)

    # Aggregate metric scores according to method
    mean_score = _aggregate_scores(method, metric_scores)

    # Compute weighted overall score per candidate
    weighted_overall = _compute_weighted_overall(
        method, weights, metric_scores, candidate_ids,
    )

    # Pairwise win_rate: for majority_vote, re-derive from per-testcase majority
    if method == "majority_vote" and pairwise_total > 0:
        win_rate, loss_rate = _aggregate_majority_vote_pairwise(
            judgements, candidate_ids,
        )
        pairwise_total_adjusted = sum(
            v for k, v in _count_pairwise_majority(judgements).items()
        )
        if pairwise_total_adjusted > 0:
            tie_count_mv = _count_pairwise_majority(judgements).get("tie", 0)
            win_rate["tie"] = round(
                tie_count_mv / pairwise_total_adjusted, 4
            )

    # Confidence intervals (95% CI via standard error) per metric per candidate
    confidence_intervals: dict[str, dict[str, dict[str, float]]] = {}
    for m, by_cid in metric_scores.items():
        ci_by_cid: dict[str, dict[str, float]] = {}
        for cid, scores in by_cid.items():
            n = len(scores)
            if n >= 2:
                avg = mean(scores)
                var = _variance(scores)
                se = math.sqrt(var / n)
                ci_by_cid[cid] = {
                    "mean": round(avg, 4),
                    "lower": round(avg - 1.96 * se, 4),
                    "upper": round(avg + 1.96 * se, 4),
                    "n": n,
                }
        if ci_by_cid:
            confidence_intervals[m] = ci_by_cid

    # Critical issue count per candidate (using per-candidate tracking)
    ci_counts: dict[str, int] = defaultdict(int)
    for jdg in judgements:
        for cid in jdg.critical_issue_candidates:
            ci_counts[cid] += 1
    critical_issue_count = dict(ci_counts) if ci_counts else {}

    # Notable failures from autochecks
    failures: list[NotableFailure] = []
    for ac in autochecks:
        if ac.checks.format_compliance and not ac.checks.format_compliance.passed:
            failures.append(
                NotableFailure(
                    testcase_id=ac.testcase_id,
                    candidate_id=ac.candidate_id,
                    reason=f"Format check failed: {ac.checks.format_compliance.details}",
                )
            )
        if (
            ac.checks.json_schema_validation
            and not ac.checks.json_schema_validation.passed
        ):
            failures.append(
                NotableFailure(
                    testcase_id=ac.testcase_id,
                    candidate_id=ac.candidate_id,
                    reason=f"Schema validation failed: {'; '.join(ac.checks.json_schema_validation.errors[:3])}",
                )
            )

    # Inference consistency (from consistency stage)
    inference_consistency: dict[str, float] = {}
    if consistencies:
        con_scores: dict[str, list[float]] = defaultdict(list)
        for rec in consistencies:
            if rec.status.ok and rec.scores.overall is not None:
                con_scores[rec.candidate_id].append(rec.scores.overall)
        for cid, scores in con_scores.items():
            if scores:
                inference_consistency[cid] = round(mean(scores), 2)

    return AggregateBlock(
        win_rate=win_rate,
        loss_rate=loss_rate,
        mean_score=mean_score,
        weighted_overall=weighted_overall,
        confidence_intervals=confidence_intervals,
        critical_issue_count=critical_issue_count,
        notable_failures=failures,
        inference_consistency=inference_consistency,
    )


def _compute_by_group(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
    autochecks: list[AutoCheckRecord],
    tc_map: dict[str, Testcase],
    consistencies: list[ConsistencyRecord] | None = None,
    group_by: str = "task_type",
    weights: dict[str, float] | None = None,
    method: str = "mean",
) -> dict[str, AggregateBlock]:
    """Compute aggregate stats grouped by task_type or bucket."""
    groups: dict[str, list[JudgementRecord]] = defaultdict(list)
    ac_groups: dict[str, list[AutoCheckRecord]] = defaultdict(list)
    con_groups: dict[str, list[ConsistencyRecord]] = defaultdict(list)

    for jdg in judgements:
        tc = tc_map.get(jdg.testcase_id)
        if not tc:
            continue
        if group_by == "task_type":
            key = tc.task_type
        elif group_by == "bucket":
            key = (tc.metadata.input_length_bucket if tc.metadata else None) or "unknown"
        else:
            key = "all"
        groups[key].append(jdg)

    for ac in autochecks:
        tc = tc_map.get(ac.testcase_id)
        if not tc:
            continue
        if group_by == "task_type":
            key = tc.task_type
        elif group_by == "bucket":
            key = (tc.metadata.input_length_bucket if tc.metadata else None) or "unknown"
        else:
            key = "all"
        ac_groups[key].append(ac)

    for con in (consistencies or []):
        tc = tc_map.get(con.testcase_id)
        if not tc:
            continue
        if group_by == "task_type":
            key = tc.task_type
        elif group_by == "bucket":
            key = (tc.metadata.input_length_bucket if tc.metadata else None) or "unknown"
        else:
            key = "all"
        con_groups[key].append(con)

    result = {}
    for key, jdgs in groups.items():
        result[key] = _compute_aggregate(
            jdgs, candidate_ids, ac_groups.get(key, []), con_groups.get(key, []),
            weights=weights, method=method,
        )

    return result


def _compute_judge_agreement(judgements: list[JudgementRecord]) -> JudgeAgreement:
    """Compute pairwise agreement rate across judges."""
    # Group pairwise judgements by (testcase_id, target pair)
    groups: dict[str, list[str]] = defaultdict(list)

    for jdg in judgements:
        if jdg.mode != "pairwise":
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{target_key}"
        groups[key].append(jdg.scores.overall_winner or "tie")

    if not groups:
        return JudgeAgreement(notes="No pairwise judgements found")

    agreements = 0
    total_pairs = 0

    for winners in groups.values():
        if len(winners) < 2:
            continue
        # Count agreement among all pairs of judge decisions
        for i in range(len(winners)):
            for j in range(i + 1, len(winners)):
                total_pairs += 1
                if winners[i] == winners[j]:
                    agreements += 1

    if total_pairs == 0:
        return JudgeAgreement(notes="Insufficient data for agreement calculation")

    rate = round(agreements / total_pairs, 4)
    return JudgeAgreement(
        pairwise_agreement_rate=rate,
        notes=f"Based on {total_pairs} judge pairs across {len(groups)} cases",
    )


def _compute_summary(judgements: list[JudgementRecord]) -> ReportSummary:
    """Compute overall summary statistics."""
    total = len(judgements)
    valid = sum(1 for j in judgements if j.scores.per_metric)
    excluded = total - valid

    stability = _compute_repeat_stability(judgements)

    return ReportSummary(
        total_judgements=total,
        valid_judgements=valid,
        excluded_judgements=excluded,
        repeat_stability=stability,
    )


def _compute_repeat_stability(judgements: list[JudgementRecord]) -> float | None:
    """Compute repeat stability: agreement rate within same (testcase, judge, target pair).

    Groups pairwise judgements by (testcase_id, judge_id, target_pair) and measures
    how often repeated judge calls agree on the winner.
    """
    groups: dict[str, list[str]] = defaultdict(list)

    for jdg in judgements:
        if jdg.mode != "pairwise":
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{jdg.judge.judge_id}:{target_key}"
        groups[key].append(jdg.scores.overall_winner or "tie")

    # Only consider groups with repeats
    repeat_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not repeat_groups:
        return None

    agreements = 0
    total_pairs = 0
    for winners in repeat_groups.values():
        for i in range(len(winners)):
            for j in range(i + 1, len(winners)):
                total_pairs += 1
                if winners[i] == winners[j]:
                    agreements += 1

    if total_pairs == 0:
        return None

    return round(agreements / total_pairs, 4)


def _write_markdown_report(report: ComparisonReport, path: Path) -> None:
    """Write a human-readable markdown report."""
    lines: list[str] = []
    lines.append(f"# Comparison Report: {report.run_id}")
    lines.append("")
    lines.append(f"- Dataset version: {report.dataset.dataset_version}")
    lines.append(f"- Testcase count: {report.dataset.testcase_count}")
    lines.append(
        f"- Candidates: {', '.join(c.candidate_id for c in report.candidates)}"
    )
    lines.append(f"- Judges: {', '.join(j.judge_id for j in report.judges)}")
    lines.append("")

    # Summary
    s = report.summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total judgements: {s.total_judgements}")
    lines.append(f"- Valid judgements: {s.valid_judgements}")
    lines.append(f"- Excluded judgements: {s.excluded_judgements}")
    if s.repeat_stability is not None:
        lines.append(f"- Repeat stability: {s.repeat_stability:.1%}")
    lines.append("")

    # Overall results
    lines.append("## Overall Results")
    lines.append("")

    if report.results.overall.win_rate:
        lines.append("> [!NOTE]")
        lines.append("> このレポートは **Win Rate (直接比較)** と **Mean Scores (個別採点)** という2つの異なる評価方式に基づいています。")
        lines.append("> 評価視点が異なるため、特定の指標スコアが高くても Win Rate で負ける（あるいはその逆）といった結果が生じ得ます。")
        lines.append("")

        lines.append("### Win Rate / Loss Rate")
        lines.append("同じ評価指標 (Rubric) を基準としつつ、各候補の回答を並べて「どちらがより優れているか」を相対的に比較 (Pairwise判定) した結果です。")
        lines.append("テストケース数が少ない場合、1件の判定が全体に大きく影響するため、トレンドを把握するための参考値として参照してください。")
        lines.append("")
        lines.append("| Candidate | Win | Loss | Tie |")
        lines.append("|-----------|-----|------|-----|")
        tie_rate_val = report.results.overall.win_rate.get("tie", 0.0)
        for cid in [c.candidate_id for c in report.candidates]:
            w = report.results.overall.win_rate.get(cid, 0.0)
            l = report.results.overall.loss_rate.get(cid, 0.0)
            lines.append(f"| {cid} | {w:.1%} | {l:.1%} | {tie_rate_val:.1%} |")
        lines.append("")

    if report.results.overall.mean_score:
        scale = report.protocol.get("scoring_scale", [1, 3, 5])
        scale_str = "/".join(str(s) for s in scale)
        cids = [c.candidate_id for c in report.candidates]
        lines.append("### Mean Scores by Metric")
        lines.append(f"各候補の回答を個別に採点し、全テストケースにわたって平均したスコア ({scale_str} 段階評価) です。")
        lines.append("特定の強みや弱みを分析するのに適しています。")
        lines.append("")
        header = "| Metric | " + " | ".join(cids) + " |"
        sep = "|--------" + "|------" * len(cids) + "|"
        lines.append(header)
        lines.append(sep)
        for metric, by_cid in sorted(report.results.overall.mean_score.items()):
            vals = " | ".join(f"{by_cid.get(c, 0.0):.2f}" for c in cids)
            lines.append(f"| {metric} | {vals} |")
        lines.append("")

    if report.results.overall.confidence_intervals:
        cids = [c.candidate_id for c in report.candidates]
        lines.append("### Confidence Intervals (95%)")
        lines.append("各指標の 95% 信頼区間（平均 ± 1.96×SE）。")
        lines.append("")
        header = "| Metric | " + " | ".join(cids) + " |"
        sep = "|--------" + "|------" * len(cids) + "|"
        lines.append(header)
        lines.append(sep)
        for metric, by_cid in sorted(report.results.overall.confidence_intervals.items()):
            vals = []
            for c in cids:
                ci = by_cid.get(c)
                if ci:
                    vals.append(f"{ci['mean']:.2f} [{ci['lower']:.2f}, {ci['upper']:.2f}]")
                else:
                    vals.append("—")
            lines.append(f"| {metric} | {' | '.join(vals)} |")
        lines.append("")

    if report.results.overall.weighted_overall:
        cids = [c.candidate_id for c in report.candidates]
        weights = report.protocol.get("aggregation", {}).get("weights", {})
        weights_str = ", ".join(f"{k}: {v}" for k, v in weights.items())
        lines.append("### Weighted Overall Score")
        lines.append(f"設定された重み ({weights_str}) に基づく加重平均スコアです。")
        lines.append("")
        for cid in cids:
            score = report.results.overall.weighted_overall.get(cid, 0.0)
            lines.append(f"- {cid}: {score:.2f}")
        lines.append("")
    # By task type
    if report.results.by_task:
        lines.append("## Results by Task Type")
        desc = "タスク種別ごとの内訳。"
        if report.results.overall.win_rate:
            desc += "Win Rate は Pairwise 比較の勝率。"
        lines.append(desc)
        lines.append("")
        for task_type, agg in report.results.by_task.items():
            lines.append(f"### {task_type}")
            if agg.win_rate:
                for cid, rate in agg.win_rate.items():
                    lines.append(f"- {cid}: {rate:.1%}")
            if agg.mean_score:
                if agg.win_rate:
                    lines.append("")
                cids = [c.candidate_id for c in report.candidates]
                header = "| Metric | " + " | ".join(cids) + " |"
                sep = "|--------" + "|------" * len(cids) + "|"
                lines.append(header)
                lines.append(sep)
                for metric, by_cid in sorted(agg.mean_score.items()):
                    vals = " | ".join(f"{by_cid.get(c, 0.0):.2f}" for c in cids)
                    lines.append(f"| {metric} | {vals} |")
            lines.append("")

    # By bucket
    if report.results.by_bucket:
        lines.append("## Results by Input Length Bucket")
        lines.append("入力長カテゴリ (S/M/L) ごとの内訳。")
        lines.append("")
        for bucket, agg in report.results.by_bucket.items():
            lines.append(f"### Bucket: {bucket}")
            if agg.win_rate:
                for cid, rate in agg.win_rate.items():
                    lines.append(f"- {cid}: {rate:.1%}")
            if agg.mean_score:
                if agg.win_rate:
                    lines.append("")
                cids = [c.candidate_id for c in report.candidates]
                header = "| Metric | " + " | ".join(cids) + " |"
                sep = "|--------" + "|------" * len(cids) + "|"
                lines.append(header)
                lines.append(sep)
                for metric, by_cid in sorted(agg.mean_score.items()):
                    vals = " | ".join(f"{by_cid.get(c, 0.0):.2f}" for c in cids)
                    lines.append(f"| {metric} | {vals} |")
            lines.append("")

    # Judge agreement
    if report.results.judge_agreement.pairwise_agreement_rate is not None:
        lines.append("## Judge Agreement")
        lines.append(
            f"- Pairwise agreement rate: {report.results.judge_agreement.pairwise_agreement_rate:.1%}"
        )
        if report.results.judge_agreement.notes:
            lines.append(f"- {report.results.judge_agreement.notes}")
        lines.append("")

    # Inference consistency
    if report.results.overall.inference_consistency:
        cids = [c.candidate_id for c in report.candidates]
        lines.append("## Inference Consistency")
        lines.append(
            "同一プロンプトへの繰り返し推論（inference_repeats）の出力間の一貫性を"
            "LLM-as-a-Judgeで評価したスコア（1〜5）。"
            "5 = 非常に一貫、1 = 不一貫。"
        )
        lines.append("")
        lines.append("| Candidate | Consistency Score (1-5) |")
        lines.append("|-----------|------------------------|")
        for cid in cids:
            score = report.results.overall.inference_consistency.get(cid)
            score_str = f"{score:.2f}" if score is not None else "—"
            lines.append(f"| {cid} | {score_str} |")
        lines.append("")

    # Critical issues
    if report.results.overall.critical_issue_count:
        lines.append("## Critical Issues")
        lines.append("")
        lines.append("| Candidate | Count |")
        lines.append("|-----------|-------|")
        for cid, count in sorted(report.results.overall.critical_issue_count.items()):
            lines.append(f"| {cid} | {count} |")
        lines.append("")

        # By task breakdown
        for task_type, agg in report.results.by_task.items():
            if agg.critical_issue_count:
                lines.append(f"### {task_type}")
                for cid, count in sorted(agg.critical_issue_count.items()):
                    lines.append(f"- {cid}: {count}")
                lines.append("")

    # Notable failures
    failures = report.results.overall.notable_failures
    if failures:
        lines.append("## Notable Failures")
        lines.append("Autocheck で検出されたフォーマット・スキーマ違反。")
        lines.append("")
        for f in failures:
            lines.append(f"- **{f.testcase_id}** ({f.candidate_id}): {f.reason}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
