"""Stage 4: Aggregation and comparison report generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from llm_eval.config import load_run_config
from llm_eval.models import (
    AggregateBlock,
    AutoCheckRecord,
    CandidateInfo,
    ComparisonReport,
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
from llm_eval.utils import mean, read_jsonl, variance, write_json, write_jsonl


def run_compare(
    config_path: str,
    judgements_path: str | None = None,
    autocheck_path: str | None = None,
    inference_path: str | None = None,
    output_path: str | None = None,
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

    # Load testcases for metadata
    raw_testcases = read_jsonl(cfg.dataset.testcases_path)
    testcases = [Testcase.model_validate(tc) for tc in raw_testcases]
    tc_map = {tc.testcase_id: tc for tc in testcases}

    # Build report
    candidate_ids = [c.candidate_id for c in cfg.candidates]

    overall = _compute_aggregate(judgements, candidate_ids, autochecks)

    by_task = _compute_by_group(
        judgements, candidate_ids, autochecks, tc_map, group_by="task_type"
    )
    by_bucket = _compute_by_group(
        judgements, candidate_ids, autochecks, tc_map, group_by="bucket"
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
    write_json(json_out, report)

    # Also write markdown summary
    md_out = json_out.with_suffix(".md")
    _write_markdown_report(report, md_out)

    return json_out


def _compute_aggregate(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
    autochecks: list[AutoCheckRecord],
) -> AggregateBlock:
    """Compute aggregate stats across all judgements."""
    # Win rate (pairwise only)
    win_counts: dict[str, int] = defaultdict(int)
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

    win_rate: dict[str, float] = {}
    if pairwise_total > 0:
        for cid in candidate_ids:
            win_rate[cid] = round(win_counts[cid] / pairwise_total, 4)
        win_rate["tie"] = round(tie_count / pairwise_total, 4)

    # Mean score per metric
    metric_scores: dict[str, list[float]] = defaultdict(list)
    for jdg in judgements:
        for metric_id, score in jdg.scores.per_metric.items():
            metric_scores[metric_id].append(score)

    mean_score = {m: round(mean(v), 2) for m, v in metric_scores.items()}
    score_var = {m: round(variance(v), 4) for m, v in metric_scores.items()}

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

    return AggregateBlock(
        win_rate=win_rate,
        mean_score=mean_score,
        score_variance=score_var,
        critical_issue_count=critical_issue_count,
        notable_failures=failures,
    )


def _compute_by_group(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
    autochecks: list[AutoCheckRecord],
    tc_map: dict[str, Testcase],
    group_by: str,
) -> dict[str, AggregateBlock]:
    """Compute aggregate stats grouped by task_type or bucket."""
    groups: dict[str, list[JudgementRecord]] = defaultdict(list)
    ac_groups: dict[str, list[AutoCheckRecord]] = defaultdict(list)

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

    result = {}
    for key, jdgs in groups.items():
        result[key] = _compute_aggregate(jdgs, candidate_ids, ac_groups.get(key, []))

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
        lines.append("### Win Rate")
        for cid, rate in report.results.overall.win_rate.items():
            lines.append(f"- {cid}: {rate:.1%}")
        lines.append("")

    if report.results.overall.mean_score:
        lines.append("### Mean Scores by Metric")
        var_data = report.results.overall.score_variance
        lines.append("| Metric | Mean | Variance |")
        lines.append("|--------|------|----------|")
        for metric, score in sorted(report.results.overall.mean_score.items()):
            v = var_data.get(metric, 0.0)
            lines.append(f"| {metric} | {score:.2f} | {v:.4f} |")
        lines.append("")

    # By task type
    if report.results.by_task:
        lines.append("## Results by Task Type")
        lines.append("")
        for task_type, agg in report.results.by_task.items():
            lines.append(f"### {task_type}")
            if agg.win_rate:
                for cid, rate in agg.win_rate.items():
                    lines.append(f"- {cid}: {rate:.1%}")
            lines.append("")

    # By bucket
    if report.results.by_bucket:
        lines.append("## Results by Input Length Bucket")
        lines.append("")
        for bucket, agg in report.results.by_bucket.items():
            lines.append(f"### Bucket: {bucket}")
            if agg.win_rate:
                for cid, rate in agg.win_rate.items():
                    lines.append(f"- {cid}: {rate:.1%}")
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
        lines.append("")
        for f in failures:
            lines.append(f"- **{f.testcase_id}** ({f.candidate_id}): {f.reason}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
