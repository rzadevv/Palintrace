#!/usr/bin/env python3
"""Run the frozen Part 4F2 contradiction NLI robustness sweep on CPU."""

from __future__ import annotations

import argparse
import gc
import platform
import re
import statistics
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memlint.semantics import LocalNLISemanticJudge, SemanticJudgment, SemanticRelation
from tools.evaluate_contradiction_policy import (
    ContradictionAggregationPolicy,
    ContradictionProbeCase,
    DirectionalResult,
    PairResult,
    PolicyReport,
    aggregate_pair_relations,
    asymmetric_results,
    evaluate_timed_pass,
    is_order_invariant,
    load_cases,
    policy_report,
)  # noqa: E402

DEVICE = "cpu"
FULL_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
MINIMUM_CONTRADICTIONS_DETECTED = 5
MAXIMUM_TEMPORAL_FALSE_POSITIVES = 0
MAXIMUM_TOTAL_COMPATIBLE_FALSE_POSITIVES = 1

PART_2_EDITOR_A = "User's favorite editor is Neovim."
PART_2_EDITOR_B = "User's favorite editor is VS Code."


@dataclass(frozen=True)
class ModelCandidate:
    """One immutable model configuration admitted to the sweep."""

    display_name: str
    model_id: str
    revision: str
    license_id: str | None
    safetensors_bytes: int


CANDIDATES = (
    ModelCandidate(
        display_name="MiniLM",
        model_id="cross-encoder/nli-MiniLM2-L6-H768",
        revision="b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
        license_id="apache-2.0",
        safetensors_bytes=328_499_560,
    ),
    ModelCandidate(
        display_name="DeBERTa v3 small",
        model_id="cross-encoder/nli-deberta-v3-small",
        revision="fa2804872c3b4bd748f38c0185cc85775361e735",
        license_id="apache-2.0",
        safetensors_bytes=567_605_820,
    ),
    ModelCandidate(
        display_name="DeBERTa v3 base",
        model_id="cross-encoder/nli-deberta-v3-base",
        revision="6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        license_id="apache-2.0",
        safetensors_bytes=737_726_552,
    ),
    ModelCandidate(
        display_name="Tasksource DeBERTa small",
        model_id="tasksource/deberta-small-long-nli",
        revision="9a77395d4d3751be9e2a69c4ae318491d9b3fffb",
        license_id="apache-2.0",
        safetensors_bytes=567_601_628,
    ),
)


@dataclass(frozen=True)
class ModelEvaluation:
    candidate: ModelCandidate
    results: tuple[PairResult, ...]
    reports: tuple[PolicyReport, PolicyReport]
    median_directional_cpu_latency_ms: float
    timed_passes: int
    timed_model_calls: int
    timed_input_tokens: int
    warmup_model_calls: int


@dataclass(frozen=True)
class ModelPolicyCombination:
    candidate: ModelCandidate
    report: PolicyReport
    median_directional_cpu_latency_ms: float


@dataclass(frozen=True)
class Part2Diagnostic:
    candidate: ModelCandidate
    a_to_b: DirectionalResult
    b_to_a: DirectionalResult


def validate_candidate_definitions(
    candidates: tuple[ModelCandidate, ...],
) -> None:
    """Reject mutable or malformed recorded model configurations."""

    if len(candidates) != 4:
        raise ValueError("the Part 4F2 sweep requires exactly four model candidates")
    if len({candidate.model_id for candidate in candidates}) != len(candidates):
        raise ValueError("Part 4F2 model IDs must be unique")
    for candidate in candidates:
        if not candidate.display_name.strip() or not candidate.model_id.strip():
            raise ValueError("Part 4F2 candidate names and model IDs must be nonblank")
        if candidate.revision.casefold() == "main":
            raise ValueError("mutable 'main' revisions are forbidden for recorded runs")
        if FULL_COMMIT_SHA.fullmatch(candidate.revision) is None:
            raise ValueError("recorded model revisions must be full lowercase commit SHAs")
        if candidate.license_id is not None and not candidate.license_id.strip():
            raise ValueError("an available model license must be nonblank")
        if candidate.safetensors_bytes <= 0:
            raise ValueError("safetensors artifact size must be positive")


def passes_readiness_gate(report: PolicyReport) -> bool:
    """Apply the pre-frozen high-precision development gate."""

    return (
        len(report.contradiction_ids) >= MINIMUM_CONTRADICTIONS_DETECTED
        and len(report.temporal_false_positive_ids)
        <= MAXIMUM_TEMPORAL_FALSE_POSITIVES
        and report.total_false_positives <= MAXIMUM_TOTAL_COMPATIBLE_FALSE_POSITIVES
    )


def model_policy_combinations(
    evaluations: tuple[ModelEvaluation, ...],
) -> tuple[ModelPolicyCombination, ...]:
    """Flatten model results without creating another aggregation policy."""

    return tuple(
        ModelPolicyCombination(
            candidate=evaluation.candidate,
            report=report,
            median_directional_cpu_latency_ms=(
                evaluation.median_directional_cpu_latency_ms
            ),
        )
        for evaluation in evaluations
        for report in evaluation.reports
    )


def select_ready_combination(
    combinations: tuple[ModelPolicyCombination, ...],
) -> ModelPolicyCombination | None:
    """Rank only gate-passing combinations by the frozen selection criteria."""

    passing = tuple(
        combination
        for combination in combinations
        if passes_readiness_gate(combination.report)
    )
    if not passing:
        return None
    return min(
        passing,
        key=lambda combination: (
            combination.report.total_false_positives,
            len(combination.report.temporal_false_positive_ids),
            -len(combination.report.contradiction_ids),
            combination.candidate.safetensors_bytes,
            combination.median_directional_cpu_latency_ms,
        ),
    )


def _relations(results: tuple[PairResult, ...]) -> tuple[tuple[SemanticRelation, ...], ...]:
    return tuple((result.a_to_b.relation, result.b_to_a.relation) for result in results)


def evaluate_candidate(
    candidate: ModelCandidate,
    cases: tuple[ContradictionProbeCase, ...],
    *,
    timed_passes: int,
) -> ModelEvaluation:
    """Evaluate one pinned model on every frozen pair in both directions."""

    judge = LocalNLISemanticJudge(
        model_id=candidate.model_id,
        revision=candidate.revision,
        device=DEVICE,
    )
    try:
        warmup = judge.judge(premise=cases[0].memory_a, hypothesis=cases[0].memory_b)
        passes = tuple(evaluate_timed_pass(judge, cases) for _ in range(timed_passes))
    finally:
        del judge
        gc.collect()

    reference_results = passes[0].results
    reference_relations = _relations(reference_results)
    if any(_relations(timed.results) != reference_relations for timed in passes[1:]):
        raise RuntimeError(
            f"directional relations changed across timed passes for {candidate.model_id}"
        )
    reports = tuple(
        policy_report(reference_results, policy)
        for policy in ContradictionAggregationPolicy
    )
    if len(reports) != 2:  # pragma: no cover - the imported policy enum is frozen
        raise RuntimeError("exactly two contradiction aggregation policies are required")
    latencies = tuple(latency for timed in passes for latency in timed.latencies_ms)
    return ModelEvaluation(
        candidate=candidate,
        results=reference_results,
        reports=(reports[0], reports[1]),
        median_directional_cpu_latency_ms=statistics.median(latencies),
        timed_passes=timed_passes,
        timed_model_calls=sum(timed.model_calls for timed in passes),
        timed_input_tokens=sum(timed.input_tokens for timed in passes),
        warmup_model_calls=warmup.usage.model_calls,
    )


def _directional_result(judgment: SemanticJudgment) -> DirectionalResult:
    return DirectionalResult(
        relation=judgment.relation,
        score=judgment.score,
        input_tokens=judgment.usage.input_tokens,
    )


def run_part_2_diagnostic(candidate: ModelCandidate) -> Part2Diagnostic:
    """Run the excluded favorite-editor pair after sweep selection is complete."""

    judge = LocalNLISemanticJudge(
        model_id=candidate.model_id,
        revision=candidate.revision,
        device=DEVICE,
    )
    try:
        a_to_b = judge.judge(premise=PART_2_EDITOR_A, hypothesis=PART_2_EDITOR_B)
        b_to_a = judge.judge(premise=PART_2_EDITOR_B, hypothesis=PART_2_EDITOR_A)
    finally:
        del judge
        gc.collect()
    return Part2Diagnostic(
        candidate=candidate,
        a_to_b=_directional_result(a_to_b),
        b_to_a=_directional_result(b_to_a),
    )


def _format_ids(case_ids: tuple[str, ...]) -> str:
    return ",".join(case_ids) if case_ids else "none"


def _contradiction_scores(result: PairResult) -> str:
    scores: list[str] = []
    if result.a_to_b.relation is SemanticRelation.CONTRADICTION:
        scores.append(f"A->B={result.a_to_b.score:.9f}")
    if result.b_to_a.relation is SemanticRelation.CONTRADICTION:
        scores.append(f"B->A={result.b_to_a.score:.9f}")
    if not scores:  # pragma: no cover - called only for policy false positives
        raise ValueError("false-positive pair has no contradiction relation")
    return ",".join(scores)


def _false_positive_results(
    results: tuple[PairResult, ...],
    report: PolicyReport,
) -> tuple[PairResult, ...]:
    false_positive_ids = set(
        report.normal_false_positive_ids + report.temporal_false_positive_ids
    )
    return tuple(result for result in results if result.case_id in false_positive_ids)


def _print_policy_report(evaluation: ModelEvaluation, report: PolicyReport) -> None:
    print(f"policy: {report.policy.value}")
    print(f"clear_contradictions_detected: {len(report.contradiction_ids)}/6")
    print(f"missed_contradiction_ids: {_format_ids(report.missed_contradiction_ids)}")
    print(
        "normal_compatible_false_positives: "
        f"{len(report.normal_false_positive_ids)}/6 "
        f"ids={_format_ids(report.normal_false_positive_ids)}"
    )
    print(
        "temporal_compatible_false_positives: "
        f"{len(report.temporal_false_positive_ids)}/6 "
        f"ids={_format_ids(report.temporal_false_positive_ids)}"
    )
    print(f"total_compatible_false_positives: {report.total_false_positives}/12")
    print(f"correct: {report.correct_count}/18")
    print(f"readiness_gate: {'PASS' if passes_readiness_gate(report) else 'FAIL'}")
    print(f"order_invariant: {is_order_invariant(evaluation.results, report.policy)}")
    for result in _false_positive_results(evaluation.results, report):
        print(
            f"false_positive: policy={report.policy.value} id={result.case_id} "
            f"contradiction_scores={_contradiction_scores(result)}"
        )


def _print_model_evaluation(evaluation: ModelEvaluation) -> None:
    candidate = evaluation.candidate
    print(f"model: {candidate.display_name}")
    print(f"model_id: {candidate.model_id}")
    print(f"revision: {candidate.revision}")
    print(f"license: {candidate.license_id or 'unavailable'}")
    print(f"safetensors_artifact_bytes: {candidate.safetensors_bytes}")
    for result in evaluation.results:
        print(
            f"directional: id={result.case_id} direction=A->B "
            f"relation={result.a_to_b.relation.value} score={result.a_to_b.score:.9f} "
            f"input_tokens={result.a_to_b.input_tokens}"
        )
        print(
            f"directional: id={result.case_id} direction=B->A "
            f"relation={result.b_to_a.relation.value} score={result.b_to_a.score:.9f} "
            f"input_tokens={result.b_to_a.input_tokens}"
        )
    asymmetries = asymmetric_results(evaluation.results)
    if not asymmetries:
        print("directional_asymmetry: none")
    for result in asymmetries:
        print(
            f"directional_asymmetry: id={result.case_id} "
            f"A->B={result.a_to_b.relation.value} "
            f"B->A={result.b_to_a.relation.value}"
        )
    for report in evaluation.reports:
        _print_policy_report(evaluation, report)
    print(f"complete_timed_passes: {evaluation.timed_passes}")
    print(
        "median_directional_cpu_latency_ms: "
        f"{evaluation.median_directional_cpu_latency_ms:.6f}"
    )
    print(f"timed_model_calls: {evaluation.timed_model_calls}")
    print(f"timed_input_tokens: {evaluation.timed_input_tokens}")
    print(f"warmup_model_calls: {evaluation.warmup_model_calls}")


def _print_high_confidence_false_positives(
    evaluations: tuple[ModelEvaluation, ...],
) -> None:
    print("high_confidence_false_positives: selected_relation_scores_not_thresholds")
    for evaluation in evaluations:
        emitted = False
        for report in evaluation.reports:
            for result in _false_positive_results(evaluation.results, report):
                emitted = True
                print(
                    f"high_confidence_false_positive: model={evaluation.candidate.model_id} "
                    f"policy={report.policy.value} id={result.case_id} "
                    f"contradiction_scores={_contradiction_scores(result)}"
                )
        if not emitted:
            print(
                "high_confidence_false_positive: "
                f"model={evaluation.candidate.model_id} none"
            )


def _print_part_2_diagnostic(diagnostic: Part2Diagnostic) -> None:
    print(f"part_2_diagnostic_model: {diagnostic.candidate.model_id}")
    print("part_2_diagnostic_excluded_from_readiness_and_selection: true")
    print(
        "part_2_diagnostic: direction=A->B "
        f"relation={diagnostic.a_to_b.relation.value} "
        f"score={diagnostic.a_to_b.score:.9f} "
        f"input_tokens={diagnostic.a_to_b.input_tokens}"
    )
    print(
        "part_2_diagnostic: direction=B->A "
        f"relation={diagnostic.b_to_a.relation.value} "
        f"score={diagnostic.b_to_a.score:.9f} "
        f"input_tokens={diagnostic.b_to_a.input_tokens}"
    )
    for policy in ContradictionAggregationPolicy:
        decision = aggregate_pair_relations(
            policy,
            diagnostic.a_to_b.relation,
            diagnostic.b_to_a.relation,
        )
        print(f"part_2_diagnostic: policy={policy.value} result={decision.value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the four-model contradiction NLI robustness development sweep on CPU; "
            "this is not benchmark performance."
        )
    )
    parser.add_argument("--timed-passes", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timed_passes < 3:
        raise ValueError("--timed-passes must be at least 3")
    validate_candidate_definitions(CANDIDATES)
    cases = load_cases()

    evaluations = tuple(
        evaluate_candidate(candidate, cases, timed_passes=args.timed_passes)
        for candidate in CANDIDATES
    )
    combinations = model_policy_combinations(evaluations)
    selection = select_ready_combination(combinations)

    # This optional diagnostic is deliberately run only after all selection calculations.
    diagnostics = tuple(run_part_2_diagnostic(candidate) for candidate in CANDIDATES)

    print("probe_kind: contradiction NLI robustness development sweep; not benchmark performance")
    print(f"device: {DEVICE}")
    print(f"cpu_architecture: {platform.machine()}")
    print(f"transformers_version: {version('transformers')}")
    print(f"torch_version: {version('torch')}")
    print(f"case_count: {len(cases)}")
    print("directional_judgments_per_complete_pass_per_model: 36")
    print(f"gate_minimum_clear_contradictions: {MINIMUM_CONTRADICTIONS_DETECTED}/6")
    print(f"gate_maximum_temporal_false_positives: {MAXIMUM_TEMPORAL_FALSE_POSITIVES}/6")
    print(
        "gate_maximum_total_compatible_false_positives: "
        f"{MAXIMUM_TOTAL_COMPATIBLE_FALSE_POSITIVES}/12"
    )
    for evaluation in evaluations:
        _print_model_evaluation(evaluation)
    _print_high_confidence_false_positives(evaluations)
    if selection is None:
        print("selected_model: none")
        print("selected_pair_policy: none")
        print(
            "selection_reason: no vanilla NLI configuration passed the pre-frozen "
            "high-precision gate"
        )
        print("selection_conclusion: VANILLA NLI NOT READY FOR INTERNAL CONTRADICTION")
    else:
        print(f"selected_model: {selection.candidate.model_id}")
        print(f"selected_pair_policy: {selection.report.policy.value}")
        print("selection_reason: highest-ranked gate-passing combination")
    for diagnostic in diagnostics:
        _print_part_2_diagnostic(diagnostic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
