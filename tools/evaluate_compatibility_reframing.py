#!/usr/bin/env python3
"""Evaluate the frozen simultaneous-compatibility NLI reframing on CPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memlint.semantics import (  # noqa: E402
    LocalNLISemanticJudge,
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
)
from tools.evaluate_contradiction_models import (  # noqa: E402
    CANDIDATES as PART_4F2_CANDIDATES,
)
from tools.evaluate_contradiction_models import ModelCandidate  # noqa: E402
from tools.evaluate_contradiction_policy import (  # noqa: E402
    DEFAULT_CASES,
    ContradictionProbeCase,
    PairClass,
    load_cases,
)

DEVICE = "cpu"
CONTRADICTION_PROBE_SHA256 = (
    "0744755a747164a9ff646a094b78fdf132e2b89de09556cf17f0189054d72744"
)
COMPATIBILITY_HYPOTHESIS = "These two memory claims can both be true as stated."
PART_2_EDITOR_A = "User's favorite editor is Neovim."
PART_2_EDITOR_B = "User's favorite editor is VS Code."

MINIMUM_INCOMPATIBLE_DETECTED = 5
MAXIMUM_COMPATIBLE_FALSE_INCOMPATIBLE = 0
MAXIMUM_TEMPORAL_FALSE_INCOMPATIBLE = 0
MINIMUM_COMPATIBLE_CORRECT = 10
MINIMUM_TEMPORAL_COMPATIBLE_CORRECT = 5

_PART_4G_MODEL_IDS = (
    "cross-encoder/nli-MiniLM2-L6-H768",
    "tasksource/deberta-small-long-nli",
)
_PART_4F2_BY_ID = {candidate.model_id: candidate for candidate in PART_4F2_CANDIDATES}
CANDIDATES = tuple(_PART_4F2_BY_ID[model_id] for model_id in _PART_4G_MODEL_IDS)
CANDIDATES_BY_NAME = {
    "minilm": CANDIDATES[0],
    "tasksource": CANDIDATES[1],
}


class CompatibilityClass(StrEnum):
    """Tool-only result of the simultaneous-compatibility question."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class DirectionalCompatibilityResult:
    nli_relation: SemanticRelation
    compatibility_class: CompatibilityClass
    score: float
    input_tokens: int


@dataclass(frozen=True)
class CompatibilityProbeResult:
    case_id: str
    expected: CompatibilityClass
    ab: DirectionalCompatibilityResult
    ba: DirectionalCompatibilityResult
    final: CompatibilityClass


@dataclass(frozen=True)
class CompatibilityReport:
    incompatible_detected_ids: tuple[str, ...]
    missed_incompatible_ids: tuple[str, ...]
    normal_compatible_correct_ids: tuple[str, ...]
    normal_false_incompatible_ids: tuple[str, ...]
    normal_uncertain_ids: tuple[str, ...]
    temporal_compatible_correct_ids: tuple[str, ...]
    temporal_false_incompatible_ids: tuple[str, ...]
    temporal_uncertain_ids: tuple[str, ...]
    order_disagreement_ids: tuple[str, ...]

    @property
    def compatible_correct_count(self) -> int:
        return len(self.normal_compatible_correct_ids) + len(
            self.temporal_compatible_correct_ids
        )

    @property
    def compatible_false_incompatible_count(self) -> int:
        return len(self.normal_false_incompatible_ids) + len(
            self.temporal_false_incompatible_ids
        )

@dataclass(frozen=True)
class TimedCompatibilityPass:
    results: tuple[CompatibilityProbeResult, ...]
    latencies_ms: tuple[float, ...]
    model_calls: int
    input_tokens: int


@dataclass(frozen=True)
class ModelEvaluation:
    candidate: ModelCandidate
    results: tuple[CompatibilityProbeResult, ...]
    report: CompatibilityReport
    median_directional_cpu_latency_ms: float
    timed_passes: int
    timed_model_calls: int
    timed_input_tokens: int
    warmup_model_calls: int


@dataclass(frozen=True)
class Part2Diagnostic:
    candidate: ModelCandidate
    ab: DirectionalCompatibilityResult
    ba: DirectionalCompatibilityResult
    final: CompatibilityClass


def validate_candidate_definitions(candidates: tuple[ModelCandidate, ...]) -> None:
    """Require exactly the two pre-frozen Part 4G model configurations."""

    if candidates != CANDIDATES or tuple(candidate.model_id for candidate in candidates) != (
        "cross-encoder/nli-MiniLM2-L6-H768",
        "tasksource/deberta-small-long-nli",
    ):
        raise ValueError("Part 4G requires exactly the frozen MiniLM and Tasksource candidates")
    expected_revisions = (
        "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
        "9a77395d4d3751be9e2a69c4ae318491d9b3fffb",
    )
    if tuple(candidate.revision for candidate in candidates) != expected_revisions:
        raise ValueError("Part 4G candidate revisions do not match the frozen commits")


def validate_fixture_hash(path: Path = DEFAULT_CASES) -> None:
    """Reject any change to the frozen contradiction-pair probe bytes."""

    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != CONTRADICTION_PROBE_SHA256:
        raise ValueError(
            "contradiction pair probe hash mismatch: "
            f"expected={CONTRADICTION_PROBE_SHA256} actual={actual}"
        )


def render_compatibility_premise(claim_1: str, claim_2: str) -> str:
    """Render the one frozen compatibility premise without normalization."""

    if not isinstance(claim_1, str) or not claim_1.strip():
        raise ValueError("claim_1 must be a nonblank string")
    if not isinstance(claim_2, str) or not claim_2.strip():
        raise ValueError("claim_2 must be a nonblank string")
    return f"Memory claim 1: {claim_1}\nMemory claim 2: {claim_2}"


def nli_relation_to_compatibility(relation: SemanticRelation) -> CompatibilityClass:
    """Apply the frozen relation-only mapping without consulting scores."""

    if relation is SemanticRelation.ENTAILMENT:
        return CompatibilityClass.COMPATIBLE
    if relation is SemanticRelation.CONTRADICTION:
        return CompatibilityClass.INCOMPATIBLE
    return CompatibilityClass.UNCERTAIN


def aggregate_compatibility(
    ab: CompatibilityClass,
    ba: CompatibilityClass,
) -> CompatibilityClass:
    """Apply the one conservative symmetric aggregation rule."""

    if ab is CompatibilityClass.INCOMPATIBLE and ba is CompatibilityClass.INCOMPATIBLE:
        return CompatibilityClass.INCOMPATIBLE
    if ab is CompatibilityClass.COMPATIBLE and ba is CompatibilityClass.COMPATIBLE:
        return CompatibilityClass.COMPATIBLE
    return CompatibilityClass.UNCERTAIN


def _expected_compatibility(case: ContradictionProbeCase) -> CompatibilityClass:
    if case.expected_pair_class is PairClass.CONTRADICTION:
        return CompatibilityClass.INCOMPATIBLE
    return CompatibilityClass.COMPATIBLE


def _directional_result(judgment: SemanticJudgment) -> DirectionalCompatibilityResult:
    return DirectionalCompatibilityResult(
        nli_relation=judgment.relation,
        compatibility_class=nli_relation_to_compatibility(judgment.relation),
        score=judgment.score,
        input_tokens=judgment.usage.input_tokens,
    )


def _timed_judgment(
    judge: SemanticJudge,
    *,
    premise: str,
) -> tuple[SemanticJudgment, float]:
    start = time.perf_counter_ns()
    judgment = judge.judge(premise=premise, hypothesis=COMPATIBILITY_HYPOTHESIS)
    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
    return judgment, latency_ms


def evaluate_timed_pass(
    judge: SemanticJudge,
    cases: tuple[ContradictionProbeCase, ...],
) -> TimedCompatibilityPass:
    """Evaluate exact AB and BA renderings for every frozen pair."""

    results: list[CompatibilityProbeResult] = []
    latencies_ms: list[float] = []
    model_calls = 0
    input_tokens = 0
    for case in cases:
        ab_judgment, ab_latency = _timed_judgment(
            judge,
            premise=render_compatibility_premise(case.memory_a, case.memory_b),
        )
        ba_judgment, ba_latency = _timed_judgment(
            judge,
            premise=render_compatibility_premise(case.memory_b, case.memory_a),
        )
        ab = _directional_result(ab_judgment)
        ba = _directional_result(ba_judgment)
        results.append(
            CompatibilityProbeResult(
                case_id=case.case_id,
                expected=_expected_compatibility(case),
                ab=ab,
                ba=ba,
                final=aggregate_compatibility(
                    ab.compatibility_class,
                    ba.compatibility_class,
                ),
            )
        )
        latencies_ms.extend((ab_latency, ba_latency))
        model_calls += ab_judgment.usage.model_calls + ba_judgment.usage.model_calls
        input_tokens += ab_judgment.usage.input_tokens + ba_judgment.usage.input_tokens
    return TimedCompatibilityPass(
        results=tuple(results),
        latencies_ms=tuple(latencies_ms),
        model_calls=model_calls,
        input_tokens=input_tokens,
    )


def compatibility_report(
    results: tuple[CompatibilityProbeResult, ...],
) -> CompatibilityReport:
    """Calculate fixed incompatibility, compatibility, abstention, and order counts."""

    incompatible_detected: list[str] = []
    missed_incompatible: list[str] = []
    normal_correct: list[str] = []
    normal_false_incompatible: list[str] = []
    normal_uncertain: list[str] = []
    temporal_correct: list[str] = []
    temporal_false_incompatible: list[str] = []
    temporal_uncertain: list[str] = []
    order_disagreements: list[str] = []

    for result in results:
        if result.ab.compatibility_class is not result.ba.compatibility_class:
            order_disagreements.append(result.case_id)
        if result.expected is CompatibilityClass.INCOMPATIBLE:
            if result.final is CompatibilityClass.INCOMPATIBLE:
                incompatible_detected.append(result.case_id)
            else:
                missed_incompatible.append(result.case_id)
            continue

        if result.case_id.startswith("N"):
            if result.final is CompatibilityClass.COMPATIBLE:
                normal_correct.append(result.case_id)
            elif result.final is CompatibilityClass.INCOMPATIBLE:
                normal_false_incompatible.append(result.case_id)
            else:
                normal_uncertain.append(result.case_id)
        elif result.case_id.startswith("T"):
            if result.final is CompatibilityClass.COMPATIBLE:
                temporal_correct.append(result.case_id)
            elif result.final is CompatibilityClass.INCOMPATIBLE:
                temporal_false_incompatible.append(result.case_id)
            else:
                temporal_uncertain.append(result.case_id)
        else:
            raise ValueError("compatible fixture IDs must start with N or T")

    return CompatibilityReport(
        incompatible_detected_ids=tuple(incompatible_detected),
        missed_incompatible_ids=tuple(missed_incompatible),
        normal_compatible_correct_ids=tuple(normal_correct),
        normal_false_incompatible_ids=tuple(normal_false_incompatible),
        normal_uncertain_ids=tuple(normal_uncertain),
        temporal_compatible_correct_ids=tuple(temporal_correct),
        temporal_false_incompatible_ids=tuple(temporal_false_incompatible),
        temporal_uncertain_ids=tuple(temporal_uncertain),
        order_disagreement_ids=tuple(order_disagreements),
    )


def passes_readiness_gate(report: CompatibilityReport) -> bool:
    """Apply the pre-frozen precision and coverage criteria."""

    return (
        len(report.incompatible_detected_ids) >= MINIMUM_INCOMPATIBLE_DETECTED
        and report.compatible_false_incompatible_count
        <= MAXIMUM_COMPATIBLE_FALSE_INCOMPATIBLE
        and len(report.temporal_false_incompatible_ids)
        <= MAXIMUM_TEMPORAL_FALSE_INCOMPATIBLE
        and report.compatible_correct_count >= MINIMUM_COMPATIBLE_CORRECT
        and len(report.temporal_compatible_correct_ids)
        >= MINIMUM_TEMPORAL_COMPATIBLE_CORRECT
    )


def select_ready_model(
    evaluations: tuple[ModelEvaluation, ...],
) -> ModelEvaluation | None:
    """Rank only gate-passing models by the frozen Part 4G priority."""

    passing = tuple(
        evaluation for evaluation in evaluations if passes_readiness_gate(evaluation.report)
    )
    if not passing:
        return None
    return min(
        passing,
        key=lambda evaluation: (
            evaluation.report.compatible_false_incompatible_count,
            -evaluation.report.compatible_correct_count,
            -len(evaluation.report.incompatible_detected_ids),
            sum(
                result.final is CompatibilityClass.UNCERTAIN
                for result in evaluation.results
            ),
            evaluation.candidate.safetensors_bytes,
            evaluation.median_directional_cpu_latency_ms,
        ),
    )


def _result_signature(
    results: tuple[CompatibilityProbeResult, ...],
) -> tuple[tuple[SemanticRelation, SemanticRelation, CompatibilityClass], ...]:
    return tuple(
        (result.ab.nli_relation, result.ba.nli_relation, result.final)
        for result in results
    )


def evaluate_candidate(
    candidate: ModelCandidate,
    cases: tuple[ContradictionProbeCase, ...],
    *,
    timed_passes: int,
) -> ModelEvaluation:
    """Run one pinned candidate without loading any other model."""

    judge = LocalNLISemanticJudge(
        model_id=candidate.model_id,
        revision=candidate.revision,
        device=DEVICE,
    )
    warmup_premise = render_compatibility_premise(
        cases[0].memory_a,
        cases[0].memory_b,
    )
    try:
        warmup = judge.judge(
            premise=warmup_premise,
            hypothesis=COMPATIBILITY_HYPOTHESIS,
        )
        passes = tuple(evaluate_timed_pass(judge, cases) for _ in range(timed_passes))
    finally:
        del judge
        gc.collect()

    reference_results = passes[0].results
    reference_signature = _result_signature(reference_results)
    if any(
        _result_signature(timed.results) != reference_signature for timed in passes[1:]
    ):
        raise RuntimeError(
            f"compatibility results changed across timed passes for {candidate.model_id}"
        )
    latencies = tuple(latency for timed in passes for latency in timed.latencies_ms)
    return ModelEvaluation(
        candidate=candidate,
        results=reference_results,
        report=compatibility_report(reference_results),
        median_directional_cpu_latency_ms=statistics.median(latencies),
        timed_passes=timed_passes,
        timed_model_calls=sum(timed.model_calls for timed in passes),
        timed_input_tokens=sum(timed.input_tokens for timed in passes),
        warmup_model_calls=warmup.usage.model_calls,
    )


def run_part_2_diagnostic(candidate: ModelCandidate) -> Part2Diagnostic:
    """Evaluate only the two visible editor strings under the frozen representation."""

    judge = LocalNLISemanticJudge(
        model_id=candidate.model_id,
        revision=candidate.revision,
        device=DEVICE,
    )
    try:
        ab_judgment = judge.judge(
            premise=render_compatibility_premise(PART_2_EDITOR_A, PART_2_EDITOR_B),
            hypothesis=COMPATIBILITY_HYPOTHESIS,
        )
        ba_judgment = judge.judge(
            premise=render_compatibility_premise(PART_2_EDITOR_B, PART_2_EDITOR_A),
            hypothesis=COMPATIBILITY_HYPOTHESIS,
        )
    finally:
        del judge
        gc.collect()
    ab = _directional_result(ab_judgment)
    ba = _directional_result(ba_judgment)
    return Part2Diagnostic(
        candidate=candidate,
        ab=ab,
        ba=ba,
        final=aggregate_compatibility(
            ab.compatibility_class,
            ba.compatibility_class,
        ),
    )


def _format_ids(case_ids: tuple[str, ...]) -> str:
    return ",".join(case_ids) if case_ids else "none"


def _print_direction(label: str, result: DirectionalCompatibilityResult) -> None:
    print(
        f"direction: {label} nli_relation={result.nli_relation.value} "
        f"score={result.score:.9f} compatibility={result.compatibility_class.value} "
        f"input_tokens={result.input_tokens}"
    )


def _is_error_or_abstention(result: CompatibilityProbeResult) -> bool:
    return result.final is not result.expected


def _print_evaluation(evaluation: ModelEvaluation) -> None:
    print(f"model_id: {evaluation.candidate.model_id}")
    print(f"revision: {evaluation.candidate.revision}")
    print(f"safetensors_artifact_bytes: {evaluation.candidate.safetensors_bytes}")
    for result in evaluation.results:
        print(f"case: id={result.case_id} expected={result.expected.value}")
        _print_direction("AB", result.ab)
        _print_direction("BA", result.ba)
        print(f"final_compatibility: id={result.case_id} result={result.final.value}")
        if _is_error_or_abstention(result):
            print(
                f"error_or_abstention: id={result.case_id} "
                f"AB={result.ab.nli_relation.value}/{result.ab.score:.9f} "
                f"BA={result.ba.nli_relation.value}/{result.ba.score:.9f} "
                f"final={result.final.value}"
            )
    for case_id in evaluation.report.order_disagreement_ids:
        result = next(item for item in evaluation.results if item.case_id == case_id)
        print(
            f"order_disagreement: id={case_id} "
            f"AB={result.ab.compatibility_class.value} "
            f"BA={result.ba.compatibility_class.value}"
        )
    report = evaluation.report
    print(
        "clear_incompatible_detected: "
        f"{len(report.incompatible_detected_ids)}/6 "
        f"ids={_format_ids(report.incompatible_detected_ids)}"
    )
    print(f"missed_incompatible_ids: {_format_ids(report.missed_incompatible_ids)}")
    print(
        "normal_compatible_correct: "
        f"{len(report.normal_compatible_correct_ids)}/6 "
        f"ids={_format_ids(report.normal_compatible_correct_ids)}"
    )
    print(
        "normal_false_incompatible: "
        f"{len(report.normal_false_incompatible_ids)}/6 "
        f"ids={_format_ids(report.normal_false_incompatible_ids)}"
    )
    print(
        "normal_uncertain: "
        f"{len(report.normal_uncertain_ids)}/6 "
        f"ids={_format_ids(report.normal_uncertain_ids)}"
    )
    print(
        "temporal_compatible_correct: "
        f"{len(report.temporal_compatible_correct_ids)}/6 "
        f"ids={_format_ids(report.temporal_compatible_correct_ids)}"
    )
    print(
        "temporal_false_incompatible: "
        f"{len(report.temporal_false_incompatible_ids)}/6 "
        f"ids={_format_ids(report.temporal_false_incompatible_ids)}"
    )
    print(
        "temporal_uncertain: "
        f"{len(report.temporal_uncertain_ids)}/6 "
        f"ids={_format_ids(report.temporal_uncertain_ids)}"
    )
    print(f"compatible_correct: {report.compatible_correct_count}/12")
    print(
        "compatible_false_incompatible: "
        f"{report.compatible_false_incompatible_count}/12"
    )
    print(f"order_disagreement_ids: {_format_ids(report.order_disagreement_ids)}")
    print(f"readiness_gate: {'PASS' if passes_readiness_gate(report) else 'FAIL'}")
    print(f"complete_timed_passes: {evaluation.timed_passes}")
    print(
        "median_directional_cpu_latency_ms: "
        f"{evaluation.median_directional_cpu_latency_ms:.6f}"
    )
    print(f"timed_model_calls: {evaluation.timed_model_calls}")
    print(f"timed_input_tokens: {evaluation.timed_input_tokens}")
    print(f"warmup_model_calls: {evaluation.warmup_model_calls}")


def _print_part_2_diagnostic(diagnostic: Part2Diagnostic) -> None:
    print(f"part_2_diagnostic_model: {diagnostic.candidate.model_id}")
    print(f"visible_memory_a: {PART_2_EDITOR_A}")
    print(f"visible_memory_b: {PART_2_EDITOR_B}")
    _print_direction("AB", diagnostic.ab)
    _print_direction("BA", diagnostic.ba)
    print(f"part_2_diagnostic_final: {diagnostic.final.value}")
    print("hidden_exclusive_value_supplied_to_detector: no")
    print(
        "part_2_observability_note: this demonstrates only a text-visible example; "
        "it does not establish detectability for every mutation carrying hidden "
        "exclusive_value metadata"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pinned simultaneous-compatibility NLI development probe on CPU; "
            "this is not benchmark performance."
        )
    )
    parser.add_argument("--candidate", choices=tuple(CANDIDATES_BY_NAME), required=True)
    parser.add_argument("--timed-passes", type=int, default=3)
    parser.add_argument(
        "--part-2-diagnostic-only",
        action="store_true",
        help="run only the excluded visible-string diagnostic after model selection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_candidate_definitions(CANDIDATES)
    candidate = CANDIDATES_BY_NAME[args.candidate]
    if args.part_2_diagnostic_only:
        _print_part_2_diagnostic(run_part_2_diagnostic(candidate))
        return 0
    if args.timed_passes < 3:
        raise ValueError("--timed-passes must be at least 3")

    validate_fixture_hash()
    cases = load_cases()
    evaluation = evaluate_candidate(candidate, cases, timed_passes=args.timed_passes)

    print(
        "probe_kind: simultaneous-compatibility NLI reframing development probe; "
        "not benchmark performance"
    )
    print(f"device: {DEVICE}")
    print(f"cpu_architecture: {platform.machine()}")
    print(f"transformers_version: {version('transformers')}")
    print(f"torch_version: {version('torch')}")
    print(f"fixture_sha256: {CONTRADICTION_PROBE_SHA256}")
    print("premise_template: Memory claim 1: <CLAIM_1>\\nMemory claim 2: <CLAIM_2>")
    print(f"hypothesis: {COMPATIBILITY_HYPOTHESIS}")
    print("relation_mapping: entailment=compatible")
    print("relation_mapping: contradiction=incompatible")
    print("relation_mapping: neutral=uncertain")
    print(
        "pair_aggregation: both incompatible=incompatible; both compatible=compatible; "
        "otherwise=uncertain"
    )
    print(f"gate_minimum_incompatible_detected: {MINIMUM_INCOMPATIBLE_DETECTED}/6")
    print(
        "gate_maximum_compatible_false_incompatible: "
        f"{MAXIMUM_COMPATIBLE_FALSE_INCOMPATIBLE}/12"
    )
    print(
        "gate_maximum_temporal_false_incompatible: "
        f"{MAXIMUM_TEMPORAL_FALSE_INCOMPATIBLE}/6"
    )
    print(f"gate_minimum_compatible_correct: {MINIMUM_COMPATIBLE_CORRECT}/12")
    print(
        "gate_minimum_temporal_compatible_correct: "
        f"{MINIMUM_TEMPORAL_COMPATIBLE_CORRECT}/6"
    )
    _print_evaluation(evaluation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
