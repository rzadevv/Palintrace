#!/usr/bin/env python3
"""Compare two symmetric contradiction-pair policies with pinned MiniLM on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any

from memlint.semantics import (
    LocalNLISemanticJudge,
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
)

MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
DEVICE = "cpu"
DEFAULT_CASES = Path("tests/fixtures/contradiction_pair_probe_v0.1.json")


class PairClass(StrEnum):
    """Development-probe pair labels."""

    CONTRADICTION = "contradiction"
    COMPATIBLE = "compatible"


class ContradictionAggregationPolicy(StrEnum):
    """The two eligible symmetric aggregation policies."""

    ANY_DIRECTION = "any_direction"
    BOTH_DIRECTIONS = "both_directions"


@dataclass(frozen=True)
class ContradictionProbeCase:
    case_id: str
    memory_a: str
    memory_b: str
    expected_pair_class: PairClass


@dataclass(frozen=True)
class DirectionalResult:
    relation: SemanticRelation
    score: float
    input_tokens: int


@dataclass(frozen=True)
class PairResult:
    case_id: str
    expected_pair_class: PairClass
    a_to_b: DirectionalResult
    b_to_a: DirectionalResult


@dataclass(frozen=True)
class PolicyReport:
    policy: ContradictionAggregationPolicy
    correct_ids: tuple[str, ...]
    contradiction_ids: tuple[str, ...]
    missed_contradiction_ids: tuple[str, ...]
    normal_false_positive_ids: tuple[str, ...]
    temporal_false_positive_ids: tuple[str, ...]
    asymmetric_contradiction_ids: tuple[str, ...]

    @property
    def correct_count(self) -> int:
        return len(self.correct_ids)

    @property
    def total_false_positives(self) -> int:
        return len(self.normal_false_positive_ids) + len(self.temporal_false_positive_ids)


@dataclass(frozen=True)
class TimedPass:
    results: tuple[PairResult, ...]
    latencies_ms: tuple[float, ...]
    model_calls: int
    input_tokens: int


def load_cases(path: Path = DEFAULT_CASES) -> tuple[ContradictionProbeCase, ...]:
    """Load the fixed development pair fixture with its exact public shape."""

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 18:
        raise ValueError("contradiction pair probe must contain exactly 18 cases")

    cases: list[ContradictionProbeCase] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "memory_a",
            "memory_b",
            "expected_pair_class",
        }:
            raise ValueError("each contradiction pair probe case must use the frozen field set")
        case_id = item["id"]
        memory_a = item["memory_a"]
        memory_b = item["memory_b"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("contradiction pair probe case IDs must be nonblank strings")
        if not isinstance(memory_a, str) or not memory_a.strip():
            raise ValueError("contradiction pair probe memory_a must be a nonblank string")
        if not isinstance(memory_b, str) or not memory_b.strip():
            raise ValueError("contradiction pair probe memory_b must be a nonblank string")
        cases.append(
            ContradictionProbeCase(
                case_id=case_id,
                memory_a=memory_a,
                memory_b=memory_b,
                expected_pair_class=PairClass(item["expected_pair_class"]),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("contradiction pair probe case IDs must be unique")
    return tuple(cases)


def aggregate_pair_relations(
    policy: ContradictionAggregationPolicy,
    a_to_b: SemanticRelation,
    b_to_a: SemanticRelation,
) -> PairClass:
    """Aggregate two directional relations without consulting their scores."""

    directional_contradictions = (
        a_to_b is SemanticRelation.CONTRADICTION,
        b_to_a is SemanticRelation.CONTRADICTION,
    )
    if policy is ContradictionAggregationPolicy.ANY_DIRECTION:
        is_contradiction = any(directional_contradictions)
    else:
        is_contradiction = all(directional_contradictions)
    return PairClass.CONTRADICTION if is_contradiction else PairClass.COMPATIBLE


def policy_report(
    results: tuple[PairResult, ...],
    policy: ContradictionAggregationPolicy,
) -> PolicyReport:
    """Calculate exact case-level outcomes for one aggregation policy."""

    correct_ids: list[str] = []
    contradiction_ids: list[str] = []
    missed_contradiction_ids: list[str] = []
    normal_false_positive_ids: list[str] = []
    temporal_false_positive_ids: list[str] = []
    asymmetric_contradiction_ids: list[str] = []

    for result in results:
        decision = aggregate_pair_relations(
            policy,
            result.a_to_b.relation,
            result.b_to_a.relation,
        )
        if decision is result.expected_pair_class:
            correct_ids.append(result.case_id)
        if result.expected_pair_class is PairClass.CONTRADICTION:
            if decision is PairClass.CONTRADICTION:
                contradiction_ids.append(result.case_id)
            else:
                missed_contradiction_ids.append(result.case_id)
        elif decision is PairClass.CONTRADICTION:
            if result.case_id.startswith("N"):
                normal_false_positive_ids.append(result.case_id)
            elif result.case_id.startswith("T"):
                temporal_false_positive_ids.append(result.case_id)
            else:
                raise ValueError("compatible probe case IDs must start with N or T")
        if (
            result.a_to_b.relation is not result.b_to_a.relation
            and decision is PairClass.CONTRADICTION
        ):
            asymmetric_contradiction_ids.append(result.case_id)

    return PolicyReport(
        policy=policy,
        correct_ids=tuple(correct_ids),
        contradiction_ids=tuple(contradiction_ids),
        missed_contradiction_ids=tuple(missed_contradiction_ids),
        normal_false_positive_ids=tuple(normal_false_positive_ids),
        temporal_false_positive_ids=tuple(temporal_false_positive_ids),
        asymmetric_contradiction_ids=tuple(asymmetric_contradiction_ids),
    )


def is_order_invariant(
    results: tuple[PairResult, ...],
    policy: ContradictionAggregationPolicy,
) -> bool:
    """Return whether swapping every directional pair preserves every decision."""

    return all(
        aggregate_pair_relations(policy, result.a_to_b.relation, result.b_to_a.relation)
        is aggregate_pair_relations(policy, result.b_to_a.relation, result.a_to_b.relation)
        for result in results
    )


def asymmetric_results(results: tuple[PairResult, ...]) -> tuple[PairResult, ...]:
    """Return directional disagreements without turning them into another policy."""

    return tuple(
        result for result in results if result.a_to_b.relation is not result.b_to_a.relation
    )


def select_primary_policy(
    reports: tuple[PolicyReport, PolicyReport],
) -> tuple[ContradictionAggregationPolicy, str]:
    """Apply the frozen false-positive-first selection priority."""

    if {report.policy for report in reports} != set(ContradictionAggregationPolicy):
        raise ValueError("selection requires exactly ANY_DIRECTION and BOTH_DIRECTIONS reports")

    ranked = sorted(
        reports,
        key=lambda report: (
            report.total_false_positives,
            len(report.temporal_false_positive_ids),
            -len(report.contradiction_ids),
            len(report.asymmetric_contradiction_ids),
            report.policy is ContradictionAggregationPolicy.ANY_DIRECTION,
        ),
    )
    selected, other = ranked
    if selected.total_false_positives != other.total_false_positives:
        reason = "fewer false positives across the 12 compatible cases"
    elif len(selected.temporal_false_positive_ids) != len(
        other.temporal_false_positive_ids
    ):
        reason = "fewer false positives across the six temporal-compatible cases"
    elif len(selected.contradiction_ids) != len(other.contradiction_ids):
        reason = "more correctly detected clear contradiction cases"
    elif len(selected.asymmetric_contradiction_ids) != len(
        other.asymmetric_contradiction_ids
    ):
        reason = "fewer pair decisions resting on a directional asymmetry"
    else:
        reason = "conservative BOTH_DIRECTIONS tie-break"
    return selected.policy, reason


def _directional_result(judgment: SemanticJudgment) -> DirectionalResult:
    return DirectionalResult(
        relation=judgment.relation,
        score=judgment.score,
        input_tokens=judgment.usage.input_tokens,
    )


def _timed_judgment(
    judge: SemanticJudge,
    *,
    premise: str,
    hypothesis: str,
) -> tuple[SemanticJudgment, float]:
    start = time.perf_counter_ns()
    judgment = judge.judge(premise=premise, hypothesis=hypothesis)
    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
    return judgment, latency_ms


def evaluate_timed_pass(
    judge: SemanticJudge,
    cases: tuple[ContradictionProbeCase, ...],
) -> TimedPass:
    """Judge every unordered pair in both directions exactly once."""

    results: list[PairResult] = []
    latencies_ms: list[float] = []
    model_calls = 0
    input_tokens = 0
    for case in cases:
        forward, forward_latency = _timed_judgment(
            judge,
            premise=case.memory_a,
            hypothesis=case.memory_b,
        )
        reverse, reverse_latency = _timed_judgment(
            judge,
            premise=case.memory_b,
            hypothesis=case.memory_a,
        )
        results.append(
            PairResult(
                case_id=case.case_id,
                expected_pair_class=case.expected_pair_class,
                a_to_b=_directional_result(forward),
                b_to_a=_directional_result(reverse),
            )
        )
        latencies_ms.extend((forward_latency, reverse_latency))
        model_calls += forward.usage.model_calls + reverse.usage.model_calls
        input_tokens += forward.usage.input_tokens + reverse.usage.input_tokens
    return TimedPass(
        results=tuple(results),
        latencies_ms=tuple(latencies_ms),
        model_calls=model_calls,
        input_tokens=input_tokens,
    )


def _relations(results: tuple[PairResult, ...]) -> tuple[tuple[SemanticRelation, ...], ...]:
    return tuple((result.a_to_b.relation, result.b_to_a.relation) for result in results)


def _format_ids(case_ids: tuple[str, ...]) -> str:
    return ",".join(case_ids) if case_ids else "none"


def _print_policy_report(report: PolicyReport, case_count: int) -> None:
    print(f"policy: {report.policy.value}")
    print(f"correct: {report.correct_count}/{case_count} ids={_format_ids(report.correct_ids)}")
    print(
        "contradictions_detected: "
        f"{len(report.contradiction_ids)} ids={_format_ids(report.contradiction_ids)}"
    )
    print(
        "contradictions_missed: "
        f"{len(report.missed_contradiction_ids)} "
        f"ids={_format_ids(report.missed_contradiction_ids)}"
    )
    print(
        "normal_compatible_false_positives: "
        f"{len(report.normal_false_positive_ids)} "
        f"ids={_format_ids(report.normal_false_positive_ids)}"
    )
    print(
        "temporal_compatible_false_positives: "
        f"{len(report.temporal_false_positive_ids)} "
        f"ids={_format_ids(report.temporal_false_positive_ids)}"
    )
    print(
        f"total_false_positives: {report.total_false_positives} "
        f"ids={_format_ids(report.normal_false_positive_ids + report.temporal_false_positive_ids)}"
    )
    print(
        "asymmetric_contradiction_decisions: "
        f"{len(report.asymmetric_contradiction_ids)} "
        f"ids={_format_ids(report.asymmetric_contradiction_ids)}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two symmetric contradiction pair policies with the pinned local CPU judge; "
            "this is a development probe, not benchmark performance."
        )
    )
    parser.add_argument("--timed-passes", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timed_passes < 3:
        raise ValueError("--timed-passes must be at least 3")
    cases = load_cases()
    judge = LocalNLISemanticJudge(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device=DEVICE,
    )

    warmup = judge.judge(premise=cases[0].memory_a, hypothesis=cases[0].memory_b)
    passes = tuple(evaluate_timed_pass(judge, cases) for _ in range(args.timed_passes))
    reference_results = passes[0].results
    reference_relations = _relations(reference_results)
    if any(_relations(pass_result.results) != reference_relations for pass_result in passes[1:]):
        raise RuntimeError("directional relations changed across complete timed passes")

    reports = tuple(
        policy_report(reference_results, policy) for policy in ContradictionAggregationPolicy
    )
    if len(reports) != 2:  # pragma: no cover - enum is frozen directly above
        raise RuntimeError("exactly two contradiction aggregation policies are required")
    candidate_policy, selection_reason = select_primary_policy((reports[0], reports[1]))

    editor_a = "User's favorite editor is Neovim."
    editor_b = "User's favorite editor is VS Code."
    editor_forward = judge.judge(premise=editor_a, hypothesis=editor_b)
    editor_reverse = judge.judge(premise=editor_b, hypothesis=editor_a)
    editor_decision = aggregate_pair_relations(
        candidate_policy,
        editor_forward.relation,
        editor_reverse.relation,
    )

    all_latencies = tuple(
        latency for pass_result in passes for latency in pass_result.latencies_ms
    )
    timed_model_calls = sum(pass_result.model_calls for pass_result in passes)
    timed_input_tokens = sum(pass_result.input_tokens for pass_result in passes)
    diagnostic_model_calls = (
        editor_forward.usage.model_calls + editor_reverse.usage.model_calls
    )
    diagnostic_input_tokens = (
        editor_forward.usage.input_tokens + editor_reverse.usage.input_tokens
    )

    print("probe_kind: contradiction pair-policy development probe; not benchmark performance")
    print(f"model_id: {MODEL_ID}")
    print(f"revision: {MODEL_REVISION}")
    print(f"device: {DEVICE}")
    print(f"cpu_architecture: {platform.machine()}")
    print(f"transformers_version: {version('transformers')}")
    print(f"torch_version: {version('torch')}")
    print(f"case_count: {len(cases)}")
    print("directional_judgments_per_complete_pass: 36")
    for result in reference_results:
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
    for report in reports:
        _print_policy_report(report, len(cases))
        print(f"order_invariant: {is_order_invariant(reference_results, report.policy)}")
    asymmetries = asymmetric_results(reference_results)
    if not asymmetries:
        print("asymmetry: none")
    for result in asymmetries:
        print(
            f"asymmetry: id={result.case_id} "
            f"A->B={result.a_to_b.relation.value} B->A={result.b_to_a.relation.value}"
        )
    print(f"priority_candidate_policy: {candidate_policy.value}")
    print(f"priority_candidate_reason: {selection_reason}")
    print(f"complete_timed_passes: {args.timed_passes}")
    print(f"median_directional_cpu_latency_ms: {statistics.median(all_latencies):.6f}")
    print(
        "latency_method: one untimed warm-up call, then median of "
        f"{len(all_latencies)} directional CPU calls "
        f"({args.timed_passes} complete 18-pair passes)"
    )
    print(f"timed_model_calls: {timed_model_calls}")
    print(f"timed_input_tokens: {timed_input_tokens}")
    print(f"warmup_model_calls: {warmup.usage.model_calls}")
    print("part_2_compatibility_diagnostic: excluded_from_policy_selection=true")
    print(
        "part_2_compatibility_diagnostic: direction=A->B "
        f"relation={editor_forward.relation.value} score={editor_forward.score:.9f} "
        f"input_tokens={editor_forward.usage.input_tokens}"
    )
    print(
        "part_2_compatibility_diagnostic: direction=B->A "
        f"relation={editor_reverse.relation.value} score={editor_reverse.score:.9f} "
        f"input_tokens={editor_reverse.usage.input_tokens}"
    )
    print(
        "part_2_compatibility_diagnostic: "
        f"priority_candidate_result={editor_decision.value}"
    )
    print(f"diagnostic_model_calls: {diagnostic_model_calls}")
    print(f"diagnostic_input_tokens: {diagnostic_input_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
