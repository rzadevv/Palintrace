#!/usr/bin/env python3
"""Compare the two frozen evidence representations with pinned MiniLM on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from memlint.semantics import (
    EvidenceCompositionStyle,
    EvidenceSegment,
    LocalNLISemanticJudge,
    SemanticJudge,
    SemanticRelation,
    compose_evidence,
)

MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
DEFAULT_CASES = Path("tests/fixtures/evidence_composition_probe_v0.1.json")


@dataclass(frozen=True)
class EvidenceProbeCase:
    case_id: str
    segments: tuple[EvidenceSegment, ...]
    hypothesis: str
    expected: SemanticRelation


@dataclass(frozen=True)
class PreparedProbeCase:
    case_id: str
    premise: str
    hypothesis: str
    expected: SemanticRelation


def _load_cases(path: Path) -> tuple[EvidenceProbeCase, ...]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 18:
        raise ValueError("composition probe must contain exactly 18 cases")

    cases: list[EvidenceProbeCase] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "segments",
            "hypothesis",
            "expected_relation",
        }:
            raise ValueError("each composition probe case must use the frozen field set")
        segments_payload = item["segments"]
        if not isinstance(segments_payload, list) or not segments_payload:
            raise ValueError("each composition probe case requires resolved segments")
        cases.append(
            EvidenceProbeCase(
                case_id=item["id"],
                segments=tuple(
                    EvidenceSegment.model_validate(segment) for segment in segments_payload
                ),
                hypothesis=item["hypothesis"],
                expected=SemanticRelation(item["expected_relation"]),
            )
        )
    return tuple(cases)


def _prepare_cases(
    cases: tuple[EvidenceProbeCase, ...],
    style: EvidenceCompositionStyle,
) -> tuple[PreparedProbeCase, ...]:
    return tuple(
        PreparedProbeCase(
            case_id=case.case_id,
            premise=compose_evidence(case.segments, style=style).text,
            hypothesis=case.hypothesis,
            expected=case.expected,
        )
        for case in cases
    )


def _evaluate_style(
    *,
    judge: SemanticJudge,
    cases: tuple[EvidenceProbeCase, ...],
    style: EvidenceCompositionStyle,
    latency_runs: int,
) -> None:
    prepared = _prepare_cases(cases, style)
    total_model_calls = 0
    total_input_tokens = 0

    def run(case: PreparedProbeCase) -> SemanticRelation:
        nonlocal total_model_calls, total_input_tokens
        judgment = judge.judge(premise=case.premise, hypothesis=case.hypothesis)
        total_model_calls += judgment.usage.model_calls
        total_input_tokens += judgment.usage.input_tokens
        return judgment.relation

    run(prepared[0])

    correct = Counter[SemanticRelation]()
    totals = Counter(case.expected for case in prepared)
    incorrect: list[tuple[str, SemanticRelation, SemanticRelation]] = []
    for case in prepared:
        predicted = run(case)
        if predicted is case.expected:
            correct[case.expected] += 1
        else:
            incorrect.append((case.case_id, case.expected, predicted))

    latencies_ms: list[float] = []
    for _ in range(latency_runs):
        for case in prepared:
            start = time.perf_counter_ns()
            run(case)
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    print(f"style: {style.value}")
    print(f"overall: {sum(correct.values())}/{len(prepared)}")
    for relation in SemanticRelation:
        print(f"{relation.value}: {correct[relation]}/{totals[relation]}")
    if not incorrect:
        print("incorrect: none")
    for case_id, expected, predicted in incorrect:
        print(
            f"incorrect: id={case_id} expected={expected.value} predicted={predicted.value}"
        )
    print(f"total_model_calls: {total_model_calls}")
    print(f"total_input_tokens: {total_input_tokens}")
    print(f"median_inference_latency_ms: {statistics.median(latencies_ms):.6f}")
    print(
        "latency_method: one untimed warm-up call, then median of "
        f"{len(latencies_ms)} individually timed CPU inference calls "
        f"({latency_runs} complete fixture passes)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the two frozen evidence representations as a development sanity probe; "
            "not a MemLint benchmark."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--latency-runs", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.latency_runs < 3:
        raise ValueError("--latency-runs must be at least 3")
    cases = _load_cases(args.cases)
    judge = LocalNLISemanticJudge(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device=args.device,
    )

    print("probe_kind: evidence-composition development probe; not a MemLint benchmark")
    print(f"model_id: {MODEL_ID}")
    print(f"revision: {MODEL_REVISION}")
    print(f"device: {args.device}")
    print(f"cpu_architecture: {platform.machine()}")
    print(f"transformers_version: {version('transformers')}")
    print(f"torch_version: {version('torch')}")
    print(f"case_count: {len(cases)}")
    for style in EvidenceCompositionStyle:
        _evaluate_style(
            judge=judge,
            cases=cases,
            style=style,
            latency_runs=args.latency_runs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
