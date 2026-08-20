#!/usr/bin/env python3
"""Run the versioned engineering probe for one pinned local NLI judge."""

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

from memlint.semantics import LocalNLISemanticJudge, SemanticRelation


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    premise: str
    hypothesis: str
    expected: SemanticRelation


def _load_cases(path: Path) -> tuple[ProbeCase, ...]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("probe cases must be a nonempty JSON list")
    cases: list[ProbeCase] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "premise",
            "hypothesis",
            "expected_relation",
        }:
            raise ValueError("each probe case must contain exactly the documented fields")
        cases.append(
            ProbeCase(
                case_id=item["id"],
                premise=item["premise"],
                hypothesis=item["hypothesis"],
                expected=SemanticRelation(item["expected_relation"]),
            )
        )
    return tuple(cases)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local NLI development sanity probe (not a MemLint benchmark)."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--latency-runs", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.latency_runs < 3:
        raise ValueError("--latency-runs must be at least 3")
    cases = _load_cases(args.cases)
    judge = LocalNLISemanticJudge(
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
    )

    total_model_calls = 0
    total_input_tokens = 0

    def run(case: ProbeCase) -> SemanticRelation:
        nonlocal total_model_calls, total_input_tokens
        judgment = judge.judge(premise=case.premise, hypothesis=case.hypothesis)
        total_model_calls += judgment.usage.model_calls
        total_input_tokens += judgment.usage.input_tokens
        return judgment.relation

    run(cases[0])

    correct = Counter[SemanticRelation]()
    totals = Counter(case.expected for case in cases)
    confusion = Counter[tuple[SemanticRelation, SemanticRelation]]()
    incorrect: list[tuple[str, SemanticRelation, SemanticRelation]] = []
    for case in cases:
        predicted = run(case)
        confusion[(case.expected, predicted)] += 1
        if predicted is case.expected:
            correct[case.expected] += 1
        else:
            incorrect.append((case.case_id, case.expected, predicted))

    latencies_ms: list[float] = []
    for _ in range(args.latency_runs):
        for case in cases:
            start = time.perf_counter_ns()
            run(case)
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    correct_count = sum(correct.values())
    print("probe_kind: development sanity probe; not a MemLint benchmark")
    print(f"model_id: {args.model_id}")
    print(f"revision: {args.revision}")
    print(f"device: {args.device}")
    print(f"cpu_architecture: {platform.machine()}")
    print(f"transformers_version: {version('transformers')}")
    print(f"torch_version: {version('torch')}")
    print(f"case_count: {len(cases)}")
    print(f"correct_count: {correct_count}")
    print(f"accuracy: {correct_count / len(cases):.6f}")
    for relation in SemanticRelation:
        print(f"{relation.value}: {correct[relation]}/{totals[relation]}")
    for expected in SemanticRelation:
        for predicted in SemanticRelation:
            print(
                "confusion: "
                f"expected={expected.value} predicted={predicted.value} "
                f"count={confusion[(expected, predicted)]}"
            )
    for case_id, expected, predicted in incorrect:
        print(
            f"incorrect: id={case_id} expected={expected.value} predicted={predicted.value}"
        )
    print(f"median_inference_latency_ms: {statistics.median(latencies_ms):.6f}")
    print(
        "latency_method: one untimed warm-up call, then median of "
        f"{len(latencies_ms)} individually timed CPU calls "
        f"({args.latency_runs} complete fixture passes)"
    )
    print(f"total_model_calls: {total_model_calls}")
    print(f"total_input_tokens: {total_input_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
