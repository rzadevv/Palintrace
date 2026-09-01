#!/usr/bin/env python3
"""Execute the preregistered strong retrieval probe after complete preflight."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import memlint.evaluation.retrieval_strong_probe as probe
from memlint.evaluation.experimental_lexical import ExperimentalLexicalRetriever
from memlint.models import MemoryScope, NormalizedMemory, NormalizedStore
from memlint.retrieval import (
    RetrievalAuditRequest,
    assess_paired_retrieval_challenge,
    run_retrieval_audit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = REPOSITORY_ROOT / probe.RETRIEVAL_STRONG_PROBE_FIXTURE_PATH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the preregistered strong retrieval-shadowing probe. "
            "No retriever or condition overrides are accepted."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional external path for deterministic result JSON; default is stdout.",
    )
    return parser


def _validate_output_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    repository = REPOSITORY_ROOT.resolve()
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("retrieval probe results must remain outside the repository")
    if resolved.exists():
        raise ValueError("retrieval probe output path must not already exist")
    if not resolved.parent.exists():
        raise ValueError("retrieval probe output parent must already exist")
    return resolved


def _build_store(
    case: probe.RetrievalStrongProbeCase,
    *,
    include_distractors: bool,
) -> NormalizedStore:
    memories = (
        case.target_memory,
        *case.baseline_other_memories,
        *(case.distractor_memories if include_distractors else ()),
    )
    scope = MemoryScope(user_id=case.scope_user_id)
    return NormalizedStore(
        adapter=probe.RETRIEVAL_STRONG_PROBE_ID,
        memories=tuple(
            NormalizedMemory(
                id=memory.id,
                content=memory.content,
                scope=scope,
                active=True,
            )
            for memory in memories
        ),
    )


def _build_retriever(store: NormalizedStore) -> ExperimentalLexicalRetriever:
    """Construct the exact frozen retriever only after runner preflight."""

    return ExperimentalLexicalRetriever(store)


def _execute_case(
    case: probe.RetrievalStrongProbeCase,
) -> probe.RetrievalStrongProbeObservation:
    baseline_store = _build_store(case, include_distractors=False)
    mutated_store = _build_store(case, include_distractors=True)
    baseline_request = RetrievalAuditRequest(
        request_id=f"{case.case_id}:baseline",
        query=case.query,
        expected_memory_ids=case.expected_memory_ids,
        top_k=case.top_k,
    )
    mutated_request = RetrievalAuditRequest(
        request_id=f"{case.case_id}:mutated",
        query=case.query,
        expected_memory_ids=case.expected_memory_ids,
        top_k=case.top_k,
    )
    baseline = run_retrieval_audit(
        store=baseline_store,
        request=baseline_request,
        retriever=_build_retriever(baseline_store),
    )
    mutated = run_retrieval_audit(
        store=mutated_store,
        request=mutated_request,
        retriever=_build_retriever(mutated_store),
    )
    paired = assess_paired_retrieval_challenge(
        baseline,
        mutated,
        policy=case.policy,
        case_id=case.case_id,
    )
    return probe.RetrievalStrongProbeObservation(
        case_id=case.case_id,
        case_kind=case.case_kind,
        challenge_family=case.challenge_family,
        domain=case.domain,
        baseline_observation=baseline,
        mutated_observation=mutated,
        paired_assessment=paired,
    )


def _execute_probe(
    spec: probe.RetrievalStrongProbeSpec,
) -> probe.RetrievalStrongProbeExecutionResult:
    observations = tuple(_execute_case(case) for case in spec.cases)
    return probe.RetrievalStrongProbeExecutionResult(
        schema_version=probe.RETRIEVAL_STRONG_PROBE_SCHEMA_VERSION,
        probe_id=probe.RETRIEVAL_STRONG_PROBE_ID,
        fixture_sha256=probe.RETRIEVAL_STRONG_PROBE_FIXTURE_SHA256,
        observations=observations,
        summary=probe.summarize_retrieval_strong_probe(observations),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight frozen bytes and design before any retrieval construction."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec = probe.preflight_retrieval_strong_probe(FROZEN_FIXTURE_PATH)
        output_path = _validate_output_path(args.output)
    except (OSError, ValueError, probe.RetrievalStrongProbeInputError) as error:
        parser.error(str(error))

    result_text = _execute_probe(spec).to_json()
    if output_path is None:
        print(result_text, end="")
    else:
        output_path.write_text(result_text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
