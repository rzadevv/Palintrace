#!/usr/bin/env python3
"""Execute the frozen H4-N confirmatory probe after complete preflight."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import memlint.evaluation.retrieval_negation_confirmatory as probe
from memlint.evaluation.experimental_lexical import ExperimentalLexicalRetriever
from memlint.models import MemoryScope, NormalizedMemory, NormalizedStore
from memlint.retrieval import (
    RetrievalAuditRequest,
    assess_paired_retrieval_challenge,
    run_retrieval_audit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = (
    REPOSITORY_ROOT / probe.RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_PATH
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen H4-N synthetic confirmatory retrieval probe. "
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
        raise ValueError("confirmatory results must remain outside the repository")
    if resolved.exists():
        raise ValueError("confirmatory output path must not already exist")
    if not resolved.parent.exists():
        raise ValueError("confirmatory output parent must already exist")
    return resolved


def _build_store(
    scenario: probe.RetrievalNegationScenario,
    condition: probe.RetrievalNegationCondition | None,
) -> NormalizedStore:
    memories = (
        scenario.baseline_memories
        if condition is None
        else scenario.memories_for_condition(condition)
    )
    scope = MemoryScope(user_id=scenario.scope_user_id)
    return NormalizedStore(
        adapter=probe.RETRIEVAL_NEGATION_CONFIRMATORY_ID,
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
    """Construct the exact frozen retriever only after fixture preflight."""

    return ExperimentalLexicalRetriever(store)


def _execute_scenario(
    scenario: probe.RetrievalNegationScenario,
) -> probe.RetrievalNegationScenarioObservation:
    baseline_store = _build_store(scenario, None)
    baseline_request = RetrievalAuditRequest(
        request_id=f"{scenario.scenario_id}:baseline",
        query=scenario.query,
        expected_memory_ids=scenario.expected_memory_ids,
        top_k=probe.FROZEN_TOP_K,
    )
    baseline_observation = run_retrieval_audit(
        store=baseline_store,
        request=baseline_request,
        retriever=_build_retriever(baseline_store),
    )

    condition_observations: list[probe.RetrievalNegationConditionObservation] = []
    for condition in probe.CONDITION_ORDER:
        mutated_store = _build_store(scenario, condition)
        request_id = f"{scenario.scenario_id}:{condition.value}"
        mutated_request = RetrievalAuditRequest(
            request_id=request_id,
            query=scenario.query,
            expected_memory_ids=scenario.expected_memory_ids,
            top_k=probe.FROZEN_TOP_K,
        )
        mutated_observation = run_retrieval_audit(
            store=mutated_store,
            request=mutated_request,
            retriever=_build_retriever(mutated_store),
        )
        paired_assessment = assess_paired_retrieval_challenge(
            baseline_observation,
            mutated_observation,
            policy=probe.FROZEN_POLICY,
            case_id=request_id,
        )
        condition_observations.append(
            probe.RetrievalNegationConditionObservation(
                scenario_id=scenario.scenario_id,
                condition=condition,
                mutated_observation=mutated_observation,
                paired_assessment=paired_assessment,
            )
        )
    return probe.RetrievalNegationScenarioObservation(
        scenario_id=scenario.scenario_id,
        domain=scenario.domain,
        baseline_observation=baseline_observation,
        conditions=tuple(condition_observations),
    )


def _execute_probe(
    spec: probe.RetrievalNegationConfirmatorySpec,
) -> probe.RetrievalNegationExecutionResult:
    scenarios = tuple(_execute_scenario(scenario) for scenario in spec.scenarios)
    return probe.RetrievalNegationExecutionResult(
        schema_version=probe.RETRIEVAL_NEGATION_CONFIRMATORY_SCHEMA_VERSION,
        probe_id=probe.RETRIEVAL_NEGATION_CONFIRMATORY_ID,
        hypothesis_id=probe.RETRIEVAL_NEGATION_CONFIRMATORY_HYPOTHESIS_ID,
        fixture_sha256=probe.RETRIEVAL_NEGATION_CONFIRMATORY_FIXTURE_SHA256,
        scenarios=scenarios,
        summary=probe.summarize_retrieval_negation_confirmatory(scenarios),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight frozen bytes and output path before any retrieval construction."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec = probe.preflight_retrieval_negation_confirmatory(FROZEN_FIXTURE_PATH)
        output_path = _validate_output_path(args.output)
    except (OSError, ValueError, probe.RetrievalNegationConfirmatoryInputError) as error:
        parser.error(str(error))

    result_text = _execute_probe(spec).to_json()
    if output_path is None:
        print(result_text, end="")
    else:
        output_path.write_text(result_text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
