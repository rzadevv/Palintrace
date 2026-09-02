from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import palintrace.evaluation.execution as execution
from palintrace.checkers import CheckerResult, CheckerStats
from palintrace.evaluation import (
    BENCHMARK_ID,
    BENCHMARK_SPEC_SHA256,
    BenchmarkExecutionProvenance,
    BenchmarkExecutionResult,
    CleanControlCaseExecution,
    CleanControlEvaluation,
    MutationTrialEvaluation,
    RetrievalCaseExecution,
    StaticCaseExecution,
    StaticDefectBenchmarkSummary,
    build_execution_provenance,
    summarize_retrieval_challenges,
)
from palintrace.mutations import BaseStoreStatus, GoldLabelUnit
from palintrace.retrieval import (
    RetrievalChallengeOutcome,
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
    RetrievalUsage,
    assess_paired_retrieval_challenge,
)
from palintrace.taxonomy import DefectClass

IMPLEMENTED = (
    DefectClass.ORPHANED_PROVENANCE,
    DefectClass.REDUNDANCY_BLOAT,
    DefectClass.STALE_ACTIVE,
    DefectClass.PRIVACY_SCOPE_VIOLATION,
    DefectClass.UNSUPPORTED_CLAIM,
)


def _checker_result(defect: DefectClass) -> CheckerResult:
    return CheckerResult(
        checker_id=defect.value,
        checker_version="1.0",
        defect_class=defect,
        findings=(),
        stats=CheckerStats(memories_scanned=1, findings_emitted=0),
    )


def _static_execution() -> StaticCaseExecution:
    defect = DefectClass.ORPHANED_PROVENANCE
    trial = MutationTrialEvaluation(
        mutation_id="mutation-toy",
        defect_class=defect,
        subtype="missing_transcript",
        gold_unit=GoldLabelUnit.MEMORY,
        base_store_status=BaseStoreStatus.CURATED_CLEAN,
        checker_id=defect.value,
        checker_version="1.0",
        injected_positive_detected=False,
        gold_matching_finding_ids=(),
        duplicate_positive_findings=0,
        verified_clean_alert_finding_ids=(),
        unknown_natural_alert_finding_ids=(),
        mutation_context_alert_finding_ids=(),
        total_findings=0,
    )
    return StaticCaseExecution(
        case_id="S-TOY",
        mutation_id="mutation-toy",
        checker_result=_checker_result(defect),
        trial_evaluation=trial,
    )


def _clean_execution() -> CleanControlCaseExecution:
    defect = DefectClass.ORPHANED_PROVENANCE
    evaluation = CleanControlEvaluation(
        case_id="C-TOY",
        defect_class=defect,
        base_fixture_id="TOY",
        checker_id=defect.value,
        checker_version="1.0",
        alert_present=False,
        finding_ids=(),
        findings_emitted=0,
    )
    return CleanControlCaseExecution(
        case_id="C-TOY",
        checker_result=_checker_result(defect),
        clean_control_evaluation=evaluation,
    )


def _retrieval_execution() -> RetrievalCaseExecution:
    common = {
        "query_sha256": "a" * 64,
        "expected_memory_ids": ("target",),
        "top_k": 3,
        "retriever_id": "experimental_lexical",
        "retriever_version": "0.1",
        "hits": (),
        "usage": RetrievalUsage(retrieval_calls=1, candidate_count=2),
    }
    baseline = RetrievalObservation(request_id="R-TOY:baseline", **common)
    mutated = RetrievalObservation(request_id="R-TOY:mutated", **common)
    paired = assess_paired_retrieval_challenge(
        baseline,
        mutated,
        policy=RetrievalSufficiencyPolicy.ALL_EXPECTED,
        case_id="R-TOY",
    )
    assert paired.outcome is RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
    return RetrievalCaseExecution(
        case_id="R-TOY",
        baseline_observation=baseline,
        mutated_observation=mutated,
        paired_assessment=paired,
    )


def _summaries() -> tuple[StaticDefectBenchmarkSummary, ...]:
    return tuple(
        StaticDefectBenchmarkSummary(
            defect_class=defect,
            positive_trials=1,
            positive_trials_detected=0,
            positive_trials_missed=1,
            injected_positive_recall=0.0,
            clean_controls=1,
            clean_controls_with_alert=0,
            clean_control_alert_rate=0.0,
            verified_clean_alerts=0,
            unknown_natural_alerts=0,
            mutation_context_alerts=0,
            duplicate_positive_findings=0,
        )
        for defect in IMPLEMENTED
    )


def test_execution_artifact_is_canonical_private_and_deterministic() -> None:
    retrieval = _retrieval_execution()
    result = BenchmarkExecutionResult(
        benchmark_id=BENCHMARK_ID,
        benchmark_spec_sha256=BENCHMARK_SPEC_SHA256,
        static_cases=(_static_execution(),),
        clean_controls=(_clean_execution(),),
        static_defect_summaries=tuple(reversed(_summaries())),
        retrieval_cases=(retrieval,),
        retrieval_summary=summarize_retrieval_challenges(
            (retrieval.paired_assessment,)
        ),
    )
    serialized = result.to_json()
    assert result.schema_version == "0.1"
    assert result.benchmark_spec_sha256 == BENCHMARK_SPEC_SHA256
    assert result.to_json() == BenchmarkExecutionResult.model_validate_json(
        serialized
    ).to_json()
    assert tuple(item.defect_class for item in result.static_defect_summaries) == tuple(
        sorted(IMPLEMENTED, key=lambda item: item.value)
    )
    payload = json.loads(serialized)
    text = serialized.lower()
    assert "timestamp" not in payload
    assert "latency" not in text
    assert "where does" not in text
    assert "memory content" not in text
    assert "replace_from" not in text
    assert "replace_to" not in text


def test_execution_artifact_requires_frozen_benchmark_sha_and_all_summaries() -> None:
    retrieval = _retrieval_execution()
    values = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_spec_sha256": BENCHMARK_SPEC_SHA256,
        "static_cases": (_static_execution(),),
        "clean_controls": (_clean_execution(),),
        "static_defect_summaries": _summaries(),
        "retrieval_cases": (retrieval,),
        "retrieval_summary": summarize_retrieval_challenges(
            (retrieval.paired_assessment,)
        ),
    }
    with pytest.raises(ValidationError, match="SHA"):
        BenchmarkExecutionResult(**{**values, "benchmark_spec_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="all five"):
        BenchmarkExecutionResult(
            **{**values, "static_defect_summaries": _summaries()[:-1]}
        )


def test_environment_provenance_is_separate_safe_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "torch": "2.test",
        "transformers": "5.test",
        "tokenizers": "0.test",
        "safetensors": "0.safe",
    }
    monkeypatch.setattr(execution, "_package_version", versions.__getitem__)
    monkeypatch.setattr(execution.platform, "python_version", lambda: "3.test")
    monkeypatch.setattr(execution.platform, "system", lambda: "ToyOS")
    monkeypatch.setattr(execution.platform, "release", lambda: "1")
    monkeypatch.setattr(execution.platform, "machine", lambda: "toy64")
    provenance = build_execution_provenance()
    assert isinstance(provenance, BenchmarkExecutionProvenance)
    assert provenance.platform == "ToyOS 1 toy64"
    assert provenance.device == "cpu"
    assert provenance.benchmark_spec_sha256 == BENCHMARK_SPEC_SHA256
    assert provenance.to_json() == BenchmarkExecutionProvenance.model_validate_json(
        provenance.to_json()
    ).to_json()
    fields = BenchmarkExecutionProvenance.model_fields
    assert {
        "hostname",
        "username",
        "home",
        "cache_path",
        "latency",
        "timestamp",
    }.isdisjoint(fields)
