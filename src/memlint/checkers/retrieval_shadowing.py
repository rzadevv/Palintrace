"""Project one recorded retrieval case into the frozen checker result envelope."""

from __future__ import annotations

from memlint.checkers.base import deterministic_finding_id
from memlint.checkers.models import (
    CheckerCost,
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
)
from memlint.retrieval import (
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
    assess_retrieval_sufficiency,
)
from memlint.taxonomy import DefectClass

_CHECKER_ID = "retrieval_shadowing"
_CHECKER_VERSION = "1.0"
_DEFECT_CLASS = DefectClass.RETRIEVAL_SHADOWING


def _insufficient_retrieval_evidence(
    observation: RetrievalObservation,
    *,
    policy: RetrievalSufficiencyPolicy,
    expected_memory_ids: tuple[str, ...],
    retrieved_expected_memory_ids: tuple[str, ...],
    missing_expected_memory_ids: tuple[str, ...],
) -> tuple[EvidenceItem, ...]:
    return (
        EvidenceItem(
            kind="insufficient_retrieval",
            message=(
                "Recorded retrieval did not satisfy the declared target sufficiency policy."
            ),
            data={
                "request_id": observation.request_id,
                "query_sha256": observation.query_sha256,
                "policy": policy.value,
                "top_k": observation.top_k,
                "retriever_id": observation.retriever_id,
                "retriever_version": observation.retriever_version,
                "expected_memory_ids": list(expected_memory_ids),
                "retrieved_expected_memory_ids": list(retrieved_expected_memory_ids),
                "missing_expected_memory_ids": list(missing_expected_memory_ids),
            },
        ),
    )


def project_retrieval_shadowing_result(
    observation: RetrievalObservation,
    *,
    policy: RetrievalSufficiencyPolicy,
) -> CheckerResult:
    """Project one recorded retrieval case without store or retriever access."""

    assessment = assess_retrieval_sufficiency(observation, policy=policy)
    findings: tuple[Finding, ...] = ()
    if not assessment.sufficient:
        evidence = _insufficient_retrieval_evidence(
            observation,
            policy=assessment.policy,
            expected_memory_ids=assessment.expected_memory_ids,
            retrieved_expected_memory_ids=assessment.retrieved_expected_memory_ids,
            missing_expected_memory_ids=assessment.missing_expected_memory_ids,
        )
        memory_ids = assessment.missing_expected_memory_ids
        findings = (
            Finding(
                finding_id=deterministic_finding_id(
                    checker_id=_CHECKER_ID,
                    checker_version=_CHECKER_VERSION,
                    defect_class=_DEFECT_CLASS,
                    memory_ids=memory_ids,
                    evidence=evidence,
                ),
                defect_class=_DEFECT_CLASS,
                memory_ids=memory_ids,
                confidence=1.0,
                evidence=evidence,
            ),
        )

    return CheckerResult(
        checker_id=_CHECKER_ID,
        checker_version=_CHECKER_VERSION,
        defect_class=_DEFECT_CLASS,
        findings=findings,
        cost=CheckerCost(),
        stats=CheckerStats(
            memories_scanned=0,
            findings_emitted=len(findings),
            details={
                "retrieval_cases_assessed": 1,
                "expected_targets": len(assessment.expected_memory_ids),
                "retrieved_expected_targets": len(
                    assessment.retrieved_expected_memory_ids
                ),
                "missing_expected_targets": len(assessment.missing_expected_memory_ids),
                "retrieval_calls": observation.usage.retrieval_calls,
                "candidate_count": observation.usage.candidate_count,
                "hits_observed": len(observation.hits),
            },
        ),
    )
