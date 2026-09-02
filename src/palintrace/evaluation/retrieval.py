"""Baseline-eligible accounting for paired retrieval challenges."""

from __future__ import annotations

from palintrace.evaluation.models import EvaluationInputError, RetrievalChallengeSummary
from palintrace.retrieval import PairedRetrievalChallengeAssessment, RetrievalChallengeOutcome


def summarize_retrieval_challenges(
    assessments: tuple[PairedRetrievalChallengeAssessment, ...],
) -> RetrievalChallengeSummary:
    """Summarize one common retrieval condition using eligible cases only."""

    if not isinstance(assessments, tuple) or not assessments:
        raise EvaluationInputError("retrieval summary requires a nonempty assessment tuple")
    if any(
        not isinstance(assessment, PairedRetrievalChallengeAssessment)
        for assessment in assessments
    ):
        raise EvaluationInputError(
            "retrieval summary accepts only PairedRetrievalChallengeAssessment values"
        )

    first = assessments[0]
    condition_fields = ("policy", "retriever_id", "retriever_version", "top_k")
    mismatches = tuple(
        field_name
        for field_name in condition_fields
        if any(
            getattr(assessment, field_name) != getattr(first, field_name)
            for assessment in assessments[1:]
        )
    )
    if mismatches:
        raise EvaluationInputError(
            "retrieval summary conditions must match exactly on: " + ", ".join(mismatches)
        )

    induced = sum(
        assessment.outcome is RetrievalChallengeOutcome.INDUCED_SHADOWING
        for assessment in assessments
    )
    resilient = sum(
        assessment.outcome is RetrievalChallengeOutcome.RESILIENT
        for assessment in assessments
    )
    baseline_insufficient = sum(
        assessment.outcome is RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
        for assessment in assessments
    )
    eligible = induced + resilient
    return RetrievalChallengeSummary(
        policy=first.policy,
        retriever_id=first.retriever_id,
        retriever_version=first.retriever_version,
        top_k=first.top_k,
        total_cases=len(assessments),
        eligible_cases=eligible,
        baseline_insufficient_cases=baseline_insufficient,
        induced_shadowing_cases=induced,
        resilient_cases=resilient,
        induced_shadowing_rate=None if eligible == 0 else induced / eligible,
    )
