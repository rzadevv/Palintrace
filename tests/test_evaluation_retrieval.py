import hashlib
import inspect

import pytest
from pydantic import ValidationError

from memlint.evaluation import (
    EvaluationInputError,
    RetrievalChallengeSummary,
    summarize_retrieval_challenges,
)
from memlint.retrieval import (
    PairedRetrievalChallengeAssessment,
    RetrievalChallengeOutcome,
    RetrievalHit,
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
    RetrievalUsage,
    assess_paired_retrieval_challenge,
)


def _observation(
    *,
    request_id: str,
    case_number: int,
    target_present: bool,
    policy_target: str | None = None,
    retriever_id: str = "controlled-retriever",
    retriever_version: str = "revision-1",
    top_k: int = 5,
) -> RetrievalObservation:
    target_id = policy_target or f"target-{case_number}"
    query = f"Controlled retrieval query {case_number}"
    return RetrievalObservation(
        request_id=request_id,
        query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        expected_memory_ids=(target_id,),
        top_k=top_k,
        retriever_id=retriever_id,
        retriever_version=retriever_version,
        hits=(RetrievalHit(memory_id=target_id, rank=1),) if target_present else (),
        usage=RetrievalUsage(retrieval_calls=1, candidate_count=int(target_present)),
    )


def _assessment(
    outcome: RetrievalChallengeOutcome,
    *,
    case_number: int,
    policy: RetrievalSufficiencyPolicy = RetrievalSufficiencyPolicy.ALL_EXPECTED,
    retriever_id: str = "controlled-retriever",
    retriever_version: str = "revision-1",
    top_k: int = 5,
) -> PairedRetrievalChallengeAssessment:
    baseline_present = outcome is not RetrievalChallengeOutcome.BASELINE_INSUFFICIENT
    mutated_present = outcome is RetrievalChallengeOutcome.RESILIENT
    baseline = _observation(
        request_id=f"baseline-{case_number}",
        case_number=case_number,
        target_present=baseline_present,
        retriever_id=retriever_id,
        retriever_version=retriever_version,
        top_k=top_k,
    )
    mutated = _observation(
        request_id=f"mutated-{case_number}",
        case_number=case_number,
        target_present=mutated_present,
        retriever_id=retriever_id,
        retriever_version=retriever_version,
        top_k=top_k,
    )
    return assess_paired_retrieval_challenge(
        baseline,
        mutated,
        policy=policy,
        case_id=f"case-{case_number}",
    )


def test_retrieval_summary_uses_only_paired_assessments() -> None:
    signature = inspect.signature(summarize_retrieval_challenges)
    assert tuple(signature.parameters) == ("assessments",)
    assert "manifest" not in signature.parameters
    assert "gold_label" not in signature.parameters
    assert "retrieval_probe" not in signature.parameters


def test_required_nine_case_cross_check_uses_baseline_eligible_denominator() -> None:
    assessments = tuple(
        [
            *(
                _assessment(
                    RetrievalChallengeOutcome.INDUCED_SHADOWING,
                    case_number=index,
                )
                for index in range(1, 3)
            ),
            *(
                _assessment(
                    RetrievalChallengeOutcome.RESILIENT,
                    case_number=index,
                )
                for index in range(3, 6)
            ),
            *(
                _assessment(
                    RetrievalChallengeOutcome.BASELINE_INSUFFICIENT,
                    case_number=index,
                )
                for index in range(6, 10)
            ),
        ]
    )

    summary = summarize_retrieval_challenges(assessments)

    assert summary.total_cases == 9
    assert summary.eligible_cases == 5
    assert summary.baseline_insufficient_cases == 4
    assert summary.induced_shadowing_cases == 2
    assert summary.resilient_cases == 3
    assert summary.induced_shadowing_rate == 2 / 5
    assert summary.policy is RetrievalSufficiencyPolicy.ALL_EXPECTED
    assert summary.retriever_id == "controlled-retriever"
    assert summary.retriever_version == "revision-1"
    assert summary.top_k == 5


def test_zero_eligible_cases_have_explicit_null_rate() -> None:
    summary = summarize_retrieval_challenges(
        tuple(
            _assessment(
                RetrievalChallengeOutcome.BASELINE_INSUFFICIENT,
                case_number=index,
            )
            for index in range(1, 4)
        )
    )

    assert summary.total_cases == 3
    assert summary.eligible_cases == 0
    assert summary.baseline_insufficient_cases == 3
    assert summary.induced_shadowing_rate is None
    assert "NaN" not in summary.to_json()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("policy", RetrievalSufficiencyPolicy.ANY_EXPECTED),
        ("retriever_id", "different-retriever"),
        ("retriever_version", "revision-2"),
        ("top_k", 10),
    ],
)
def test_mixed_retrieval_conditions_are_rejected(
    field: str,
    replacement: object,
) -> None:
    common = _assessment(
        RetrievalChallengeOutcome.INDUCED_SHADOWING,
        case_number=1,
    )
    options: dict[str, object] = {
        "policy": RetrievalSufficiencyPolicy.ALL_EXPECTED,
        "retriever_id": "controlled-retriever",
        "retriever_version": "revision-1",
        "top_k": 5,
    }
    options[field] = replacement
    changed = _assessment(
        RetrievalChallengeOutcome.RESILIENT,
        case_number=2,
        policy=options["policy"],  # type: ignore[arg-type]
        retriever_id=options["retriever_id"],  # type: ignore[arg-type]
        retriever_version=options["retriever_version"],  # type: ignore[arg-type]
        top_k=options["top_k"],  # type: ignore[arg-type]
    )

    with pytest.raises(EvaluationInputError, match=field):
        summarize_retrieval_challenges((common, changed))


def test_different_queries_targets_and_case_ids_may_share_one_summary() -> None:
    first = _assessment(
        RetrievalChallengeOutcome.INDUCED_SHADOWING,
        case_number=1,
    )
    second = _assessment(
        RetrievalChallengeOutcome.RESILIENT,
        case_number=2,
    )
    assert first.query_sha256 != second.query_sha256
    assert first.expected_memory_ids != second.expected_memory_ids
    assert first.case_id != second.case_id

    summary = summarize_retrieval_challenges((first, second))

    assert summary.total_cases == 2
    assert summary.eligible_cases == 2
    assert summary.induced_shadowing_rate == 0.5


def test_retrieval_summary_requires_nonempty_typed_assessments() -> None:
    with pytest.raises(EvaluationInputError, match="nonempty"):
        summarize_retrieval_challenges(())
    with pytest.raises(EvaluationInputError, match="only PairedRetrieval"):
        summarize_retrieval_challenges((object(),))  # type: ignore[arg-type]


def test_retrieval_summary_is_self_validating_and_has_no_accuracy_claim() -> None:
    summary = summarize_retrieval_challenges(
        (
            _assessment(
                RetrievalChallengeOutcome.INDUCED_SHADOWING,
                case_number=1,
            ),
            _assessment(
                RetrievalChallengeOutcome.RESILIENT,
                case_number=2,
            ),
        )
    )
    invalid_counts = summary.model_dump()
    invalid_counts["eligible_cases"] = 1
    invalid_rate = summary.model_dump()
    invalid_rate["induced_shadowing_rate"] = 0.9

    with pytest.raises(ValidationError):
        RetrievalChallengeSummary.model_validate(invalid_counts)
    with pytest.raises(ValidationError):
        RetrievalChallengeSummary.model_validate(invalid_rate)

    assert set(RetrievalChallengeSummary.model_fields).isdisjoint(
        {"accuracy", "recall", "precision", "f1", "checker_performance"}
    )


@pytest.mark.parametrize(
    "invalid_rate",
    [float("nan"), float("inf"), -0.01, 1.01, 1, True, "0.5", None],
)
def test_nonempty_eligible_rate_is_strict_finite_and_required(
    invalid_rate: object,
) -> None:
    payload = summarize_retrieval_challenges(
        (
            _assessment(
                RetrievalChallengeOutcome.INDUCED_SHADOWING,
                case_number=1,
            ),
        )
    ).model_dump()
    payload["induced_shadowing_rate"] = invalid_rate

    with pytest.raises(ValidationError):
        RetrievalChallengeSummary.model_validate(payload)


def test_retrieval_summary_serialization_is_deterministic_and_query_free() -> None:
    assessment = _assessment(
        RetrievalChallengeOutcome.INDUCED_SHADOWING,
        case_number=17,
    )
    summary = summarize_retrieval_challenges((assessment,))

    first = summary.to_json()
    second = summarize_retrieval_challenges((assessment,)).to_json()
    assert first == second
    assert RetrievalChallengeSummary.model_validate_json(first) == summary
    assert "Controlled retrieval query" not in first
    assert "query" not in RetrievalChallengeSummary.model_fields
    assert "expected_memory_ids" not in RetrievalChallengeSummary.model_fields
