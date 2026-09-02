"""Provider-independent retrieval audit and runtime observation contracts."""

from palintrace.retrieval.base import (
    RetrievalError,
    RetrievalInputError,
    RetrievalObservationError,
    Retriever,
    retriever_identity,
    run_retrieval_audit,
    validate_retrieval_audit_request,
)
from palintrace.retrieval.challenge import (
    PairedRetrievalChallengeAssessment,
    RetrievalChallengeInputError,
    RetrievalChallengeOutcome,
    assess_paired_retrieval_challenge,
)
from palintrace.retrieval.models import (
    RetrievalAuditRequest,
    RetrievalHit,
    RetrievalObservation,
    RetrievalResponse,
    RetrievalUsage,
)
from palintrace.retrieval.policy import (
    RetrievalSufficiencyAssessment,
    RetrievalSufficiencyPolicy,
    assess_retrieval_sufficiency,
)

__all__ = [
    "RetrievalAuditRequest",
    "RetrievalChallengeInputError",
    "RetrievalChallengeOutcome",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalInputError",
    "RetrievalObservation",
    "RetrievalObservationError",
    "RetrievalResponse",
    "RetrievalSufficiencyAssessment",
    "RetrievalSufficiencyPolicy",
    "RetrievalUsage",
    "Retriever",
    "PairedRetrievalChallengeAssessment",
    "assess_paired_retrieval_challenge",
    "assess_retrieval_sufficiency",
    "retriever_identity",
    "run_retrieval_audit",
    "validate_retrieval_audit_request",
]
