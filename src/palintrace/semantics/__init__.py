"""Provider-independent semantic contracts and transcript evidence resolution."""

from palintrace.semantics.base import SemanticJudge, semantic_judge_identity
from palintrace.semantics.composition import (
    PRIMARY_EVIDENCE_COMPOSITION_STYLE,
    ComposedEvidence,
    EvidenceCompositionStyle,
    SemanticCompositionError,
    compose_evidence,
)
from palintrace.semantics.evidence import resolve_declared_evidence
from palintrace.semantics.local_nli import (
    LocalNLISemanticJudge,
    SemanticDependencyError,
    SemanticInputError,
    SemanticInputTooLongError,
    SemanticJudgeError,
    SemanticModelConfigError,
)
from palintrace.semantics.models import (
    EvidenceIssueKind,
    EvidenceResolution,
    EvidenceResolutionIssue,
    EvidenceSegment,
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
)

__all__ = [
    "ComposedEvidence",
    "EvidenceCompositionStyle",
    "EvidenceIssueKind",
    "EvidenceResolution",
    "EvidenceResolutionIssue",
    "EvidenceSegment",
    "LocalNLISemanticJudge",
    "PRIMARY_EVIDENCE_COMPOSITION_STYLE",
    "SemanticCompositionError",
    "SemanticDependencyError",
    "SemanticInputError",
    "SemanticInputTooLongError",
    "SemanticJudge",
    "SemanticJudgeError",
    "SemanticJudgment",
    "SemanticModelConfigError",
    "SemanticRelation",
    "SemanticUsage",
    "compose_evidence",
    "resolve_declared_evidence",
    "semantic_judge_identity",
]
