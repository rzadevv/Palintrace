"""Provider-independent semantic contracts and transcript evidence resolution."""

from memlint.semantics.base import SemanticJudge, semantic_judge_identity
from memlint.semantics.composition import (
    PRIMARY_EVIDENCE_COMPOSITION_STYLE,
    ComposedEvidence,
    EvidenceCompositionStyle,
    SemanticCompositionError,
    compose_evidence,
)
from memlint.semantics.evidence import resolve_declared_evidence
from memlint.semantics.local_nli import (
    LocalNLISemanticJudge,
    SemanticDependencyError,
    SemanticInputError,
    SemanticInputTooLongError,
    SemanticJudgeError,
    SemanticModelConfigError,
)
from memlint.semantics.models import (
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
