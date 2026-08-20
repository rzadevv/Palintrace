"""Provider-independent semantic contracts and transcript evidence resolution."""

from memlint.semantics.base import SemanticJudge, semantic_judge_identity
from memlint.semantics.evidence import resolve_declared_evidence
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
    "EvidenceIssueKind",
    "EvidenceResolution",
    "EvidenceResolutionIssue",
    "EvidenceSegment",
    "SemanticJudge",
    "SemanticJudgment",
    "SemanticRelation",
    "SemanticUsage",
    "resolve_declared_evidence",
    "semantic_judge_identity",
]
