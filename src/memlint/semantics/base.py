"""Small provider-independent semantic judge protocol."""

from __future__ import annotations

from typing import Protocol

from memlint.semantics.models import SemanticJudgment


class SemanticJudge(Protocol):
    """Classify a hypothesis relative to a premise with a stable nonblank identity."""

    judge_id: str
    judge_version: str

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        """Return the directional semantic relation for evidence and claim text."""


def semantic_judge_identity(judge: SemanticJudge) -> tuple[str, str]:
    """Return a judge's declared identity after validating its runtime values."""

    judge_id = judge.judge_id
    judge_version = judge.judge_version
    if not isinstance(judge_id, str) or not judge_id.strip():
        raise ValueError("semantic judge_id must be a nonblank string")
    if not isinstance(judge_version, str) or not judge_version.strip():
        raise ValueError("semantic judge_version must be a nonblank string")
    return judge_id, judge_version
