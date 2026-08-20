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

