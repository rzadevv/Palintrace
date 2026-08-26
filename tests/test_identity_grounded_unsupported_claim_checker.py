from __future__ import annotations

import hashlib

import pytest

import memlint.checkers.unsupported_claim_identity_grounded as candidate_module
from memlint.checkers import Checker, CheckerError, CheckerInputError
from memlint.checkers.unsupported_claim_identity_grounded import (
    IdentityGroundedUnsupportedClaimChecker,
)
from memlint.models import (
    MemoryScope,
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from memlint.semantics import (
    SemanticInputTooLongError,
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
    SpeakerIdentityBinding,
    SpeakerIdentityBindings,
)
from memlint.taxonomy import DefectClass


class _FakeJudge:
    judge_id = "test-nli:identity-grounded"
    judge_version = "revision-1"

    def __init__(
        self,
        responses: dict[tuple[str, str], SemanticJudgment | Exception] | None = None,
        *,
        default: SemanticJudgment | Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        response = self.responses.get((premise, hypothesis), self.default)
        if response is None:
            raise AssertionError("no fake semantic response supplied for input pair")
        if isinstance(response, Exception):
            raise response
        return response


class _FailIfCalledJudge:
    judge_id = "test-nli:must-not-run"
    judge_version = "revision-1"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls += 1
        raise AssertionError(f"semantic judge must not run: {premise!r}, {hypothesis!r}")


def _judgment(
    relation: SemanticRelation,
    *,
    score: float = 0.8,
    model_calls: int = 1,
    input_tokens: int = 9,
    output_tokens: int = 0,
) -> SemanticJudgment:
    return SemanticJudgment(
        relation=relation,
        score=score,
        usage=SemanticUsage(
            model_calls=model_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _memory(
    memory_id: str,
    content: str,
    *source_refs: SourceRef,
    provenance_status: ProvenanceStatus = ProvenanceStatus.DECLARED,
    scope: MemoryScope | None = None,
    raw: dict[str, str] | None = None,
) -> NormalizedMemory:
    return NormalizedMemory(
        id=memory_id,
        content=content,
        source_refs=source_refs,
        provenance_status=provenance_status,
        scope=scope or MemoryScope(),
        active=True,
        raw=raw or {},
    )


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(adapter="test", memories=memories)


def _transcript(
    transcript_id: str,
    *turns: TranscriptTurn,
) -> Transcript:
    return Transcript(id=transcript_id, turns=turns)


def _single_turn_transcripts(
    evidence_text: str,
    *,
    transcript_id: str = "t1",
    role: str = "user",
) -> TranscriptSet:
    return TranscriptSet(
        transcripts=(
            _transcript(
                transcript_id,
                TranscriptTurn(index=0, role=role, content=evidence_text),
            ),
        )
    )


def _bindings(
    *values: tuple[str, int, str],
) -> SpeakerIdentityBindings:
    return SpeakerIdentityBindings(
        bindings=tuple(
            SpeakerIdentityBinding(
                transcript_id=transcript_id,
                turn_idx=turn_idx,
                speaker_label=speaker_label,
            )
            for transcript_id, turn_idx, speaker_label in values
        )
    )


def _checker(
    judge: _FakeJudge | _FailIfCalledJudge,
    *bindings: tuple[str, int, str],
) -> IdentityGroundedUnsupportedClaimChecker:
    return IdentityGroundedUnsupportedClaimChecker(
        judge=judge,
        speaker_bindings=_bindings(*bindings),
    )


def test_checker_has_separate_candidate_identity_and_implements_protocol() -> None:
    checker: Checker = _checker(
        _FakeJudge(default=_judgment(SemanticRelation.ENTAILMENT)),
        ("t1", 0, "Lina"),
    )
    assert checker.checker_id == "unsupported_claim_identity_grounded"
    assert checker.checker_version == "0.1"
    assert checker.defect_class is DefectClass.UNSUPPORTED_CLAIM


def test_candidate_is_not_public_default_or_cli_integrated() -> None:
    import memlint.checkers as checkers
    from memlint.cli import CHECKER_NAMES

    assert "IdentityGroundedUnsupportedClaimChecker" not in checkers.__all__
    assert not hasattr(checkers, "IdentityGroundedUnsupportedClaimChecker")
    assert "unsupported_claim_identity_grounded" not in CHECKER_NAMES


def test_constructor_requires_explicit_binding_contract() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.ENTAILMENT))
    with pytest.raises(TypeError, match="speaker_bindings must be"):
        IdentityGroundedUnsupportedClaimChecker(
            judge=judge,
            speaker_bindings=None,  # type: ignore[arg-type]
        )


def test_missing_transcript_set_raises_candidate_input_error() -> None:
    checker = _checker(_FailIfCalledJudge(), ("t1", 0, "Lina"))
    with pytest.raises(
        CheckerInputError,
        match="^unsupported_claim_identity_grounded checker requires a TranscriptSet$",
    ):
        checker.check(_store())


def test_resolved_clean_entailment_uses_exact_grounded_premise() -> None:
    evidence_text = "I prefer tea."
    hypothesis = "Lina prefers tea."
    grounded = "The speaker is Lina.\nI prefer tea."
    judge = _FakeJudge(default=_judgment(SemanticRelation.ENTAILMENT, input_tokens=12))
    result = _checker(judge, ("t1", 0, "Lina")).check(
        _store(_memory("m1", hypothesis, SourceRef(transcript_id="t1", turn_idx=0))),
        transcripts=_single_turn_transcripts(evidence_text),
    )

    assert judge.calls == [(grounded, hypothesis)]
    assert result.findings == ()
    assert result.stats.details["assessed_memories"] == 1
    assert result.stats.details["entailment_judgments"] == 1
    assert result.cost.model_calls == 1
    assert result.cost.input_tokens == 12


def test_resolved_neutral_emits_speaker_grounded_finding() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL, score=0.73))
    result = _checker(judge, ("t1", 0, "Lina")).check(
        _store(
            _memory(
                "m1",
                "Lina prefers coffee.",
                SourceRef(transcript_id="t1", turn_idx=0),
            )
        ),
        transcripts=_single_turn_transcripts("I prefer tea."),
    )

    assert len(result.findings) == 1
    evidence = result.findings[0].evidence[0]
    assert evidence.kind == "speaker_grounded_declared_evidence_not_entailing"
    assert evidence.data["semantic_relation"] == "neutral"
    assert evidence.data["composition_style"] == "plain"
    assert evidence.data["identity_grounding"] == "explicit_turn_binding_v0.1"
    assert result.stats.details["neutral_judgments"] == 1


def test_resolved_contradiction_emits_speaker_grounded_finding() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.CONTRADICTION))
    result = _checker(judge, ("t1", 0, "Lina")).check(
        _store(
            _memory(
                "m1",
                "Lina dislikes tea.",
                SourceRef(transcript_id="t1", turn_idx=0),
            )
        ),
        transcripts=_single_turn_transcripts("I prefer tea."),
    )

    assert len(result.findings) == 1
    assert result.findings[0].evidence[0].data["semantic_relation"] == "contradiction"
    assert result.stats.details["contradiction_judgments"] == 1


def test_missing_binding_abstains_without_judge_or_finding() -> None:
    judge = _FailIfCalledJudge()
    result = _checker(judge).check(
        _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0))),
        transcripts=_single_turn_transcripts("Valid evidence."),
    )

    assert judge.calls == 0
    assert result.findings == ()
    assert result.cost.model_calls == 0
    assert result.stats.details["skipped_identity_unavailable"] == 1


def test_transcript_level_reference_abstains_without_plain_fallback() -> None:
    judge = _FailIfCalledJudge()
    result = _checker(judge, ("t1", 0, "Lina")).check(
        _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1"))),
        transcripts=_single_turn_transcripts("Valid transcript-level evidence."),
    )

    assert judge.calls == 0
    assert result.findings == ()
    assert result.cost.model_calls == 0
    assert result.stats.details["skipped_identity_unavailable"] == 1


def test_conflicting_bindings_abstain_without_plain_fallback() -> None:
    judge = _FailIfCalledJudge()
    result = _checker(
        judge,
        ("t1", 0, "Alice"),
        ("t2", 0, "Bob"),
    ).check(
        _store(
            _memory(
                "m1",
                "Claim.",
                SourceRef(transcript_id="t1", turn_idx=0),
                SourceRef(transcript_id="t2", turn_idx=0),
            )
        ),
        transcripts=TranscriptSet(
            transcripts=(
                _transcript("t1", TranscriptTurn(index=0, role="user", content="first")),
                _transcript("t2", TranscriptTurn(index=0, role="user", content="second")),
            )
        ),
    )

    assert judge.calls == 0
    assert result.findings == ()
    assert result.cost.model_calls == 0
    assert result.stats.details["skipped_identity_conflict"] == 1


def test_same_speaker_multi_source_uses_one_prefix_and_canonical_plain_order() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = _checker(
        judge,
        ("t2", 0, "Alice"),
        ("t1", 1, "Alice"),
        ("t1", 0, "Alice"),
    ).check(
        _store(
            _memory(
                "m1",
                "Stored claim.",
                SourceRef(transcript_id="t2", turn_idx=0),
                SourceRef(transcript_id="t1", turn_idx=1),
                SourceRef(transcript_id="t1", turn_idx=0),
            )
        ),
        transcripts=TranscriptSet(
            transcripts=(
                _transcript(
                    "t2",
                    TranscriptTurn(index=0, role="assistant", content="third"),
                ),
                _transcript(
                    "t1",
                    TranscriptTurn(index=0, role="user", content="first"),
                    TranscriptTurn(index=1, role="user", content="second"),
                ),
            )
        ),
    )

    expected = "The speaker is Alice.\nfirst\nsecond\nthird"
    assert judge.calls == [(expected, "Stored claim.")]
    assert judge.calls[0][0].count("The speaker is Alice.") == 1
    evidence = result.findings[0].evidence[0]
    assert evidence.data["identity_source_turn_count"] == 3
    assert evidence.data["unique_segment_count"] == 3


def test_evidence_resolution_issue_precedes_identity_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_identity_resolution(*_args: object) -> None:
        raise AssertionError("identity resolution must not run")

    monkeypatch.setattr(candidate_module, "resolve_speaker_identity", fail_identity_resolution)
    judge = _FailIfCalledJudge()
    result = _checker(judge).check(
        _store(
            _memory(
                "m1",
                "Claim.",
                SourceRef(transcript_id="missing", turn_idx=0),
            )
        ),
        transcripts=TranscriptSet(),
    )

    assert judge.calls == 0
    assert result.stats.details["skipped_resolution_issues"] == 1


def test_non_declared_memory_precedes_identity_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_identity_resolution(*_args: object) -> None:
        raise AssertionError("identity resolution must not run")

    monkeypatch.setattr(candidate_module, "resolve_speaker_identity", fail_identity_resolution)
    judge = _FailIfCalledJudge()
    result = _checker(judge).check(
        _store(
            _memory(
                "m1",
                "Claim.",
                provenance_status=ProvenanceStatus.UNAVAILABLE,
            )
        ),
        transcripts=TranscriptSet(),
    )

    assert judge.calls == 0
    assert result.stats.details["skipped_non_declared_provenance"] == 1
    assert result.stats.details["declared_memories"] == 0


def test_input_too_long_abstains_without_model_cost() -> None:
    too_long = SemanticInputTooLongError(observed_tokens=700, maximum_tokens=512)
    judge = _FakeJudge(default=too_long)
    result = _checker(judge, ("t1", 0, "Lina")).check(
        _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0))),
        transcripts=_single_turn_transcripts("Oversized evidence."),
    )

    assert judge.calls == [("The speaker is Lina.\nOversized evidence.", "Claim.")]
    assert result.findings == ()
    assert result.stats.details["skipped_input_too_long"] == 1
    assert result.stats.details["assessed_memories"] == 0
    assert result.cost.model_dump() == {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_unexpected_judge_failure_is_wrapped_with_only_memory_context() -> None:
    transcript_text = "PRIVATE TRANSCRIPT CONTENT"
    memory_text = "PRIVATE MEMORY CONTENT"
    judge = _FakeJudge(default=RuntimeError(f"failed on {transcript_text} and {memory_text}"))

    with pytest.raises(CheckerError) as captured:
        _checker(judge, ("t1", 0, "Lina")).check(
            _store(
                _memory(
                    "safe-memory-id",
                    memory_text,
                    SourceRef(transcript_id="t1", turn_idx=0),
                )
            ),
            transcripts=_single_turn_transcripts(transcript_text),
        )

    message = str(captured.value)
    assert "safe-memory-id" in message
    assert transcript_text not in message
    assert memory_text not in message
    assert captured.value.__cause__ is None


def test_repeated_identical_runs_have_deterministic_result_and_finding_id() -> None:
    store = _store(
        _memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0))
    )
    transcripts = _single_turn_transcripts("Evidence.")
    response = _judgment(SemanticRelation.NEUTRAL, score=0.71)

    first = _checker(_FakeJudge(default=response), ("t1", 0, "Lina")).check(
        store,
        transcripts=transcripts,
    )
    second = _checker(_FakeJudge(default=response), ("t1", 0, "Lina")).check(
        store,
        transcripts=transcripts,
    )

    assert first.to_json() == second.to_json()
    assert first.findings[0].finding_id == second.findings[0].finding_id


def test_result_privacy_excludes_speaker_transcript_memory_and_raw_text() -> None:
    speaker_label = "DO_NOT_LEAK_MIREYA"
    transcript_text = "PRIVATE TRANSCRIPT STRING 3e24"
    memory_text = "PRIVATE MEMORY STRING 4f91"
    raw_text = "PRIVATE RAW STRING 8d12"
    grounded = f"The speaker is {speaker_label}.\n{transcript_text}"
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = _checker(judge, ("transcript-public-id", 0, speaker_label)).check(
        _store(
            _memory(
                "memory-public-id",
                memory_text,
                SourceRef(transcript_id="transcript-public-id", turn_idx=0),
                raw={"private": raw_text},
            )
        ),
        transcripts=_single_turn_transcripts(
            transcript_text,
            transcript_id="transcript-public-id",
        ),
    )
    serialized = result.to_json()

    assert speaker_label not in serialized
    assert transcript_text not in serialized
    assert memory_text not in serialized
    assert raw_text not in serialized
    evidence = result.findings[0].evidence[0]
    assert evidence.data["premise_sha256"] == hashlib.sha256(
        grounded.encode("utf-8")
    ).hexdigest()
    assert evidence.data["hypothesis_sha256"] == hashlib.sha256(
        memory_text.encode("utf-8")
    ).hexdigest()
    assert evidence.data["premise_sha256"] in serialized
    assert evidence.data["hypothesis_sha256"] in serialized


def test_grounding_is_independent_of_claim_scope_and_raw_fields() -> None:
    source_ref = SourceRef(transcript_id="t1", turn_idx=0)
    first = _memory(
        "m1",
        "Alice prefers tea.",
        source_ref,
        scope=MemoryScope(user_id="Bob", agent_id="Mallory"),
        raw={"speaker_name": "Not Alice"},
    )
    second = _memory(
        "m2",
        "Bob prefers coffee.",
        source_ref,
        scope=MemoryScope(user_id="Different"),
        raw={"speaker_name": "Different"},
    )
    judge = _FakeJudge(default=_judgment(SemanticRelation.ENTAILMENT))
    result = _checker(judge, ("t1", 0, "Explicit Speaker")).check(
        _store(first, second),
        transcripts=_single_turn_transcripts("I stated this."),
    )

    assert result.findings == ()
    assert judge.calls == [
        ("The speaker is Explicit Speaker.\nI stated this.", first.content),
        ("The speaker is Explicit Speaker.\nI stated this.", second.content),
    ]


def test_declared_memory_paths_form_exact_observable_partition() -> None:
    too_long = SemanticInputTooLongError(observed_tokens=700, maximum_tokens=512)
    responses = {
        ("The speaker is Lina.\nlong evidence", "Long claim."): too_long,
        ("The speaker is Lina.\nentailing evidence", "Entailed claim."): _judgment(
            SemanticRelation.ENTAILMENT,
            input_tokens=5,
        ),
        ("The speaker is Lina.\nneutral evidence", "Neutral claim."): _judgment(
            SemanticRelation.NEUTRAL,
            input_tokens=7,
        ),
        (
            "The speaker is Lina.\ncontradicting evidence",
            "Contradicted claim.",
        ): _judgment(
            SemanticRelation.CONTRADICTION,
            input_tokens=11,
        ),
    }
    judge = _FakeJudge(responses=responses)
    checker = _checker(
        judge,
        ("t-conflict-a", 0, "Alice"),
        ("t-conflict-b", 0, "Bob"),
        ("t-long", 0, "Lina"),
        ("t-entail", 0, "Lina"),
        ("t-neutral", 0, "Lina"),
        ("t-contradict", 0, "Lina"),
    )
    store = _store(
        _memory(
            "01-non-declared",
            "No provenance.",
            provenance_status=ProvenanceStatus.UNAVAILABLE,
        ),
        _memory(
            "02-resolution-issue",
            "Broken claim.",
            SourceRef(transcript_id="missing", turn_idx=0),
        ),
        _memory("03-no-evidence", "Empty claim.", SourceRef(transcript_id="t-empty")),
        _memory(
            "04-unavailable",
            "Unavailable claim.",
            SourceRef(transcript_id="t-unavailable", turn_idx=0),
        ),
        _memory(
            "05-conflict",
            "Conflict claim.",
            SourceRef(transcript_id="t-conflict-a", turn_idx=0),
            SourceRef(transcript_id="t-conflict-b", turn_idx=0),
        ),
        _memory(
            "06-too-long",
            "Long claim.",
            SourceRef(transcript_id="t-long", turn_idx=0),
        ),
        _memory(
            "07-entailment",
            "Entailed claim.",
            SourceRef(transcript_id="t-entail", turn_idx=0),
        ),
        _memory(
            "08-neutral",
            "Neutral claim.",
            SourceRef(transcript_id="t-neutral", turn_idx=0),
        ),
        _memory(
            "09-contradiction",
            "Contradicted claim.",
            SourceRef(transcript_id="t-contradict", turn_idx=0),
        ),
    )
    transcripts = TranscriptSet(
        transcripts=(
            _transcript("t-empty"),
            _transcript(
                "t-unavailable",
                TranscriptTurn(index=0, role="user", content="unavailable evidence"),
            ),
            _transcript(
                "t-conflict-a",
                TranscriptTurn(index=0, role="user", content="conflict first"),
            ),
            _transcript(
                "t-conflict-b",
                TranscriptTurn(index=0, role="user", content="conflict second"),
            ),
            _transcript(
                "t-long",
                TranscriptTurn(index=0, role="user", content="long evidence"),
            ),
            _transcript(
                "t-entail",
                TranscriptTurn(index=0, role="user", content="entailing evidence"),
            ),
            _transcript(
                "t-neutral",
                TranscriptTurn(index=0, role="user", content="neutral evidence"),
            ),
            _transcript(
                "t-contradict",
                TranscriptTurn(index=0, role="user", content="contradicting evidence"),
            ),
        )
    )

    result = checker.check(store, transcripts=transcripts)

    assert result.stats.memories_scanned == 9
    assert result.stats.findings_emitted == 2
    assert result.cost.model_dump() == {
        "model_calls": 3,
        "input_tokens": 23,
        "output_tokens": 0,
    }
    assert len(judge.calls) == 4
    details = dict(result.stats.details)
    assert details == {
        "declared_memories": 8,
        "assessed_memories": 3,
        "skipped_non_declared_provenance": 1,
        "skipped_resolution_issues": 1,
        "skipped_no_evidence": 1,
        "skipped_identity_unavailable": 1,
        "skipped_identity_conflict": 1,
        "skipped_input_too_long": 1,
        "entailment_judgments": 1,
        "neutral_judgments": 1,
        "contradiction_judgments": 1,
    }
    assert details["declared_memories"] == (
        details["assessed_memories"]
        + details["skipped_resolution_issues"]
        + details["skipped_no_evidence"]
        + details["skipped_identity_unavailable"]
        + details["skipped_identity_conflict"]
        + details["skipped_input_too_long"]
    )
    assert details["assessed_memories"] == (
        details["entailment_judgments"]
        + details["neutral_judgments"]
        + details["contradiction_judgments"]
    )
