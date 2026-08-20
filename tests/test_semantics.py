from collections.abc import Callable

import pytest
from pydantic import ValidationError

from memlint.checkers import OrphanedProvenanceChecker
from memlint.models import (
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from memlint.semantics import (
    EvidenceIssueKind,
    EvidenceResolution,
    EvidenceResolutionIssue,
    EvidenceSegment,
    SemanticJudge,
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
    resolve_declared_evidence,
)


def _turn(index: int, content: str, role: str = "user") -> TranscriptTurn:
    return TranscriptTurn(index=index, role=role, content=content)


def _transcripts(*transcripts: Transcript) -> TranscriptSet:
    return TranscriptSet(transcripts=transcripts)


def _transcript(
    transcript_id: str = "t1",
    *turns: TranscriptTurn,
) -> Transcript:
    return Transcript(id=transcript_id, turns=turns or (_turn(0, "Evidence text."),))


def _memory(*refs: SourceRef, content: str = "A stored claim.") -> NormalizedMemory:
    return NormalizedMemory(
        id="m1",
        content=content,
        source_refs=refs,
        provenance_status=ProvenanceStatus.DECLARED,
        raw={"backend": "ignored"},
    )


class FakeJudge:
    judge_id = "fake"
    judge_version = "1"

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        relation = (
            SemanticRelation.ENTAILMENT
            if hypothesis.casefold() in premise.casefold()
            else SemanticRelation.NEUTRAL
        )
        return SemanticJudgment(relation=relation, score=1.0)


def test_semantic_relation_has_exact_stable_values() -> None:
    assert tuple(SemanticRelation) == (
        SemanticRelation.ENTAILMENT,
        SemanticRelation.CONTRADICTION,
        SemanticRelation.NEUTRAL,
    )
    assert [relation.value for relation in SemanticRelation] == [
        "entailment",
        "contradiction",
        "neutral",
    ]


def test_fake_judge_satisfies_directional_contract() -> None:
    judge: SemanticJudge = FakeJudge()

    judgment = judge.judge(
        premise="The user prefers Python.",
        hypothesis="prefers Python",
    )

    assert judge.judge_id == "fake"
    assert judge.judge_version == "1"
    assert judgment == SemanticJudgment(
        relation=SemanticRelation.ENTAILMENT,
        score=1.0,
    )


@pytest.mark.parametrize("field", ["model_calls", "input_tokens", "output_tokens"])
def test_semantic_usage_rejects_negative_counts(field: str) -> None:
    with pytest.raises(ValidationError):
        SemanticUsage.model_validate({field: -1})


@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_semantic_judgment_rejects_invalid_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        SemanticJudgment(relation=SemanticRelation.NEUTRAL, score=score)


def test_semantic_judgment_rejects_invalid_relation_and_usage() -> None:
    with pytest.raises(ValidationError):
        SemanticJudgment.model_validate({"relation": "supported", "score": 0.5})
    with pytest.raises(ValidationError):
        SemanticJudgment.model_validate(
            {
                "relation": "neutral",
                "score": 0.5,
                "usage": {"model_calls": -1},
            }
        )


def test_semantic_models_are_frozen_and_forbid_extra_fields() -> None:
    judgment = SemanticJudgment(relation=SemanticRelation.NEUTRAL, score=0.5)
    with pytest.raises(ValidationError):
        judgment.score = 0.2
    with pytest.raises(ValidationError):
        SemanticUsage.model_validate({"unexpected": 1})
    with pytest.raises(ValidationError):
        SemanticJudgment.model_validate(
            {"relation": "neutral", "score": 0.5, "unexpected": True}
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EvidenceSegment(
            source_ref_index=-1,
            transcript_id="t1",
            turn_idx=0,
            role="user",
            text="text",
        ),
        lambda: EvidenceSegment(
            source_ref_index=0,
            transcript_id=" ",
            turn_idx=0,
            role="user",
            text="text",
        ),
        lambda: EvidenceSegment(
            source_ref_index=0,
            transcript_id="t1",
            turn_idx=0,
            role=" ",
            text="text",
        ),
        lambda: EvidenceSegment.model_validate(
            {
                "source_ref_index": 0,
                "transcript_id": "t1",
                "turn_idx": 0,
                "role": "user",
                "text": "text",
                "unexpected": True,
            }
        ),
    ],
    ids=("negative-index", "blank-transcript", "blank-role", "extra"),
)
def test_evidence_segment_validation(factory: Callable[[], object]) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_evidence_resolution_models_validate_issue_shapes_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EvidenceResolutionIssue(
            kind=EvidenceIssueKind.MISSING_TURN,
            source_ref_index=0,
            transcript_id="t1",
        )
    with pytest.raises(ValidationError):
        EvidenceResolutionIssue(
            kind=EvidenceIssueKind.INVALID_SPAN,
            source_ref_index=0,
            transcript_id="t1",
            turn_idx=0,
            span=(0, 2),
            turn_length=2,
        )
    with pytest.raises(ValidationError):
        EvidenceResolution.model_validate({"unexpected": True})


def test_resolves_exact_character_span_with_python_character_indexing() -> None:
    memory = _memory(SourceRef(transcript_id="t1", turn_idx=0, span=(1, 3)))
    transcripts = _transcripts(_transcript("t1", _turn(0, "A🙂BC", role="user")))

    resolution = resolve_declared_evidence(memory, transcripts)

    assert resolution.issues == ()
    assert resolution.segments == (
        EvidenceSegment(
            source_ref_index=0,
            transcript_id="t1",
            turn_idx=0,
            role="user",
            span=(1, 3),
            text="🙂B",
        ),
    )


def test_resolves_whole_turn_without_concatenation() -> None:
    memory = _memory(SourceRef(transcript_id="t1", turn_idx=1))
    transcripts = _transcripts(
        _transcript("t1", _turn(0, "First."), _turn(1, "Second.", role="assistant"))
    )

    resolution = resolve_declared_evidence(memory, transcripts)

    assert [(segment.turn_idx, segment.span, segment.text) for segment in resolution.segments] == [
        (1, None, "Second.")
    ]


def test_resolves_whole_transcript_as_one_segment_per_ordered_turn() -> None:
    memory = _memory(SourceRef(transcript_id="t1"))
    transcripts = _transcripts(
        _transcript(
            "t1",
            _turn(0, "First."),
            _turn(1, "Second.", role="assistant"),
        )
    )

    resolution = resolve_declared_evidence(memory, transcripts)

    assert [(segment.turn_idx, segment.role, segment.text) for segment in resolution.segments] == [
        (0, "user", "First."),
        (1, "assistant", "Second."),
    ]
    assert all(segment.span is None for segment in resolution.segments)


def test_whole_transcript_defensively_sorts_out_of_order_turns() -> None:
    memory = _memory(SourceRef(transcript_id="t1"))
    malformed_transcript = Transcript.model_construct(
        id="t1",
        turns=(_turn(2, "Third."), _turn(0, "First."), _turn(1, "Second.")),
        metadata={},
    )
    transcripts = TranscriptSet.model_construct(
        schema_version="0.1",
        transcripts=(malformed_transcript,),
    )

    resolution = resolve_declared_evidence(memory, transcripts)

    assert [segment.turn_idx for segment in resolution.segments] == [0, 1, 2]


def test_missing_transcript_returns_one_minimal_issue() -> None:
    memory = _memory(SourceRef(transcript_id="missing", turn_idx=9, span=(0, 3)))

    resolution = resolve_declared_evidence(memory, TranscriptSet())

    assert resolution.segments == ()
    assert resolution.issues == (
        EvidenceResolutionIssue(
            kind=EvidenceIssueKind.MISSING_TRANSCRIPT,
            source_ref_index=0,
            transcript_id="missing",
        ),
    )


def test_missing_turn_returns_issue_without_text_or_span() -> None:
    memory = _memory(SourceRef(transcript_id="t1", turn_idx=9, span=(0, 3)))

    resolution = resolve_declared_evidence(memory, _transcripts(_transcript("t1")))

    assert resolution.segments == ()
    assert resolution.issues == (
        EvidenceResolutionIssue(
            kind=EvidenceIssueKind.MISSING_TURN,
            source_ref_index=0,
            transcript_id="t1",
            turn_idx=9,
        ),
    )


def test_invalid_span_returns_coordinates_without_turn_text() -> None:
    memory = _memory(SourceRef(transcript_id="t1", turn_idx=0, span=(1, 8)))

    resolution = resolve_declared_evidence(
        memory,
        _transcripts(_transcript("t1", _turn(0, "short"))),
    )

    assert resolution.segments == ()
    assert resolution.issues == (
        EvidenceResolutionIssue(
            kind=EvidenceIssueKind.INVALID_SPAN,
            source_ref_index=0,
            transcript_id="t1",
            turn_idx=0,
            span=(1, 8),
            turn_length=5,
        ),
    )
    assert "short" not in resolution.to_json()


def test_span_ending_exactly_at_turn_length_is_valid() -> None:
    memory = _memory(SourceRef(transcript_id="t1", turn_idx=0, span=(1, 5)))

    resolution = resolve_declared_evidence(
        memory,
        _transcripts(_transcript("t1", _turn(0, "short"))),
    )

    assert resolution.issues == ()
    assert resolution.segments[0].text == "hort"


@pytest.mark.parametrize(
    "status",
    [ProvenanceStatus.UNAVAILABLE, ProvenanceStatus.KNOWN_ABSENT],
)
def test_non_declared_provenance_has_no_segments_or_issues(
    status: ProvenanceStatus,
) -> None:
    memory = NormalizedMemory(id="m1", content="Claim.", provenance_status=status)

    resolution = resolve_declared_evidence(memory, TranscriptSet())

    assert resolution == EvidenceResolution()


def test_multiple_and_duplicate_source_refs_are_preserved() -> None:
    duplicate = SourceRef(transcript_id="t1", turn_idx=0)
    memory = _memory(
        duplicate,
        SourceRef(transcript_id="t1", turn_idx=1, span=(0, 3)),
        duplicate,
    )
    transcripts = _transcripts(
        _transcript("t1", _turn(0, "First."), _turn(1, "Second."))
    )

    resolution = resolve_declared_evidence(memory, transcripts)

    assert len(resolution.segments) == 3
    assert [segment.source_ref_index for segment in resolution.segments] == [0, 2, 1]
    assert [segment.text for segment in resolution.segments] == ["First.", "First.", "Sec"]


def test_valid_and_broken_references_are_returned_separately() -> None:
    memory = _memory(
        SourceRef(transcript_id="t1", turn_idx=0),
        SourceRef(transcript_id="missing"),
        content="Semantics are not evaluated.",
    )

    resolution = resolve_declared_evidence(memory, _transcripts(_transcript("t1")))

    assert len(resolution.segments) == 1
    assert len(resolution.issues) == 1
    assert resolution.issues[0].kind is EvidenceIssueKind.MISSING_TRANSCRIPT


def test_resolution_serialization_is_canonical_and_deterministic() -> None:
    first_segment = EvidenceSegment(
        source_ref_index=1,
        transcript_id="t2",
        turn_idx=0,
        role="assistant",
        text="Second transcript.",
    )
    second_segment = EvidenceSegment(
        source_ref_index=0,
        transcript_id="t1",
        turn_idx=1,
        role="user",
        text="First transcript.",
    )
    first_issue = EvidenceResolutionIssue(
        kind=EvidenceIssueKind.MISSING_TRANSCRIPT,
        source_ref_index=3,
        transcript_id="z-missing",
    )
    second_issue = EvidenceResolutionIssue(
        kind=EvidenceIssueKind.MISSING_TURN,
        source_ref_index=2,
        transcript_id="a-transcript",
        turn_idx=9,
    )

    first = EvidenceResolution(
        segments=(first_segment, second_segment),
        issues=(first_issue, second_issue),
    )
    second = EvidenceResolution(
        segments=(second_segment, first_segment),
        issues=(second_issue, first_issue),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json() == first.to_json()


@pytest.mark.parametrize(
    ("source_ref", "transcripts", "expected_kind"),
    [
        (
            SourceRef(transcript_id="missing"),
            TranscriptSet(),
            EvidenceIssueKind.MISSING_TRANSCRIPT,
        ),
        (
            SourceRef(transcript_id="t1", turn_idx=9),
            _transcripts(_transcript("t1")),
            EvidenceIssueKind.MISSING_TURN,
        ),
        (
            SourceRef(transcript_id="t1", turn_idx=0, span=(0, 99)),
            _transcripts(_transcript("t1")),
            EvidenceIssueKind.INVALID_SPAN,
        ),
    ],
)
def test_resolution_issues_agree_with_orphaned_provenance_checker(
    source_ref: SourceRef,
    transcripts: TranscriptSet,
    expected_kind: EvidenceIssueKind,
) -> None:
    memory = _memory(source_ref)

    resolution = resolve_declared_evidence(memory, transcripts)
    checker_result = OrphanedProvenanceChecker().check(
        NormalizedStore(adapter="test", memories=(memory,)),
        transcripts=transcripts,
    )

    assert len(resolution.issues) == 1
    assert resolution.issues[0].kind is expected_kind
    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == (memory.id,)
    assert checker_result.findings[0].evidence[0].kind == expected_kind.value
