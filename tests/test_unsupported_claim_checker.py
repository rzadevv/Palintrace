from __future__ import annotations

import hashlib
import os

import pytest

from memlint.checkers import (
    Checker,
    CheckerError,
    CheckerInputError,
    CheckerResult,
    UnsupportedClaimChecker,
)
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
    EvidenceCompositionStyle,
    LocalNLISemanticJudge,
    SemanticInputTooLongError,
    SemanticJudgeError,
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
)
from memlint.taxonomy import DefectClass

MINILM_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
MINILM_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"


class _FakeJudge:
    def __init__(
        self,
        responses: dict[tuple[str, str], SemanticJudgment | Exception] | None = None,
        *,
        default: SemanticJudgment | Exception | None = None,
        judge_id: str = "test-nli:deterministic",
        judge_version: str = "revision-1",
    ) -> None:
        self.judge_id = judge_id
        self.judge_version = judge_version
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
    raw: dict[str, str] | None = None,
) -> NormalizedMemory:
    return NormalizedMemory(
        id=memory_id,
        content=content,
        source_refs=source_refs,
        provenance_status=provenance_status,
        active=True,
        raw=raw or {},
    )


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(adapter="test", memories=memories)


def _transcripts(
    *turns: TranscriptTurn,
    transcript_id: str = "t1",
) -> TranscriptSet:
    return TranscriptSet(
        transcripts=(Transcript(id=transcript_id, turns=turns),),
    )


def _single_turn_transcripts(content: str, *, role: str = "user") -> TranscriptSet:
    return _transcripts(TranscriptTurn(index=0, role=role, content=content))


def _single_memory_result(
    *,
    premise: str,
    hypothesis: str,
    relation: SemanticRelation,
    score: float = 0.8,
    composition_style: EvidenceCompositionStyle | None = None,
) -> tuple[_FakeJudge, CheckerResult]:
    judge = _FakeJudge(default=_judgment(relation, score=score))
    checker = (
        UnsupportedClaimChecker(judge)
        if composition_style is None
        else UnsupportedClaimChecker(judge, composition_style=composition_style)
    )
    result = checker.check(
        _store(
            _memory(
                "m1",
                hypothesis,
                SourceRef(transcript_id="t1", turn_idx=0),
            )
        ),
        transcripts=_single_turn_transcripts(premise),
    )
    return judge, result


def test_checker_implements_protocol_and_declares_identity() -> None:
    checker: Checker = UnsupportedClaimChecker(
        _FakeJudge(default=_judgment(SemanticRelation.ENTAILMENT))
    )

    assert checker.checker_id == "unsupported_claim"
    assert checker.checker_version == "1.0"
    assert checker.defect_class is DefectClass.UNSUPPORTED_CLAIM


def test_constructor_validates_and_captures_judge_identity_once() -> None:
    judge = _FakeJudge(
        default=_judgment(SemanticRelation.NEUTRAL),
        judge_id="test-nli:captured",
        judge_version="revision-original",
    )
    checker = UnsupportedClaimChecker(judge)
    judge.judge_id = "test-nli:changed"
    judge.judge_version = "revision-changed"

    result = checker.check(
        _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0))),
        transcripts=_single_turn_transcripts("Evidence."),
    )

    evidence = result.findings[0].evidence[0]
    assert evidence.data["judge_id"] == "test-nli:captured"
    assert evidence.data["judge_version"] == "revision-original"


def test_missing_transcript_set_raises_exact_input_error() -> None:
    checker = UnsupportedClaimChecker(
        _FakeJudge(default=_judgment(SemanticRelation.ENTAILMENT))
    )

    with pytest.raises(
        CheckerInputError,
        match="^unsupported_claim checker requires a TranscriptSet$",
    ):
        checker.check(_store())


def test_contradiction_emits_one_memory_finding_with_semantic_evidence() -> None:
    premise = "I live in Berlin."
    hypothesis = "The user lives in Munich."
    score = 0.91
    judge, result = _single_memory_result(
        premise=premise,
        hypothesis=hypothesis,
        relation=SemanticRelation.CONTRADICTION,
        score=score,
    )

    assert judge.calls == [(premise, hypothesis)]
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.memory_ids == ("m1",)
    assert finding.confidence == score
    assert finding.defect_class is DefectClass.UNSUPPORTED_CLAIM
    assert len(finding.evidence) == 1
    evidence = finding.evidence[0]
    assert evidence.kind == "declared_evidence_not_entailing"
    assert evidence.data["semantic_relation"] == "contradiction"
    assert evidence.data["composition_style"] == "plain"
    assert evidence.data["premise_sha256"] == hashlib.sha256(premise.encode()).hexdigest()
    assert evidence.data["hypothesis_sha256"] == hashlib.sha256(
        hypothesis.encode()
    ).hexdigest()
    assert evidence.data["source_coordinates"] == (
        {
            "transcript_id": "t1",
            "turn_idx": 0,
            "span": None,
            "role": "user",
        },
    )
    assert "score" not in evidence.data
    assert result.stats.details["contradiction_judgments"] == 1


def test_neutral_emits_one_unsupported_finding() -> None:
    _, result = _single_memory_result(
        premise="I may move to Cologne.",
        hypothesis="The user lives in Cologne.",
        relation=SemanticRelation.NEUTRAL,
        score=0.701,
    )

    assert len(result.findings) == 1
    assert result.findings[0].confidence == 0.701
    assert result.findings[0].evidence[0].data["semantic_relation"] == "neutral"
    assert result.stats.details["neutral_judgments"] == 1


def test_entailment_is_clean_but_records_cost_and_assessment() -> None:
    judge = _FakeJudge(
        default=_judgment(
            SemanticRelation.ENTAILMENT,
            score=0.88,
            model_calls=1,
            input_tokens=12,
            output_tokens=0,
        )
    )
    checker = UnsupportedClaimChecker(judge)
    result = checker.check(
        _store(
            _memory(
                "m1",
                "The user prefers Python.",
                SourceRef(transcript_id="t1", turn_idx=0),
            )
        ),
        transcripts=_single_turn_transcripts("I prefer Python."),
    )

    assert result.findings == ()
    assert result.cost.model_dump() == {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 0,
    }
    assert result.stats.details["assessed_memories"] == 1
    assert result.stats.details["entailment_judgments"] == 1


@pytest.mark.parametrize(
    "status",
    [ProvenanceStatus.UNAVAILABLE, ProvenanceStatus.KNOWN_ABSENT],
)
def test_non_declared_provenance_is_skipped_without_judgment(
    status: ProvenanceStatus,
) -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = UnsupportedClaimChecker(judge).check(
        _store(
            NormalizedMemory(
                id="m1",
                content="Unassessable claim.",
                provenance_status=status,
            )
        ),
        transcripts=TranscriptSet(),
    )

    assert result.findings == ()
    assert judge.calls == []
    assert result.stats.details["declared_memories"] == 0
    assert result.stats.details["skipped_non_declared_provenance"] == 1


@pytest.mark.parametrize(
    ("source_ref", "transcripts"),
    [
        (SourceRef(transcript_id="missing"), TranscriptSet()),
        (
            SourceRef(transcript_id="t1", turn_idx=4),
            _single_turn_transcripts("Existing evidence."),
        ),
        (
            SourceRef(transcript_id="t1", turn_idx=0, span=(0, 99)),
            _single_turn_transcripts("Short."),
        ),
    ],
    ids=("missing_transcript", "missing_turn", "invalid_span"),
)
def test_any_structural_resolution_issue_causes_abstention(
    source_ref: SourceRef,
    transcripts: TranscriptSet,
) -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = UnsupportedClaimChecker(judge).check(
        _store(_memory("m1", "Claim.", source_ref)),
        transcripts=transcripts,
    )

    assert result.findings == ()
    assert judge.calls == []
    assert result.stats.details["skipped_resolution_issues"] == 1
    assert result.cost.model_calls == 0


def test_mixed_valid_and_broken_references_abstain_completely() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = UnsupportedClaimChecker(judge).check(
        _store(
            _memory(
                "m1",
                "Claim.",
                SourceRef(transcript_id="t1", turn_idx=0),
                SourceRef(transcript_id="missing", turn_idx=0),
            )
        ),
        transcripts=_single_turn_transcripts("Valid partial evidence."),
    )

    assert result.findings == ()
    assert judge.calls == []
    assert result.stats.details["skipped_resolution_issues"] == 1


def test_existing_empty_transcript_is_unassessable_no_evidence() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = UnsupportedClaimChecker(judge).check(
        _store(_memory("m1", "Claim.", SourceRef(transcript_id="empty"))),
        transcripts=_transcripts(transcript_id="empty"),
    )

    assert result.findings == ()
    assert judge.calls == []
    assert result.stats.details["skipped_no_evidence"] == 1


def test_whitespace_only_composed_premise_is_unassessable_no_evidence() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = UnsupportedClaimChecker(judge).check(
        _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0))),
        transcripts=_single_turn_transcripts(" \t\n "),
    )

    assert result.findings == ()
    assert judge.calls == []
    assert result.stats.details["skipped_no_evidence"] == 1


def test_input_too_long_abstains_without_model_usage_and_continues() -> None:
    too_long = SemanticInputTooLongError(observed_tokens=700, maximum_tokens=512)
    judge = _FakeJudge(
        responses={
            ("Oversized evidence.", "First claim."): too_long,
            ("Supported evidence.", "Second claim."): _judgment(
                SemanticRelation.ENTAILMENT,
                input_tokens=6,
            ),
        }
    )
    store = _store(
        _memory("m1", "First claim.", SourceRef(transcript_id="t1", turn_idx=0)),
        _memory("m2", "Second claim.", SourceRef(transcript_id="t1", turn_idx=1)),
    )
    transcripts = _transcripts(
        TranscriptTurn(index=0, role="user", content="Oversized evidence."),
        TranscriptTurn(index=1, role="user", content="Supported evidence."),
    )

    result = UnsupportedClaimChecker(judge).check(store, transcripts=transcripts)

    assert result.findings == ()
    assert len(judge.calls) == 2
    assert result.stats.details["skipped_input_too_long"] == 1
    assert result.stats.details["assessed_memories"] == 1
    assert result.cost.model_calls == 1
    assert result.cost.input_tokens == 6


def test_generic_judge_error_is_wrapped_without_semantic_input_text() -> None:
    premise = "PRIVATE TRANSCRIPT CONTENT 8192"
    hypothesis = "PRIVATE MEMORY CLAIM 4096"
    judge = _FakeJudge(
        default=SemanticJudgeError(f"failed on {premise} versus {hypothesis}")
    )
    checker = UnsupportedClaimChecker(judge)

    with pytest.raises(CheckerError) as captured:
        checker.check(
            _store(
                _memory(
                    "safe-memory-id",
                    hypothesis,
                    SourceRef(transcript_id="t1", turn_idx=0),
                )
            ),
            transcripts=_single_turn_transcripts(premise),
        )

    message = str(captured.value)
    assert "safe-memory-id" in message
    assert premise not in message
    assert hypothesis not in message
    assert captured.value.__cause__ is None


def test_unexpected_runtime_error_is_not_silently_converted_to_result() -> None:
    judge = _FakeJudge(default=RuntimeError("backend failed"))

    with pytest.raises(CheckerError, match="semantic judgment failed"):
        UnsupportedClaimChecker(judge).check(
            _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0))),
            transcripts=_single_turn_transcripts("Evidence."),
        )


def test_default_plain_multi_evidence_uses_canonical_deduplicated_premise() -> None:
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    store = _store(
        _memory(
            "m1",
            "Exact hypothesis.",
            SourceRef(transcript_id="t2", turn_idx=0),
            SourceRef(transcript_id="t1", turn_idx=1),
            SourceRef(transcript_id="t1", turn_idx=0),
            SourceRef(transcript_id="t1", turn_idx=0),
        )
    )
    transcripts = TranscriptSet(
        transcripts=(
            Transcript(
                id="t2",
                turns=(TranscriptTurn(index=0, role="assistant", content="third"),),
            ),
            Transcript(
                id="t1",
                turns=(
                    TranscriptTurn(index=0, role="user", content="first"),
                    TranscriptTurn(index=1, role="user", content="second"),
                ),
            ),
        )
    )

    result = UnsupportedClaimChecker(judge).check(store, transcripts=transcripts)

    assert judge.calls == [("first\nsecond\nthird", "Exact hypothesis.")]
    evidence = result.findings[0].evidence[0]
    assert "segment_count" not in evidence.data
    assert evidence.data["unique_segment_count"] == 3
    assert evidence.data["source_coordinates"] == (
        {"transcript_id": "t1", "turn_idx": 0, "span": None, "role": "user"},
        {"transcript_id": "t1", "turn_idx": 1, "span": None, "role": "user"},
        {
            "transcript_id": "t2",
            "turn_idx": 0,
            "span": None,
            "role": "assistant",
        },
    )


def test_duplicate_source_ref_does_not_change_semantic_finding_identity() -> None:
    premise = "Canonical evidence."
    hypothesis = "Stored claim."
    source_ref = SourceRef(transcript_id="t1", turn_idx=0)
    single_declaration = _store(_memory("m1", hypothesis, source_ref))
    duplicate_declaration = _store(_memory("m1", hypothesis, source_ref, source_ref))
    transcripts = _single_turn_transcripts(premise)
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL, score=0.73))
    checker = UnsupportedClaimChecker(judge)

    single_result = checker.check(single_declaration, transcripts=transcripts)
    duplicate_result = checker.check(duplicate_declaration, transcripts=transcripts)

    assert judge.calls == [(premise, hypothesis), (premise, hypothesis)]
    single_finding = single_result.findings[0]
    duplicate_finding = duplicate_result.findings[0]
    single_evidence = single_finding.evidence[0].data
    duplicate_evidence = duplicate_finding.evidence[0].data
    assert single_evidence["premise_sha256"] == duplicate_evidence["premise_sha256"]
    assert single_evidence["hypothesis_sha256"] == duplicate_evidence["hypothesis_sha256"]
    assert single_evidence["source_coordinates"] == duplicate_evidence["source_coordinates"]
    assert len(single_evidence["source_coordinates"]) == 1
    assert single_evidence["unique_segment_count"] == 1
    assert duplicate_evidence["unique_segment_count"] == 1
    assert single_evidence["semantic_relation"] == "neutral"
    assert duplicate_evidence["semantic_relation"] == "neutral"
    assert "segment_count" not in single_evidence
    assert "segment_count" not in duplicate_evidence
    assert single_evidence == duplicate_evidence
    assert single_finding.finding_id == duplicate_finding.finding_id


def test_role_labeled_composition_override_changes_only_premise_and_evidence() -> None:
    judge, result = _single_memory_result(
        premise="Declared fact.",
        hypothesis="Stored claim.",
        relation=SemanticRelation.NEUTRAL,
        composition_style=EvidenceCompositionStyle.ROLE_LABELED,
    )

    assert judge.calls == [("user: Declared fact.", "Stored claim.")]
    evidence = result.findings[0].evidence[0]
    assert evidence.data["composition_style"] == "role_labeled"
    assert evidence.data["premise_sha256"] == hashlib.sha256(
        b"user: Declared fact."
    ).hexdigest()


def test_cost_aggregates_successful_usage_and_skips_contribute_zero() -> None:
    judge = _FakeJudge(
        responses={
            ("Evidence 1.", "Claim 1."): _judgment(
                SemanticRelation.ENTAILMENT,
                input_tokens=5,
            ),
            ("Evidence 2.", "Claim 2."): _judgment(
                SemanticRelation.NEUTRAL,
                input_tokens=7,
            ),
            ("Evidence 3.", "Claim 3."): _judgment(
                SemanticRelation.CONTRADICTION,
                input_tokens=11,
            ),
        }
    )
    store = _store(
        _memory("m1", "Claim 1.", SourceRef(transcript_id="t1", turn_idx=0)),
        _memory("m2", "Claim 2.", SourceRef(transcript_id="t1", turn_idx=1)),
        _memory("m3", "Claim 3.", SourceRef(transcript_id="t1", turn_idx=2)),
        NormalizedMemory(id="m4", content="No provenance."),
        _memory("m5", "Broken.", SourceRef(transcript_id="missing")),
    )
    transcripts = _transcripts(
        TranscriptTurn(index=0, role="user", content="Evidence 1."),
        TranscriptTurn(index=1, role="user", content="Evidence 2."),
        TranscriptTurn(index=2, role="user", content="Evidence 3."),
    )

    result = UnsupportedClaimChecker(judge).check(store, transcripts=transcripts)

    assert result.cost.model_dump() == {
        "model_calls": 3,
        "input_tokens": 23,
        "output_tokens": 0,
    }
    details = result.stats.details
    assert result.stats.memories_scanned == 5
    assert result.stats.findings_emitted == 2
    assert details["declared_memories"] == 4
    assert details["assessed_memories"] == 3
    assert details["skipped_non_declared_provenance"] == 1
    assert details["skipped_resolution_issues"] == 1
    assert details["skipped_no_evidence"] == 0
    assert details["skipped_input_too_long"] == 0
    assert details["entailment_judgments"] == 1
    assert details["neutral_judgments"] == 1
    assert details["contradiction_judgments"] == 1
    assert details["declared_memories"] == (
        details["assessed_memories"]
        + details["skipped_resolution_issues"]
        + details["skipped_no_evidence"]
        + details["skipped_input_too_long"]
    )
    assert details["assessed_memories"] == (
        details["entailment_judgments"]
        + details["neutral_judgments"]
        + details["contradiction_judgments"]
    )


def test_store_reordering_does_not_change_serialized_result() -> None:
    memories = (
        _memory("m2", "Claim 2.", SourceRef(transcript_id="t1", turn_idx=1)),
        _memory("m1", "Claim 1.", SourceRef(transcript_id="t1", turn_idx=0)),
    )
    transcripts = _transcripts(
        TranscriptTurn(index=0, role="user", content="Evidence 1."),
        TranscriptTurn(index=1, role="user", content="Evidence 2."),
    )
    response = _judgment(SemanticRelation.NEUTRAL, score=0.72)

    first = UnsupportedClaimChecker(_FakeJudge(default=response)).check(
        _store(*memories), transcripts=transcripts
    )
    second = UnsupportedClaimChecker(_FakeJudge(default=response)).check(
        _store(*reversed(memories)), transcripts=transcripts
    )

    assert first.to_json() == second.to_json()


def test_score_only_jitter_does_not_change_finding_id() -> None:
    _, first = _single_memory_result(
        premise="Related evidence.",
        hypothesis="Claim.",
        relation=SemanticRelation.NEUTRAL,
        score=0.701,
    )
    _, second = _single_memory_result(
        premise="Related evidence.",
        hypothesis="Claim.",
        relation=SemanticRelation.NEUTRAL,
        score=0.702,
    )

    assert first.findings[0].finding_id == second.findings[0].finding_id
    assert first.findings[0].confidence != second.findings[0].confidence


def test_relation_change_changes_finding_id() -> None:
    _, neutral = _single_memory_result(
        premise="Evidence.",
        hypothesis="Claim.",
        relation=SemanticRelation.NEUTRAL,
    )
    _, contradiction = _single_memory_result(
        premise="Evidence.",
        hypothesis="Claim.",
        relation=SemanticRelation.CONTRADICTION,
    )

    assert neutral.findings[0].finding_id != contradiction.findings[0].finding_id


def test_judge_version_change_changes_finding_id() -> None:
    store = _store(_memory("m1", "Claim.", SourceRef(transcript_id="t1", turn_idx=0)))
    transcripts = _single_turn_transcripts("Evidence.")
    response = _judgment(SemanticRelation.NEUTRAL)

    first = UnsupportedClaimChecker(
        _FakeJudge(default=response, judge_version="revision-a")
    ).check(store, transcripts=transcripts)
    second = UnsupportedClaimChecker(
        _FakeJudge(default=response, judge_version="revision-b")
    ).check(store, transcripts=transcripts)

    assert first.findings[0].finding_id != second.findings[0].finding_id


def test_checker_result_excludes_full_semantic_and_raw_text() -> None:
    premise = "PRIVATE TRANSCRIPT STRING db55e1b7"
    hypothesis = "PRIVATE MEMORY STRING 0ed37c8a"
    raw_text = "PRIVATE RAW BACKEND STRING 81c8c2df"
    judge = _FakeJudge(default=_judgment(SemanticRelation.NEUTRAL))
    result = UnsupportedClaimChecker(judge).check(
        _store(
            _memory(
                "memory-public-id",
                hypothesis,
                SourceRef(transcript_id="transcript-public-id", turn_idx=0),
                raw={"private": raw_text},
            )
        ),
        transcripts=_transcripts(
            TranscriptTurn(index=0, role="user", content=premise),
            transcript_id="transcript-public-id",
        ),
    )
    serialized = result.to_json()

    assert premise not in serialized
    assert hypothesis not in serialized
    assert raw_text not in serialized
    assert "memory-public-id" in serialized
    assert "transcript-public-id" in serialized


def test_part_two_unsupported_gold_unit_matches_without_checker_manifest_access() -> None:
    from memlint.mutations import MutationRequest, mutate
    from memlint.serialization import load_store, load_transcripts

    base_store = load_store("examples/mutation-store.json")
    transcripts = load_transcripts("examples/mutation-transcripts.json")
    mutation_result = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.UNSUPPORTED_CLAIM,
            target_memory_id="preference-python",
            replace_from="Python",
            replace_to="Rust",
            seed=42,
        ),
        transcripts,
    )
    judge = _FakeJudge(
        default=_judgment(SemanticRelation.ENTAILMENT),
        responses={
            ("I prefer Python.", "User prefers Rust."): _judgment(
                SemanticRelation.NEUTRAL,
                score=0.75,
            )
        },
    )

    checker_result = UnsupportedClaimChecker(judge).check(
        mutation_result.mutated_store,
        transcripts=transcripts,
    )
    manifest = mutation_result.manifest

    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == manifest.gold_label.memory_ids


def test_unsupported_claim_is_exposed_in_audit_cli() -> None:
    from memlint.cli import CHECKER_NAMES

    assert "unsupported_claim" in CHECKER_NAMES


@pytest.mark.skipif(
    os.getenv("MEMLINT_RUN_LOCAL_NLI") != "1",
    reason="set MEMLINT_RUN_LOCAL_NLI=1 for the pinned CPU MiniLM integration",
)
def test_pinned_minilm_simple_unsupported_checker_integration() -> None:
    judge = LocalNLISemanticJudge(
        model_id=MINILM_MODEL_ID,
        revision=MINILM_REVISION,
        device="cpu",
    )
    transcript_set = _single_turn_transcripts("I prefer Python.")
    source_ref = SourceRef(transcript_id="t1", turn_idx=0)
    supported = _store(_memory("preference", "User prefers Python.", source_ref))
    unsupported = _store(_memory("preference", "User prefers Rust.", source_ref))
    checker = UnsupportedClaimChecker(judge)

    supported_result = checker.check(supported, transcripts=transcript_set)
    unsupported_result = checker.check(unsupported, transcripts=transcript_set)

    assert supported_result.findings == ()
    assert supported_result.stats.details["entailment_judgments"] == 1
    assert len(unsupported_result.findings) == 1
    assert unsupported_result.findings[0].evidence[0].data[
        "semantic_relation"
    ] in {"neutral", "contradiction"}
