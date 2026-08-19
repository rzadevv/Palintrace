import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from memlint.checkers import (
    Checker,
    CheckerInputError,
    CheckerResult,
    PrincipalBoundaryRule,
    PrivacyScopeViolationChecker,
    ScopeDimension,
    ScopeIsolationPolicy,
    load_scope_policy,
)
from memlint.cli import main
from memlint.models import (
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    TranscriptSet,
)
from memlint.mutations import MutationRequest, mutate
from memlint.serialization import load_store
from memlint.taxonomy import DefectClass


def _memory(
    memory_id: str,
    *,
    user_id: str | None = "user-a",
    agent_id: str | None = "agent-a",
    session_id: str | None = "session-1",
    content: str = "User prefers Python.",
    **changes: object,
) -> NormalizedMemory:
    payload: dict[str, object] = {
        "id": memory_id,
        "content": content,
        "created_at": datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 8, 2, 10, 0, tzinfo=UTC),
        "source_refs": (SourceRef(transcript_id="t1", turn_idx=0, span=(0, 8)),),
        "provenance_status": ProvenanceStatus.DECLARED,
        "scope": {
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
        },
        "active": True,
        "supersedes": (),
        "embedding": (0.1, 0.2),
        "raw": {"backend_note": memory_id},
    }
    payload.update(changes)
    return NormalizedMemory.model_validate(payload)


def _replica(
    memory: NormalizedMemory,
    memory_id: str,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> NormalizedMemory:
    scope_updates: dict[str, str | None] = {}
    if user_id is not None:
        scope_updates["user_id"] = user_id
    if agent_id is not None:
        scope_updates["agent_id"] = agent_id
    return memory.model_copy(
        update={
            "id": memory_id,
            "scope": memory.scope.model_copy(update=scope_updates),
            "raw": {"backend_note": f"replica-{memory_id}"},
        }
    )


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(adapter="test", memories=memories)


def _rule(
    dimension: ScopeDimension,
    authoritative: str,
    *destinations: str,
) -> PrincipalBoundaryRule:
    return PrincipalBoundaryRule(
        dimension=dimension,
        authoritative_source_principal=authoritative,
        prohibited_destination_principals=destinations,
    )


def _policy(
    dimension: ScopeDimension,
    authoritative: str,
    *destinations: str,
) -> ScopeIsolationPolicy:
    return ScopeIsolationPolicy(rules=(_rule(dimension, authoritative, *destinations),))


def _expected_replica_digest(memory: NormalizedMemory, dimension: str) -> str:
    payload = memory.semantic_dict()
    payload.pop("id")
    scope = payload["scope"]
    assert isinstance(scope, dict)
    scope.pop(dimension)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_user_policy_emits_destination_only_structural_finding() -> None:
    authoritative = _memory("authoritative")
    destination = _replica(authoritative, "destination", user_id="user-b")
    checker: Checker = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    )

    result = checker.check(_store(authoritative, destination))

    assert checker.checker_id == "privacy_scope_violation"
    assert checker.checker_version == "1.0"
    assert checker.defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION
    assert finding.memory_ids == ("destination",)
    assert finding.confidence == 1.0
    assert len(finding.evidence) == 1
    evidence = finding.evidence[0]
    assert evidence.kind == "prohibited_exact_replica"
    assert evidence.model_dump(mode="json")["data"] == {
        "authoritative_source_memory_id": "authoritative",
        "scope_dimension": "user_id",
        "authoritative_source_principal": "user-a",
        "destination_principal": "user-b",
        "replica_sha256": _expected_replica_digest(authoritative, "user_id"),
    }
    assert authoritative.content not in result.to_json()
    assert result.cost.model_calls == 0
    assert result.cost.input_tokens == 0
    assert result.cost.output_tokens == 0
    assert result.stats.memories_scanned == 2
    assert result.stats.findings_emitted == 1
    assert result.stats.details == {
        "policy_rules_scanned": 1,
        "authoritative_candidates": 1,
        "destination_candidates": 1,
        "exact_replica_matches": 1,
    }


def test_agent_policy_emits_prohibited_agent_record() -> None:
    authoritative = _memory("agent-source")
    destination = _replica(authoritative, "agent-destination", agent_id="agent-b")
    result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.AGENT_ID, "agent-a", "agent-b")
    ).check(_store(authoritative, destination))

    assert tuple(finding.memory_ids for finding in result.findings) == (
        ("agent-destination",),
    )
    assert result.findings[0].evidence[0].data["scope_dimension"] == "agent_id"


def test_policy_direction_selects_target_for_same_observable_store() -> None:
    record_a = _memory("record-a")
    record_b = _replica(record_a, "record-b", user_id="user-b")
    store = _store(record_a, record_b)

    a_authoritative = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    ).check(store)
    b_authoritative = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-b", "user-a")
    ).check(store)

    assert tuple(finding.memory_ids for finding in a_authoritative.findings) == (("record-b",),)
    assert tuple(finding.memory_ids for finding in b_authoritative.findings) == (("record-a",),)


def test_multiple_authoritative_matches_aggregate_for_one_destination() -> None:
    first = _memory("source-1")
    second = first.model_copy(update={"id": "source-2", "raw": {"other": True}})
    destination = _replica(first, "destination", user_id="user-b")

    result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    ).check(_store(first, destination, second))

    assert tuple(finding.memory_ids for finding in result.findings) == (("destination",),)
    assert tuple(
        item.data["authoritative_source_memory_id"]
        for item in result.findings[0].evidence
    ) == ("source-1", "source-2")
    assert result.stats.details["exact_replica_matches"] == 2


def test_multiple_prohibited_destinations_get_separate_findings() -> None:
    authoritative = _memory("source")
    destination_b = _replica(authoritative, "destination-b", user_id="user-b")
    destination_c = _replica(authoritative, "destination-c", user_id="user-c")

    result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-c", "user-b")
    ).check(_store(destination_c, authoritative, destination_b))

    assert tuple(finding.memory_ids for finding in result.findings) == (
        ("destination-b",),
        ("destination-c",),
    )


def test_store_policy_and_destination_order_do_not_affect_output() -> None:
    source = _memory("source")
    destination_b = _replica(source, "destination-b", user_id="user-b")
    destination_c = _replica(source, "destination-c", user_id="user-c")
    irrelevant_rule = _rule(ScopeDimension.AGENT_ID, "agent-x", "agent-y")
    user_rule_first = _rule(ScopeDimension.USER_ID, "user-a", "user-c", "user-b")
    user_rule_second = _rule(ScopeDimension.USER_ID, "user-a", "user-b", "user-c")
    first_policy = ScopeIsolationPolicy(rules=(user_rule_first, irrelevant_rule))
    second_policy = ScopeIsolationPolicy(rules=(irrelevant_rule, user_rule_second))

    first = PrivacyScopeViolationChecker(first_policy).check(
        _store(source, destination_b, destination_c)
    )
    second = PrivacyScopeViolationChecker(second_policy).check(
        _store(destination_c, destination_b, source), transcripts=TranscriptSet()
    )

    assert first_policy.to_json() == second_policy.to_json()
    assert first.to_json() == second.to_json()


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("content", "User prefers Rust."),
        ("created_at", datetime(2026, 8, 3, 10, 0, tzinfo=UTC)),
        ("updated_at", datetime(2026, 8, 4, 10, 0, tzinfo=UTC)),
        ("source_refs", (SourceRef(transcript_id="other", turn_idx=0),)),
        ("active", False),
        ("supersedes", ("older",)),
        ("embedding", (0.1, 0.3)),
    ],
)
def test_other_portable_fields_must_match_exactly(change: str, value: object) -> None:
    source = _memory("source")
    destination = _replica(source, "destination", user_id="user-b").model_copy(
        update={change: value}
    )

    result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    ).check(_store(source, destination))

    assert result.findings == ()


def test_same_content_with_different_provenance_is_not_a_replica() -> None:
    source = _memory("source")
    destination = _memory(
        "destination",
        user_id="user-b",
        source_refs=(),
        provenance_status=ProvenanceStatus.UNAVAILABLE,
    )

    result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    ).check(_store(source, destination))

    assert result.findings == ()


@pytest.mark.parametrize(
    "content",
    [
        "user prefers python.",
        "User prefers Python. ",
        "Python is the user's preferred language.",
    ],
    ids=("case", "whitespace", "paraphrase"),
)
def test_content_is_not_normalized_or_compared_semantically(content: str) -> None:
    source = _memory("source")
    destination = _replica(source, "destination", user_id="user-b").model_copy(
        update={"content": content}
    )

    result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    ).check(_store(source, destination))

    assert result.findings == ()


def test_nonapplicable_unknown_and_same_principals_are_skipped() -> None:
    policy = _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    checker = PrivacyScopeViolationChecker(policy)
    same_principal = _memory("same-principal")
    unknown_source = _memory("unknown-source", user_id=None)
    destination = _replica(unknown_source, "destination", user_id="user-b")
    authoritative = _memory("authoritative")
    unknown_destination = authoritative.model_copy(
        update={
            "id": "unknown-destination",
            "scope": authoritative.scope.model_copy(update={"user_id": None}),
        }
    )

    assert checker.check(_store(same_principal, _memory("same-principal-2"))).findings == ()
    assert checker.check(_store(unknown_source, destination)).findings == ()
    assert checker.check(_store(authoritative, unknown_destination)).findings == ()
    assert PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-c", "user-d")
    ).check(
        _store(authoritative, _replica(authoritative, "user-b", user_id="user-b"))
    ).findings == ()


def test_unknown_agent_principals_are_skipped() -> None:
    unknown_source = _memory("source", agent_id=None)
    destination = _replica(unknown_source, "destination", agent_id="agent-b")
    source = _memory("known-source")
    unknown_destination = source.model_copy(
        update={
            "id": "unknown-destination",
            "scope": source.scope.model_copy(update={"agent_id": None}),
        }
    )
    checker = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.AGENT_ID, "agent-a", "agent-b")
    )

    assert checker.check(_store(unknown_source, destination)).findings == ()
    assert checker.check(_store(source, unknown_destination)).findings == ()


def test_nonconfigured_scope_dimensions_must_match() -> None:
    source = _memory("source")
    user_destination = _replica(source, "user-destination", user_id="user-b").model_copy(
        update={
            "scope": source.scope.model_copy(
                update={"user_id": "user-b", "agent_id": "agent-b"}
            )
        }
    )
    agent_destination = _replica(
        source, "agent-destination", agent_id="agent-b"
    ).model_copy(
        update={
            "scope": source.scope.model_copy(
                update={"user_id": "user-b", "agent_id": "agent-b"}
            )
        }
    )

    user_result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.USER_ID, "user-a", "user-b")
    ).check(_store(source, user_destination))
    agent_result = PrivacyScopeViolationChecker(
        _policy(ScopeDimension.AGENT_ID, "agent-a", "agent-b")
    ).check(_store(source, agent_destination))

    assert user_result.findings == ()
    assert agent_result.findings == ()


def test_policy_is_required_by_checker_constructor() -> None:
    with pytest.raises(TypeError):
        PrivacyScopeViolationChecker()  # type: ignore[call-arg]


def test_policy_is_frozen_canonical_and_contains_only_boundary_configuration() -> None:
    first = ScopeIsolationPolicy(
        rules=(
            _rule(ScopeDimension.USER_ID, "user-a", "user-c"),
            _rule(ScopeDimension.AGENT_ID, "agent-a", "agent-b"),
            _rule(ScopeDimension.USER_ID, "user-a", "user-b"),
        )
    )
    second = ScopeIsolationPolicy(
        rules=(
            _rule(ScopeDimension.USER_ID, "user-a", "user-b", "user-c"),
            _rule(ScopeDimension.AGENT_ID, "agent-a", "agent-b"),
        )
    )

    assert first.to_json() == second.to_json()
    assert tuple(rule.dimension for rule in first.rules) == (
        ScopeDimension.AGENT_ID,
        ScopeDimension.USER_ID,
    )
    assert first.rules[1].prohibited_destination_principals == ("user-b", "user-c")
    policy_json = first.to_json()
    for forbidden in (
        "mutation_id",
        "memory_id",
        "gold_label",
        "target_memory_id",
        "created_memory_id",
    ):
        assert forbidden not in policy_json
    with pytest.raises(ValidationError):
        first.rules = ()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ScopeIsolationPolicy(
            schema_version="9.9", rules=(_rule(ScopeDimension.USER_ID, "a", "b"),)
        ),
        lambda: ScopeIsolationPolicy(rules=()),
        lambda: _rule(ScopeDimension.USER_ID, " ", "b"),
        lambda: _rule(ScopeDimension.USER_ID, "a"),
        lambda: _rule(ScopeDimension.USER_ID, "a", " "),
        lambda: _rule(ScopeDimension.USER_ID, "a", "b", "b"),
        lambda: _rule(ScopeDimension.USER_ID, "a", "a"),
        lambda: PrincipalBoundaryRule.model_validate(
            {
                "dimension": ScopeDimension.USER_ID,
                "authoritative_source_principal": "a",
                "prohibited_destination_principals": ("b",),
                "unexpected": True,
            }
        ),
    ],
    ids=(
        "schema",
        "rules",
        "blank-authoritative",
        "empty-destinations",
        "blank-destination",
        "duplicate-destination",
        "authoritative-prohibited",
        "extra-field",
    ),
)
def test_policy_rejects_invalid_structures(factory: Callable[[], object]) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "invalid scope policy"),
        ('{"schema_version":"9.9","rules":[]}', "schema_version"),
        ('{"schema_version":"0.1","rules":[]}', "at least one rule"),
        (
            '{"schema_version":"0.1","rules":[{"dimension":"session_id",'
            '"authoritative_source_principal":"a",'
            '"prohibited_destination_principals":["b"]}]}',
            "Input should be",
        ),
    ],
)
def test_policy_loader_reports_invalid_json_and_policy(
    text: str, message: str, tmp_path: Path
) -> None:
    path = tmp_path / "policy.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(CheckerInputError, match=message):
        load_scope_policy(path)


def test_policy_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CheckerInputError, match="could not read scope policy"):
        load_scope_policy(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("subtype", "dimension", "authoritative", "destination", "destination_field"),
    [
        ("cross_user_copy", ScopeDimension.USER_ID, "user-a", "user-b", "destination_user_id"),
        (
            "cross_agent_copy",
            ScopeDimension.AGENT_ID,
            "agent-a",
            "agent-b",
            "destination_agent_id",
        ),
    ],
)
def test_checker_matches_scope_mutation_gold_without_receiving_manifest(
    subtype: str,
    dimension: ScopeDimension,
    authoritative: str,
    destination: str,
    destination_field: str,
) -> None:
    request_data: dict[str, object] = {
        "defect_class": DefectClass.PRIVACY_SCOPE_VIOLATION,
        "subtype": subtype,
        "target_memory_id": "preference-python",
        destination_field: destination,
        "seed": 42,
    }
    mutation_result = mutate(
        load_store("examples/mutation-store.json"),
        MutationRequest.model_validate(request_data),
    )
    policy = _policy(dimension, authoritative, destination)

    checker_result = PrivacyScopeViolationChecker(policy).check(mutation_result.mutated_store)
    manifest = mutation_result.manifest

    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == manifest.gold_label.memory_ids


def test_scope_audit_cli_supports_policy_direction_and_ignores_transcripts(
    tmp_path: Path,
) -> None:
    record_a = _memory("record-a")
    record_b = _replica(record_a, "record-b", user_id="user-b")
    store_path = tmp_path / "store.json"
    forward_policy_path = tmp_path / "forward-policy.json"
    reverse_policy_path = tmp_path / "reverse-policy.json"
    forward_output = tmp_path / "forward.json"
    with_transcripts_output = tmp_path / "with-transcripts.json"
    reverse_output = tmp_path / "reverse.json"
    _store(record_a, record_b).to_json(store_path)
    _policy(ScopeDimension.USER_ID, "user-a", "user-b").to_json(forward_policy_path)
    _policy(ScopeDimension.USER_ID, "user-b", "user-a").to_json(reverse_policy_path)

    common = [
        "audit",
        "--store",
        str(store_path),
        "--checker",
        "privacy_scope_violation",
    ]
    assert (
        main(
            [
                *common,
                "--scope-policy",
                str(forward_policy_path),
                "--output",
                str(forward_output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                *common,
                "--transcripts",
                "examples/mutation-transcripts.json",
                "--scope-policy",
                str(forward_policy_path),
                "--output",
                str(with_transcripts_output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                *common,
                "--scope-policy",
                str(reverse_policy_path),
                "--output",
                str(reverse_output),
            ]
        )
        == 0
    )

    forward = CheckerResult.model_validate_json(forward_output.read_text(encoding="utf-8"))
    reverse = CheckerResult.model_validate_json(reverse_output.read_text(encoding="utf-8"))
    assert tuple(finding.memory_ids for finding in forward.findings) == (("record-b",),)
    assert tuple(finding.memory_ids for finding in reverse.findings) == (("record-a",),)
    assert forward_output.read_bytes() == with_transcripts_output.read_bytes()


def test_scope_audit_cli_handles_no_applicable_rule_and_requires_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    no_match_policy = tmp_path / "no-match.json"
    output = tmp_path / "output.json"
    _policy(ScopeDimension.USER_ID, "user-c", "user-d").to_json(no_match_policy)
    common = [
        "audit",
        "--store",
        "examples/mutation-store.json",
        "--checker",
        "privacy_scope_violation",
    ]

    assert main([*common, "--scope-policy", str(no_match_policy), "--output", str(output)]) == 0
    assert CheckerResult.model_validate_json(output.read_text(encoding="utf-8")).findings == ()
    with pytest.raises(SystemExit, match="2"):
        main(common)
    assert "privacy_scope_violation checker requires --scope-policy" in capsys.readouterr().err


def test_scope_policy_is_rejected_for_other_checkers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--checker",
                "stale_active",
                "--scope-policy",
                "examples/scope-policy.json",
                "--output",
                str(tmp_path / "output.json"),
            ]
        )

    assert "--scope-policy is only valid for privacy_scope_violation" in capsys.readouterr().err
