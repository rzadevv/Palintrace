from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import palintrace.preflight as preflight_module
from palintrace.adapters import adapter_capabilities
from palintrace.audit import SkipReason, run_aggregate_audit
from palintrace.checker_requirements import PUBLIC_CHECKER_IDS
from palintrace.checkers import (
    OrphanedProvenanceChecker,
    PrincipalBoundaryRule,
    PrivacyScopeViolationChecker,
    RedundancyBloatChecker,
    ScopeDimension,
    ScopeIsolationPolicy,
    StaleActiveChecker,
    UnsupportedClaimChecker,
)
from palintrace.models import (
    MemoryScope,
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    TranscriptSet,
)
from palintrace.preflight import (
    PREFLIGHT_REPORT_SCHEMA_VERSION,
    PreflightChecker,
    PreflightLimitation,
    PreflightReport,
    PreflightStatus,
    run_preflight,
)


def _memory(
    memory_id: str,
    *,
    declared: bool = True,
    user_id: str | None = "user-a",
    agent_id: str | None = None,
    session_id: str | None = None,
    active: bool | None = True,
) -> NormalizedMemory:
    source_refs = (SourceRef(transcript_id="t1", turn_idx=0),) if declared else ()
    return NormalizedMemory(
        id=memory_id,
        content=f"Memory {memory_id}.",
        source_refs=source_refs,
        provenance_status=(
            ProvenanceStatus.DECLARED if declared else ProvenanceStatus.UNAVAILABLE
        ),
        scope=MemoryScope(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        ),
        active=active,
    )


def _store(*memories: NormalizedMemory, adapter: str = "file") -> NormalizedStore:
    return NormalizedStore(adapter=adapter, memories=memories)


def _policy(*dimensions: ScopeDimension) -> ScopeIsolationPolicy:
    return ScopeIsolationPolicy(
        rules=tuple(
            PrincipalBoundaryRule(
                dimension=dimension,
                authoritative_source_principal=f"{dimension.value}-a",
                prohibited_destination_principals=(f"{dimension.value}-b",),
            )
            for dimension in dimensions
        )
    )


def _check(report: PreflightReport, checker_id: str) -> PreflightChecker:
    return next(item for item in report.checks if item.checker_id == checker_id)


def _ready(checker_id: str) -> PreflightChecker:
    return PreflightChecker(
        checker_id=checker_id,
        status=PreflightStatus.READY,
        total_memories=0,
        assessable_memories=0,
    )


def _valid_report() -> PreflightReport:
    return PreflightReport(
        adapter="file",
        adapter_capabilities=adapter_capabilities("file"),
        checks=tuple(_ready(checker_id) for checker_id in reversed(PUBLIC_CHECKER_IDS)),
    )


def test_preflight_schema_and_enum_values_are_exact() -> None:
    assert PREFLIGHT_REPORT_SCHEMA_VERSION == "0.1"
    assert tuple(PreflightStatus) == (
        PreflightStatus.READY,
        PreflightStatus.DEGRADED,
        PreflightStatus.BLOCKED,
    )
    assert tuple(PreflightLimitation) == (
        PreflightLimitation.NON_DECLARED_PROVENANCE,
        PreflightLimitation.UNSCOPED_MEMORIES,
        PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,
        PreflightLimitation.SUPERSESSION_UNSUPPORTED,
        PreflightLimitation.POLICY_SCOPE_UNAVAILABLE,
        PreflightLimitation.POLICY_SCOPE_UNSUPPORTED,
    )

    with pytest.raises(ValidationError, match="unsupported preflight report schema_version"):
        PreflightReport(
            schema_version="0.2",
            adapter="file",
            adapter_capabilities=adapter_capabilities("file"),
            checks=tuple(_ready(checker_id) for checker_id in PUBLIC_CHECKER_IDS),
        )


def test_preflight_models_are_frozen_and_forbid_extra_fields() -> None:
    checker = _ready("stale_active")
    report = _valid_report()

    with pytest.raises(ValidationError, match="frozen"):
        checker.status = PreflightStatus.BLOCKED
    with pytest.raises(ValidationError, match="frozen"):
        report.adapter = "other"
    with pytest.raises(ValidationError, match="extra"):
        PreflightChecker(
            checker_id="stale_active",
            status=PreflightStatus.READY,
            total_memories=0,
            assessable_memories=0,
            message="no",
        )
    with pytest.raises(ValidationError, match="extra"):
        PreflightReport.model_validate({**report.model_dump(), "generated_at": "now"})


def test_report_requires_complete_unique_public_checker_set_and_canonicalizes_order() -> None:
    assert tuple(item.checker_id for item in _valid_report().checks) == PUBLIC_CHECKER_IDS

    with pytest.raises(ValidationError, match="missing public checker IDs"):
        PreflightReport(
            adapter="file",
            adapter_capabilities=adapter_capabilities("file"),
            checks=tuple(_ready(checker_id) for checker_id in PUBLIC_CHECKER_IDS[:-1]),
        )
    with pytest.raises(ValidationError, match="checker IDs must be unique"):
        PreflightReport(
            adapter="file",
            adapter_capabilities=adapter_capabilities("file"),
            checks=(
                *tuple(_ready(checker_id) for checker_id in PUBLIC_CHECKER_IDS),
                _ready("stale_active"),
            ),
        )
    with pytest.raises(ValidationError, match="unknown public checker_id"):
        PreflightChecker(
            checker_id="unknown",
            status=PreflightStatus.READY,
            total_memories=0,
            assessable_memories=0,
        )


@pytest.mark.parametrize(
    "checker",
    [
        PreflightChecker(
            checker_id="stale_active",
            status=PreflightStatus.READY,
            total_memories=1,
            assessable_memories=1,
        ),
        PreflightChecker(
            checker_id="stale_active",
            status=PreflightStatus.DEGRADED,
            total_memories=1,
            assessable_memories=0,
            limitations=(PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,),
        ),
        PreflightChecker(
            checker_id="unsupported_claim",
            status=PreflightStatus.BLOCKED,
            total_memories=1,
            assessable_memories=0,
            blocked_reasons=(SkipReason.MISSING_TRANSCRIPTS,),
        ),
    ],
)
def test_valid_status_invariants(checker: PreflightChecker) -> None:
    assert isinstance(checker, PreflightChecker)


@pytest.mark.parametrize(
    "data",
    [
        {"status": "ready", "limitations": ["active_state_unavailable"]},
        {"status": "ready", "blocked_reasons": ["missing_transcripts"]},
        {"status": "degraded"},
        {
            "status": "degraded",
            "limitations": ["active_state_unavailable"],
            "blocked_reasons": ["missing_transcripts"],
        },
        {"status": "blocked"},
        {
            "status": "blocked",
            "blocked_reasons": ["missing_transcripts"],
            "limitations": ["active_state_unavailable"],
        },
        {
            "status": "blocked",
            "assessable_memories": 1,
            "blocked_reasons": ["missing_transcripts"],
        },
    ],
)
def test_invalid_status_invariants_are_rejected(data: dict[str, object]) -> None:
    values: dict[str, object] = {
        "checker_id": "stale_active",
        "total_memories": 1,
        "assessable_memories": 0,
        **data,
    }
    with pytest.raises(ValidationError):
        PreflightChecker.model_validate(values)


@pytest.mark.parametrize(
    ("total", "assessable"),
    [(-1, 0), (1, -1), (1, 2), (True, 0), (1, False)],
)
def test_count_validation(total: object, assessable: object) -> None:
    with pytest.raises(ValidationError):
        PreflightChecker(
            checker_id="stale_active",
            status=PreflightStatus.READY,
            total_memories=total,
            assessable_memories=assessable,
        )


def test_reason_and_limitation_sequences_are_unique_and_canonical() -> None:
    blocked = PreflightChecker(
        checker_id="unsupported_claim",
        status=PreflightStatus.BLOCKED,
        total_memories=1,
        assessable_memories=0,
        blocked_reasons=(
            SkipReason.MISSING_SEMANTIC_CONFIGURATION,
            SkipReason.MISSING_TRANSCRIPTS,
        ),
    )
    degraded = PreflightChecker(
        checker_id="stale_active",
        status=PreflightStatus.DEGRADED,
        total_memories=1,
        assessable_memories=0,
        limitations=(
            PreflightLimitation.SUPERSESSION_UNSUPPORTED,
            PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,
        ),
    )

    assert blocked.blocked_reasons == (
        SkipReason.MISSING_TRANSCRIPTS,
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )
    assert degraded.limitations == (
        PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,
        PreflightLimitation.SUPERSESSION_UNSUPPORTED,
    )
    with pytest.raises(ValidationError, match="blocked reasons must be unique"):
        PreflightChecker(
            checker_id="unsupported_claim",
            status=PreflightStatus.BLOCKED,
            total_memories=1,
            assessable_memories=0,
            blocked_reasons=(
                SkipReason.MISSING_TRANSCRIPTS,
                SkipReason.MISSING_TRANSCRIPTS,
            ),
        )
    with pytest.raises(ValidationError, match="limitations must be unique"):
        PreflightChecker(
            checker_id="stale_active",
            status=PreflightStatus.DEGRADED,
            total_memories=1,
            assessable_memories=0,
            limitations=(
                PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,
                PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,
            ),
        )


def test_report_serialization_is_deterministic_has_no_runtime_metadata_and_matches_file(
    tmp_path: Path,
) -> None:
    report = _valid_report()
    output = tmp_path / "preflight.json"

    first = report.to_json(output)
    second = report.to_json()
    payload = json.loads(first)

    assert first == second
    assert output.read_text(encoding="utf-8") == first
    assert first.endswith("\n")
    assert not {
        "timestamp",
        "generated_at",
        "duration",
        "hostname",
        "cwd",
        "clean",
        "passed",
    } & payload.keys()


@pytest.mark.parametrize("adapter", ["file", "mem0"])
def test_builtin_adapter_capability_contract_is_embedded(adapter: str) -> None:
    report = run_preflight(_store(adapter=adapter))

    assert report.adapter == adapter
    assert report.adapter_capabilities == adapter_capabilities(adapter)
    assert report.adapter_capabilities.schema_version == "0.1"


def test_unknown_adapter_has_no_guessed_capabilities_and_still_succeeds() -> None:
    report = run_preflight(
        _store(_memory("m1", agent_id="agent-a"), adapter="custom-exporter"),
        transcripts=TranscriptSet(),
        scope_policy=_policy(ScopeDimension.USER_ID, ScopeDimension.AGENT_ID),
        semantic_configured=True,
    )

    assert report.adapter == "custom-exporter"
    assert report.adapter_capabilities is None
    assert all(item.status is PreflightStatus.READY for item in report.checks)


def test_orphaned_provenance_ready_degraded_and_blocked() -> None:
    declared = _memory("declared")
    unavailable = _memory("unavailable", declared=False)

    ready = _check(
        run_preflight(_store(declared), transcripts=TranscriptSet()),
        "orphaned_provenance",
    )
    degraded = _check(
        run_preflight(_store(declared, unavailable), transcripts=TranscriptSet()),
        "orphaned_provenance",
    )
    blocked = _check(run_preflight(_store(declared)), "orphaned_provenance")

    assert (ready.status, ready.assessable_memories) == (PreflightStatus.READY, 1)
    assert degraded.status is PreflightStatus.DEGRADED
    assert degraded.assessable_memories == 1
    assert degraded.limitations == (PreflightLimitation.NON_DECLARED_PROVENANCE,)
    assert blocked.status is PreflightStatus.BLOCKED
    assert blocked.assessable_memories == 0
    assert blocked.blocked_reasons == (SkipReason.MISSING_TRANSCRIPTS,)
    assert blocked.limitations == ()


def test_unsupported_claim_ready_degraded_and_all_blocker_combinations() -> None:
    declared = _memory("declared")
    unavailable = _memory("unavailable", declared=False)
    ready = _check(
        run_preflight(
            _store(declared), transcripts=TranscriptSet(), semantic_configured=True
        ),
        "unsupported_claim",
    )
    degraded = _check(
        run_preflight(
            _store(declared, unavailable),
            transcripts=TranscriptSet(),
            semantic_configured=True,
        ),
        "unsupported_claim",
    )
    missing_semantic = _check(
        run_preflight(_store(declared), transcripts=TranscriptSet()),
        "unsupported_claim",
    )
    missing_both = _check(run_preflight(_store(declared)), "unsupported_claim")

    assert ready.status is PreflightStatus.READY
    assert degraded.limitations == (PreflightLimitation.NON_DECLARED_PROVENANCE,)
    assert missing_semantic.blocked_reasons == (
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )
    assert missing_both.blocked_reasons == (
        SkipReason.MISSING_TRANSCRIPTS,
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )


@pytest.mark.parametrize(
    ("memories", "status", "assessable"),
    [
        ((_memory("scoped"),), PreflightStatus.READY, 1),
        (
            (_memory("scoped"), _memory("unscoped", user_id=None)),
            PreflightStatus.DEGRADED,
            1,
        ),
        ((_memory("unscoped", user_id=None),), PreflightStatus.DEGRADED, 0),
    ],
)
def test_redundancy_assessability_uses_observable_scope(
    memories: tuple[NormalizedMemory, ...],
    status: PreflightStatus,
    assessable: int,
) -> None:
    check = _check(run_preflight(_store(*memories)), "redundancy_bloat")

    assert check.status is status
    assert check.assessable_memories == assessable
    assert check.limitations == (
        (PreflightLimitation.UNSCOPED_MEMORIES,)
        if status is PreflightStatus.DEGRADED
        else ()
    )


def test_stale_assessability_combines_record_and_static_limitations_canonically() -> None:
    ready = _check(run_preflight(_store(_memory("active"))), "stale_active")
    unavailable = _check(
        run_preflight(_store(_memory("unknown", active=None))), "stale_active"
    )
    unsupported = _check(
        run_preflight(
            _store(_memory("active"), _memory("unknown", active=None), adapter="graphiti")
        ),
        "stale_active",
    )

    assert ready.status is PreflightStatus.READY
    assert unavailable.limitations == (PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,)
    assert unsupported.assessable_memories == 1
    assert unsupported.limitations == (
        PreflightLimitation.ACTIVE_STATE_UNAVAILABLE,
        PreflightLimitation.SUPERSESSION_UNSUPPORTED,
    )


def test_privacy_assessability_uses_every_policy_dimension_and_missing_policy_blocks() -> None:
    store = _store(
        _memory("both", agent_id="agent-a"),
        _memory("user-only"),
        _memory("neither", user_id=None),
    )
    user_only = _check(
        run_preflight(store, scope_policy=_policy(ScopeDimension.USER_ID)),
        "privacy_scope_violation",
    )
    both = _check(
        run_preflight(
            store,
            scope_policy=_policy(ScopeDimension.USER_ID, ScopeDimension.AGENT_ID),
        ),
        "privacy_scope_violation",
    )
    blocked = _check(run_preflight(store), "privacy_scope_violation")

    assert user_only.assessable_memories == 2
    assert user_only.limitations == (PreflightLimitation.POLICY_SCOPE_UNAVAILABLE,)
    assert both.assessable_memories == 1
    assert both.limitations == (PreflightLimitation.POLICY_SCOPE_UNAVAILABLE,)
    assert blocked.status is PreflightStatus.BLOCKED
    assert blocked.blocked_reasons == (SkipReason.MISSING_SCOPE_POLICY,)


def test_conditional_policy_capability_does_not_degrade_observed_complete_scope() -> None:
    check = _check(
        run_preflight(
            _store(_memory("m1"), adapter="graphiti"),
            scope_policy=_policy(ScopeDimension.USER_ID),
        ),
        "privacy_scope_violation",
    )

    assert check.status is PreflightStatus.READY
    assert check.limitations == ()


def test_explicitly_unsupported_policy_capability_is_reported_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capabilities = adapter_capabilities("file").model_copy(
        update={"user_scope": "unsupported"}
    )
    monkeypatch.setattr(
        preflight_module, "adapter_capabilities", lambda _adapter: capabilities
    )

    check = _check(
        run_preflight(
            _store(_memory("first"), _memory("second")),
            scope_policy=_policy(ScopeDimension.USER_ID),
        ),
        "privacy_scope_violation",
    )

    assert check.status is PreflightStatus.DEGRADED
    assert check.assessable_memories == 2
    assert check.limitations == (PreflightLimitation.POLICY_SCOPE_UNSUPPORTED,)


@pytest.mark.parametrize("adapter", ["file", "graphiti"])
def test_empty_store_is_not_an_observed_coverage_limitation(adapter: str) -> None:
    report = run_preflight(
        _store(adapter=adapter),
        transcripts=TranscriptSet(),
        scope_policy=_policy(ScopeDimension.USER_ID),
        semantic_configured=True,
    )

    assert all(item.total_memories == item.assessable_memories == 0 for item in report.checks)
    expected_stale = (
        PreflightStatus.DEGRADED if adapter == "graphiti" else PreflightStatus.READY
    )
    assert _check(report, "stale_active").status is expected_stale
    assert all(
        item.status is PreflightStatus.READY
        for item in report.checks
        if item.checker_id != "stale_active"
    )


def test_external_blockers_match_aggregate_skip_reasons() -> None:
    store = _store(_memory("m1"), adapter="custom-exporter")
    aggregate = run_aggregate_audit(store)
    preflight = run_preflight(store)

    for skipped in aggregate.skipped:
        check = _check(preflight, skipped.checker_id)
        assert check.status is PreflightStatus.BLOCKED
        assert check.blocked_reasons == skipped.reasons


def test_run_preflight_never_executes_public_checkers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_check(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("checker execution attempted")

    for checker_type in (
        OrphanedProvenanceChecker,
        RedundancyBloatChecker,
        StaleActiveChecker,
        PrivacyScopeViolationChecker,
        UnsupportedClaimChecker,
    ):
        monkeypatch.setattr(checker_type, "check", fail_check)

    report = run_preflight(
        _store(_memory("m1")),
        transcripts=TranscriptSet(),
        scope_policy=_policy(ScopeDimension.USER_ID),
        semantic_configured=True,
    )

    assert len(report.checks) == 5


def test_semantic_configuration_does_not_load_model_dependencies() -> None:
    modules_before = {name: sys.modules.get(name) for name in ("torch", "transformers")}

    run_preflight(
        _store(_memory("m1")),
        transcripts=TranscriptSet(),
        semantic_configured=True,
    )

    assert {name: sys.modules.get(name) for name in modules_before} == modules_before


def test_preflight_module_has_no_forbidden_architecture_imports() -> None:
    path = Path("src/palintrace/preflight.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith(
            (
                "palintrace.evaluation",
                "palintrace.mutations",
                "palintrace.retrieval",
                "palintrace.semantics.local_nli",
            )
        )
        for module in imports
    )
    assert not imports & {"mem0", "graphiti_core", "letta_client"}
