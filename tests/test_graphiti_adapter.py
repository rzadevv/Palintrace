import json
from pathlib import Path
from typing import Any

import pytest

from palintrace.adapters.base import AdapterDataError
from palintrace.adapters.graphiti import (
    GraphitiAdapter,
    GroupScopeDimension,
    normalize_graphiti_record,
)
from palintrace.checkers import RedundancyBloatChecker
from palintrace.models import MemoryScope, ProvenanceStatus


def _fixture_record() -> dict[str, Any]:
    return json.loads(Path("tests/fixtures/graphiti.json").read_text(encoding="utf-8"))["edges"][0]


def test_graphiti_fixture_normalization_preserves_temporal_graph_data() -> None:
    memory = normalize_graphiti_record(_fixture_record(), scope=MemoryScope(user_id="user-123"))

    assert memory.id == "m1"
    assert memory.content == "User prefers Python."
    assert memory.active is True
    assert memory.source_refs == ()
    assert memory.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert memory.embedding == (0.1, 0.2)
    assert memory.raw["episodes"] == ["conversation-1"]
    assert memory.raw["valid_at"] == "2026-08-10T14:20:00+02:00"
    assert memory.raw["source_node_uuid"] == "person-1"


def test_graphiti_invalidated_edge_is_inactive_but_does_not_infer_supersession() -> None:
    record = _fixture_record() | {"invalid_at": "2026-08-11T00:00:00+02:00"}
    memory = normalize_graphiti_record(record)

    assert memory.active is False
    assert memory.supersedes == ()


def test_graphiti_uses_only_explicit_episode_to_transcript_mapping() -> None:
    memory = normalize_graphiti_record(
        _fixture_record(),
        episode_transcript_map={
            "conversation-1": {
                "transcript_id": "transcript-1",
                "turn_idx": 4,
                "span": [0, 20],
            }
        },
    )

    assert memory.source_refs[0].transcript_id == "transcript-1"
    assert memory.provenance_status is ProvenanceStatus.DECLARED
    assert memory.raw["episodes"] == ["conversation-1"]


def test_graphiti_fixture_adapter_is_synchronous_and_offline() -> None:
    store = GraphitiAdapter(records=[_fixture_record()]).dump()

    assert store.adapter == "graphiti"
    assert len(store) == 1


def test_graphiti_default_leaves_group_id_out_of_scope() -> None:
    memory = normalize_graphiti_record(_fixture_record())
    adapter_store = GraphitiAdapter(records=[_fixture_record()]).dump()

    assert memory.scope == MemoryScope()
    assert memory.raw["group_id"] == "user-123-group"
    assert adapter_store.memories[0].scope == MemoryScope()


@pytest.mark.parametrize("dimension", ["user_id", "agent_id", "session_id"])
def test_graphiti_group_scope_maps_group_id_into_the_chosen_dimension(
    dimension: GroupScopeDimension,
) -> None:
    memory = normalize_graphiti_record(_fixture_record(), group_scope=dimension)
    adapter_store = GraphitiAdapter(
        records=[_fixture_record()], group_scope=dimension
    ).dump()

    assert getattr(memory.scope, dimension) == "user-123-group"
    assert memory.scope.model_dump(exclude={dimension}) == {
        field: None
        for field in ("user_id", "agent_id", "session_id")
        if field != dimension
    }
    assert adapter_store.memories[0].scope == memory.scope


@pytest.mark.parametrize(
    "group_id",
    [None, "", "   ", 123],
    ids=("missing", "empty", "blank", "non_string"),
)
def test_graphiti_group_scope_requires_a_usable_group_id(group_id: object) -> None:
    record = _fixture_record()
    if group_id is None:
        record.pop("group_id")
    else:
        record["group_id"] = group_id

    with pytest.raises(AdapterDataError, match="non-blank string field 'group_id'"):
        normalize_graphiti_record(record, group_scope="user_id")


def test_graphiti_group_scope_conflicting_with_caller_scope_is_rejected() -> None:
    with pytest.raises(AdapterDataError, match="conflicts with the supplied scope user_id"):
        normalize_graphiti_record(
            _fixture_record(),
            scope=MemoryScope(user_id="user-123"),
            group_scope="user_id",
        )


def test_graphiti_group_scope_matching_caller_scope_is_accepted() -> None:
    memory = normalize_graphiti_record(
        _fixture_record(),
        scope=MemoryScope(user_id="user-123-group"),
        group_scope="user_id",
    )

    assert memory.scope == MemoryScope(user_id="user-123-group")


def test_graphiti_group_scope_combines_with_other_caller_dimensions() -> None:
    memory = normalize_graphiti_record(
        _fixture_record(),
        scope=MemoryScope(agent_id="agent-7", session_id="session-2"),
        group_scope="user_id",
    )

    assert memory.scope == MemoryScope(
        user_id="user-123-group",
        agent_id="agent-7",
        session_id="session-2",
    )


def test_graphiti_group_scope_is_never_inferred_from_the_group_id_value() -> None:
    record = _fixture_record() | {"group_id": "user-123-group"}

    memory = normalize_graphiti_record(record, group_scope="session_id")

    assert memory.scope == MemoryScope(session_id="user-123-group")
    assert memory.scope.user_id is None


def test_group_scoped_graphiti_store_becomes_eligible_for_redundancy_bloat() -> None:
    first = _fixture_record()
    second = _fixture_record() | {"uuid": "m2"}

    unscoped = GraphitiAdapter(records=[first, second]).dump()
    scoped = GraphitiAdapter(records=[first, second], group_scope="user_id").dump()

    unscoped_result = RedundancyBloatChecker().check(unscoped)
    scoped_result = RedundancyBloatChecker().check(scoped)

    assert unscoped_result.findings == ()
    assert unscoped_result.stats.details["unscoped_memories_skipped"] == 2
    assert tuple(finding.memory_ids for finding in scoped_result.findings) == (("m1", "m2"),)
