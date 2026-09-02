import json
from pathlib import Path
from typing import Any

from palintrace.adapters.graphiti import GraphitiAdapter, normalize_graphiti_record
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
