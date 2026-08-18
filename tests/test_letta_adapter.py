import json
from pathlib import Path
from typing import Any

from memlint.adapters.letta import LettaAdapter, normalize_letta_record
from memlint.models import MemoryScope, ProvenanceStatus


def _archival_fixture() -> dict[str, Any]:
    value = json.loads(Path("tests/fixtures/letta.json").read_text(encoding="utf-8"))
    return value["passages"][0] | {"memory_type": "archival"}


def test_letta_archival_fixture_normalization() -> None:
    memory = normalize_letta_record(
        _archival_fixture(), scope=MemoryScope(user_id="user-123", agent_id="agent-1")
    )

    assert memory.id == "m1"
    assert memory.content == "User prefers Python."
    assert memory.active is True
    assert memory.scope.agent_id == "agent-1"
    assert memory.embedding == (0.1, 0.2)
    assert memory.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert memory.raw["memory_type"] == "archival"


def test_letta_core_block_has_no_fabricated_timestamp_or_embedding() -> None:
    memory = normalize_letta_record(
        {"id": "block-1", "value": "User prefers Python.", "memory_type": "core"},
        scope=MemoryScope(agent_id="agent-1"),
    )

    assert memory.created_at is None
    assert memory.updated_at is None
    assert memory.embedding is None
    assert memory.active is True


def test_letta_deleted_archival_passage_is_inactive() -> None:
    memory = normalize_letta_record(_archival_fixture() | {"is_deleted": True})

    assert memory.active is False


def test_letta_supplied_source_refs_are_declared() -> None:
    memory = normalize_letta_record(
        {
            "id": "block-1",
            "value": "User prefers Python.",
            "memory_type": "core",
            "source_refs": [{"transcript_id": "transcript-1"}],
        }
    )

    assert memory.provenance_status is ProvenanceStatus.DECLARED


def test_letta_transport_calls_documented_agent_memory_endpoints() -> None:
    class Blocks:
        def list(self, agent_id: str, **kwargs: Any) -> list[dict[str, Any]]:
            assert agent_id == "agent-1"
            assert kwargs == {"after": None, "limit": 200}
            return [{"id": "block-1", "value": "Core memory", "label": "human"}]

    class Passages:
        def list(self, agent_id: str, **kwargs: Any) -> list[dict[str, Any]]:
            assert agent_id == "agent-1"
            assert kwargs == {"after": None, "limit": 200}
            return [{"id": "passage-1", "text": "Archival memory", "is_deleted": False}]

    class Agents:
        blocks = Blocks()
        passages = Passages()

    class FakeClient:
        agents = Agents()

    store = LettaAdapter(client=FakeClient(), agent_id="agent-1").dump()

    assert [memory.raw["memory_type"] for memory in store.memories] == ["core", "archival"]
