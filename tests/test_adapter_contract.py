import json
from pathlib import Path
from typing import Any

from palintrace.adapters import FileAdapter
from palintrace.adapters.graphiti import normalize_graphiti_record
from palintrace.adapters.letta import normalize_letta_record
from palintrace.adapters.mem0 import normalize_mem0_record
from palintrace.models import MemoryScope

FIXTURES = Path("tests/fixtures")


def _load(name: str, key: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))[key][0]


def test_equivalent_backend_records_share_the_same_normalized_core_fields() -> None:
    """Equivalent backend records preserve the same normalized core fields."""

    scope = MemoryScope(user_id="user-123")
    memories = [
        FileAdapter(FIXTURES / "file_store.json").dump().memories[0],
        normalize_mem0_record(_load("mem0.json", "results")),
        normalize_graphiti_record(_load("graphiti.json", "edges"), scope=scope),
        normalize_letta_record(
            _load("letta.json", "passages") | {"memory_type": "archival"}, scope=scope
        ),
    ]

    shared = [
        {
            "id": memory.id,
            "content": memory.content,
            "created_at": memory.created_at,
            "user_id": memory.scope.user_id,
        }
        for memory in memories
    ]
    assert shared == [shared[0]] * 4
    assert len({json.dumps(memory.raw, sort_keys=True) for memory in memories}) == 4


def test_backend_semantics_are_not_forced_into_false_equivalence() -> None:
    scope = MemoryScope(user_id="user-123")
    mem0 = normalize_mem0_record(_load("mem0.json", "results"))
    graphiti = normalize_graphiti_record(_load("graphiti.json", "edges"), scope=scope)

    assert mem0.active is None
    assert graphiti.active is True
    assert mem0.source_refs == ()
    assert graphiti.source_refs == ()
