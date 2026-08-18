import json
from pathlib import Path
from typing import Any

from memlint.adapters.mem0 import Mem0Adapter, normalize_mem0_record
from memlint.models import ProvenanceStatus


def _fixture_record() -> dict[str, Any]:
    return json.loads(Path("tests/fixtures/mem0.json").read_text(encoding="utf-8"))["results"][0]


def test_mem0_fixture_normalization() -> None:
    memory = normalize_mem0_record(_fixture_record())

    assert memory.id == "m1"
    assert memory.content == "User prefers Python."
    assert memory.scope.user_id == "user-123"
    assert memory.active is None
    assert memory.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert memory.raw["metadata"] == {"category": "preference"}


def test_mem0_transport_uses_documented_paginated_get_all_shape() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def get_all(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            record = _fixture_record() | {"id": f"m{kwargs['page']}"}
            return {
                "count": 2,
                "next": "next" if kwargs["page"] == 1 else None,
                "previous": None,
                "results": [record],
            }

    client = FakeClient()
    store = Mem0Adapter(client=client, filters={"user_id": "user-123"}, page_size=1).dump()

    assert len(store) == 2
    assert [call["page"] for call in client.calls] == [1, 2]
    assert all(call["filters"] == {"user_id": "user-123"} for call in client.calls)

