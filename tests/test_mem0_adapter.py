import json
from pathlib import Path
from typing import Any

import pytest

from memlint.adapters import AdapterDataError
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


@pytest.mark.parametrize(
    ("filters", "scope_field", "expected"),
    [
        ({"user_id": "user-123"}, "user_id", "user-123"),
        ({"agent_id": "agent-123"}, "agent_id", "agent-123"),
        ({"run_id": "run-123"}, "session_id", "run-123"),
    ],
)
def test_mem0_query_scope_fills_missing_record_scope(
    filters: dict[str, str], scope_field: str, expected: str
) -> None:
    memory = (
        Mem0Adapter(
            records=[{"id": "m1", "memory": "A memory"}],
            filters=filters,
        )
        .dump()
        .memories[0]
    )

    assert getattr(memory.scope, scope_field) == expected


def test_mem0_record_scope_takes_precedence_over_query_scope() -> None:
    memory = (
        Mem0Adapter(
            records=[
                {
                    "id": "m1",
                    "memory": "A memory",
                    "user_id": "record-user",
                    "agent_id": "record-agent",
                    "session_id": "record-session",
                }
            ],
            filters={
                "user_id": "query-user",
                "agent_id": "query-agent",
                "run_id": "query-run",
            },
        )
        .dump()
        .memories[0]
    )

    assert memory.scope.user_id == "record-user"
    assert memory.scope.agent_id == "record-agent"
    assert memory.scope.session_id == "record-session"


def test_mem0_scope_is_not_inferred_from_metadata() -> None:
    memory = normalize_mem0_record(
        {"id": "m1", "memory": "A memory", "metadata": {"user_id": "hidden-user"}}
    )

    assert memory.scope.user_id is None


def test_mem0_live_export_requires_an_entity_filter() -> None:
    with pytest.raises(AdapterDataError, match="at least one entity filter"):
        Mem0Adapter(client=object()).dump()


@pytest.mark.parametrize(
    "filters",
    [
        {"user_id": "user-123"},
        {"agent_id": "agent-123"},
        {"run_id": "run-123"},
    ],
)
def test_mem0_live_export_accepts_each_entity_scope(filters: dict[str, str]) -> None:
    class EmptyClient:
        def get_all(self, **kwargs: Any) -> list[Any]:
            assert kwargs["filters"] == filters
            return []

    store = Mem0Adapter(client=EmptyClient(), filters=filters).dump()

    assert len(store) == 0


@pytest.mark.parametrize("page_size", [0, -1, 201, 1000, True, 1.5])
def test_mem0_live_export_rejects_invalid_page_size(page_size: Any) -> None:
    with pytest.raises(AdapterDataError, match="integer between 1 and 200"):
        Mem0Adapter(
            client=object(),
            filters={"user_id": "user-123"},
            page_size=page_size,
        ).dump()


def test_mem0_fixture_mode_does_not_require_live_arguments() -> None:
    store = Mem0Adapter(
        records=[{"id": "m1", "memory": "A memory"}],
        page_size=0,
    ).dump()

    assert len(store) == 1
