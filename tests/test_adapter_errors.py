import pytest

from palintrace.adapters import AdapterAuthenticationError, AdapterDataError
from palintrace.adapters.graphiti import GraphitiAdapter
from palintrace.adapters.letta import LettaAdapter
from palintrace.adapters.mem0 import Mem0Adapter


def test_mem0_missing_credentials_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM0_API_KEY", raising=False)

    with pytest.raises(AdapterAuthenticationError, match="MEM0_API_KEY"):
        Mem0Adapter(filters={"user_id": "user-1"}).dump()


def test_graphiti_requires_explicit_group_and_connection_configuration() -> None:
    with pytest.raises(AdapterDataError, match="group_id"):
        GraphitiAdapter().dump()
    with pytest.raises(AdapterAuthenticationError, match="uri, user, and password"):
        GraphitiAdapter(group_ids=["group-1"]).dump()


def test_letta_missing_agent_or_credentials_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LETTA_API_KEY", raising=False)

    with pytest.raises(AdapterDataError, match="agent_id"):
        LettaAdapter().dump()
    with pytest.raises(AdapterAuthenticationError, match="LETTA_API_KEY"):
        LettaAdapter(agent_id="agent-1").dump()
