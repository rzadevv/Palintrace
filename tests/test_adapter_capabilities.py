from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from palintrace import adapters
from palintrace.adapters import (
    ADAPTER_CAPABILITIES_SCHEMA_VERSION,
    AdapterCapabilities,
    adapter_capabilities,
)

CAPABILITY_FIELDS = (
    "provenance",
    "created_at",
    "updated_at",
    "active_state",
    "supersession",
    "user_scope",
    "agent_scope",
    "session_scope",
    "embeddings",
)

BUILTIN_MATRIX = {
    "file": {
        "provenance": "supported",
        "created_at": "supported",
        "updated_at": "supported",
        "active_state": "supported",
        "supersession": "supported",
        "user_scope": "supported",
        "agent_scope": "supported",
        "session_scope": "supported",
        "embeddings": "supported",
    },
    "mem0": {
        "provenance": "conditional",
        "created_at": "supported",
        "updated_at": "supported",
        "active_state": "conditional",
        "supersession": "conditional",
        "user_scope": "supported",
        "agent_scope": "supported",
        "session_scope": "supported",
        "embeddings": "conditional",
    },
    "graphiti": {
        "provenance": "conditional",
        "created_at": "supported",
        "updated_at": "unsupported",
        "active_state": "supported",
        "supersession": "unsupported",
        "user_scope": "conditional",
        "agent_scope": "conditional",
        "session_scope": "conditional",
        "embeddings": "conditional",
    },
    "letta": {
        "provenance": "conditional",
        "created_at": "supported",
        "updated_at": "supported",
        "active_state": "supported",
        "supersession": "unsupported",
        "user_scope": "conditional",
        "agent_scope": "conditional",
        "session_scope": "unsupported",
        "embeddings": "conditional",
    },
}


def _capability_payload(state: str = "supported") -> dict[str, object]:
    return {"adapter": "test", **dict.fromkeys(CAPABILITY_FIELDS, state)}


def test_capability_schema_version() -> None:
    assert ADAPTER_CAPABILITIES_SCHEMA_VERSION == "0.1"
    assert adapter_capabilities("file").schema_version == "0.1"
    assert json.loads(adapter_capabilities("file").to_json())["schema_version"] == "0.1"

    with pytest.raises(ValidationError, match="unsupported schema_version"):
        AdapterCapabilities.model_validate(
            {**_capability_payload(), "schema_version": "0.2"}
        )


@pytest.mark.parametrize("state", ["supported", "conditional", "unsupported"])
def test_capability_states_are_accepted(state: str) -> None:
    capabilities = AdapterCapabilities.model_validate(_capability_payload(state))

    assert all(getattr(capabilities, field) == state for field in CAPABILITY_FIELDS)


@pytest.mark.parametrize("state", ["unknown", "available", "partial", "yes"])
def test_other_capability_states_are_rejected(state: str) -> None:
    with pytest.raises(ValidationError):
        AdapterCapabilities.model_validate(_capability_payload(state))


def test_capability_serialization_is_deterministic(tmp_path: Path) -> None:
    capabilities = adapter_capabilities("graphiti")
    output = tmp_path / "graphiti-capabilities.json"

    first = capabilities.to_json()
    second = capabilities.to_json()
    written = capabilities.to_json(output)

    assert first == second == written
    assert json.loads(first)["adapter"] == "graphiti"
    assert output.read_text(encoding="utf-8") == written
    assert output.read_bytes() == written.encode("utf-8")
    for forbidden in (
        "generated_at",
        "executed_at",
        "timestamp",
        "hostname",
        '"cwd"',
        "package runtime",
        "backend runtime version",
    ):
        assert forbidden not in first


@pytest.mark.parametrize(("adapter", "expected"), BUILTIN_MATRIX.items())
def test_builtin_capability_matrix(adapter: str, expected: dict[str, str]) -> None:
    capabilities = adapter_capabilities(adapter)
    serialized = capabilities.model_dump(mode="json")

    assert capabilities.adapter == adapter
    assert set(serialized) == {"schema_version", "adapter", *CAPABILITY_FIELDS}
    assert {field: serialized[field] for field in CAPABILITY_FIELDS} == expected


@pytest.mark.parametrize("adapter", ["unknown", "Mem0", "MEM0", "custom"])
def test_unknown_adapter_is_rejected_without_name_normalization(adapter: str) -> None:
    with pytest.raises(ValueError, match="unknown adapter"):
        adapter_capabilities(adapter)


def test_capability_model_is_frozen_and_forbids_extra_fields() -> None:
    capabilities = adapter_capabilities("file")

    with pytest.raises(ValidationError, match="frozen"):
        capabilities.provenance = "unsupported"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AdapterCapabilities.model_validate(
            {**_capability_payload(), "observability": "supported"}
        )


def test_adapter_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="adapter must not be blank"):
        AdapterCapabilities.model_validate({**_capability_payload(), "adapter": "  "})


def test_key_declarations_match_current_adapter_contracts() -> None:
    graphiti = adapter_capabilities("graphiti")
    letta = adapter_capabilities("letta")

    assert graphiti.updated_at == graphiti.supersession == "unsupported"
    assert letta.session_scope == letta.supersession == "unsupported"
    assert adapter_capabilities("mem0").provenance == "conditional"
    assert graphiti.provenance == "conditional"
    assert adapter_capabilities("file").supersession == "supported"


def test_adapter_package_exports_capability_api() -> None:
    assert {
        "ADAPTER_CAPABILITIES_SCHEMA_VERSION",
        "AdapterCapabilities",
        "adapter_capabilities",
    } <= set(adapters.__all__)
