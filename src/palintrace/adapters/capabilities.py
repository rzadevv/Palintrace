"""Static capability contracts for built-in adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ADAPTER_CAPABILITIES_SCHEMA_VERSION = "0.1"

_CapabilityState = Literal["supported", "conditional", "unsupported"]


class AdapterCapabilities(BaseModel):
    """Machine-readable support contract for one adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ADAPTER_CAPABILITIES_SCHEMA_VERSION
    adapter: str
    provenance: _CapabilityState
    created_at: _CapabilityState
    updated_at: _CapabilityState
    active_state: _CapabilityState
    supersession: _CapabilityState
    user_scope: _CapabilityState
    agent_scope: _CapabilityState
    session_scope: _CapabilityState
    embeddings: _CapabilityState

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        if value != ADAPTER_CAPABILITIES_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {value!r}")
        return value

    @field_validator("adapter")
    @classmethod
    def adapter_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adapter must not be blank")
        return value

    def to_json(
        self,
        output: str | Path | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize deterministically and optionally write to ``output``."""

        text = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        if indent is not None:
            text += "\n"
        if output is not None:
            Path(output).write_text(text, encoding="utf-8")
        return text


_BUILTIN_CAPABILITIES: Mapping[str, AdapterCapabilities] = MappingProxyType(
    {
        "file": AdapterCapabilities(
            adapter="file",
            provenance="supported",
            created_at="supported",
            updated_at="supported",
            active_state="supported",
            supersession="supported",
            user_scope="supported",
            agent_scope="supported",
            session_scope="supported",
            embeddings="supported",
        ),
        "mem0": AdapterCapabilities(
            adapter="mem0",
            provenance="conditional",
            created_at="supported",
            updated_at="supported",
            active_state="conditional",
            supersession="conditional",
            user_scope="supported",
            agent_scope="supported",
            session_scope="supported",
            embeddings="conditional",
        ),
        "graphiti": AdapterCapabilities(
            adapter="graphiti",
            provenance="conditional",
            created_at="supported",
            updated_at="unsupported",
            active_state="supported",
            supersession="unsupported",
            user_scope="conditional",
            agent_scope="conditional",
            session_scope="conditional",
            embeddings="conditional",
        ),
        "letta": AdapterCapabilities(
            adapter="letta",
            provenance="conditional",
            created_at="supported",
            updated_at="supported",
            active_state="supported",
            supersession="unsupported",
            user_scope="conditional",
            agent_scope="conditional",
            session_scope="unsupported",
            embeddings="conditional",
        ),
    }
)


def adapter_capabilities(adapter: str) -> AdapterCapabilities:
    """Return the declared capabilities for one built-in adapter."""

    try:
        return _BUILTIN_CAPABILITIES[adapter]
    except KeyError:
        raise ValueError(f"unknown adapter: {adapter!r}") from None
