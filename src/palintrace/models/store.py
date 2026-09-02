"""A deterministic collection of normalized memories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, field_validator

from palintrace.models.memory import NormalizedMemory

SCHEMA_VERSION = "0.1"


class NormalizedStore(BaseModel):
    """Versioned normalized memory export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_VERSION
    adapter: str
    exported_at: AwareDatetime | None = None
    memories: tuple[NormalizedMemory, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def supported_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {value!r}")
        return value

    @field_validator("adapter")
    @classmethod
    def adapter_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adapter must not be blank")
        return value

    @field_validator("memories")
    @classmethod
    def memory_ids_must_be_unique(
        cls, value: tuple[NormalizedMemory, ...]
    ) -> tuple[NormalizedMemory, ...]:
        ids = [memory.id for memory in value]
        if len(set(ids)) != len(ids):
            duplicates = sorted({memory_id for memory_id in ids if ids.count(memory_id) > 1})
            raise ValueError(f"duplicate normalized memory IDs: {duplicates}")
        return value

    def __len__(self) -> int:
        return len(self.memories)

    def get(self, memory_id: str) -> NormalizedMemory | None:
        """Return a memory by ID, or ``None`` when it is absent."""

        return next((memory for memory in self.memories if memory.id == memory_id), None)

    def to_dict(self, *, include_raw: bool = True) -> dict[str, JsonValue]:
        """Return a JSON-compatible dictionary in stable schema order."""

        payload = self.model_dump(mode="json")
        if not include_raw:
            for memory in payload["memories"]:
                memory.pop("raw", None)
        return cast(dict[str, JsonValue], payload)

    def to_json(
        self,
        output: str | Path | None = None,
        *,
        indent: int | None = 2,
        include_raw: bool = True,
    ) -> str:
        """Serialize deterministically and optionally write the result to ``output``."""

        text = json.dumps(
            self.to_dict(include_raw=include_raw),
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
