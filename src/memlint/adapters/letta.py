"""Letta core-block and archival-passage normalization."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import ValidationError

from memlint.adapters.base import (
    AdapterAuthenticationError,
    AdapterDataError,
    AdapterDependencyError,
    MemoryAdapter,
    deterministic_memory_id,
    normalize_records,
    page_items,
    record_mapping,
    transport_error,
)
from memlint.models import MemoryScope, NormalizedMemory, ProvenanceStatus, SourceRef
from memlint.models.store import NormalizedStore

LettaMemoryType = Literal["core", "archival"]


class LettaAdapter(MemoryAdapter):
    """Export an agent's attached core blocks and archival passages."""

    name = "letta"

    def __init__(
        self,
        *,
        records: Iterable[Any] | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
    ):
        self._records = records
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._agent_id = agent_id
        self._user_id = user_id

    def dump(self) -> NormalizedStore:
        records = self._records if self._records is not None else self._fetch_records()
        scope = MemoryScope(user_id=self._user_id, agent_id=self._agent_id)
        return normalize_records(
            self.name,
            records,
            lambda record: normalize_letta_record(record, scope=scope),
        )

    def _fetch_records(self) -> list[dict[str, Any]]:
        if not self._agent_id:
            raise AdapterDataError("Letta live export requires agent_id")
        client = self._client
        if client is None:
            api_key = self._api_key or os.getenv("LETTA_API_KEY")
            if not api_key and not self._base_url:
                raise AdapterAuthenticationError(
                    "Letta live export requires LETTA_API_KEY/api_key or a self-hosted base_url"
                )
            try:
                from letta_client import Letta  # type: ignore[import-not-found]
            except ImportError as error:
                raise AdapterDependencyError(
                    "Letta live export requires: pip install 'memlint[letta]'"
                ) from error
            options: dict[str, str] = {}
            if api_key:
                options["api_key"] = api_key
            if self._base_url:
                options["base_url"] = self._base_url
            client = Letta(**options)

        records: list[dict[str, Any]] = []
        try:
            block_after: str | None = None
            while True:
                blocks_page = client.agents.blocks.list(
                    self._agent_id, after=block_after, limit=200
                )
                blocks = page_items(blocks_page)
                for block in blocks:
                    item = record_mapping(block)
                    item["memory_type"] = "core"
                    records.append(item)
                if len(blocks) < 200:
                    break
                last_block_id = record_mapping(blocks[-1]).get("id")
                if last_block_id is None:
                    raise AdapterDataError("Letta block pagination requires block IDs")
                block_after = str(last_block_id)

            after: str | None = None
            while True:
                passages = client.agents.passages.list(
                    self._agent_id, after=after, limit=200
                )
                page = page_items(passages)
                for passage in page:
                    item = record_mapping(passage)
                    item["memory_type"] = "archival"
                    records.append(item)
                if len(page) < 200:
                    break
                last = record_mapping(page[-1]).get("id")
                if last is None:
                    raise AdapterDataError("Letta passage pagination requires passage IDs")
                after = str(last)
        except AdapterDataError:
            raise
        except Exception as error:
            raise transport_error("Letta", error) from error
        return records


def normalize_letta_record(
    record: Any, *, scope: MemoryScope | None = None
) -> NormalizedMemory:
    """Normalize a Letta core block or archival passage."""

    source = record_mapping(record)
    memory_type = source.get("memory_type")
    if memory_type not in ("core", "archival"):
        raise AdapterDataError("Letta record requires memory_type 'core' or 'archival'")
    content_field = "value" if memory_type == "core" else "text"
    content = source.get(content_field)
    if not isinstance(content, str):
        raise AdapterDataError(
            f"Letta {memory_type} record requires string field {content_field!r}"
        )
    normalized_scope = scope or MemoryScope()

    refs_supplied = "source_refs" in source
    refs_value = source.get("source_refs", [])
    if not isinstance(refs_value, list):
        raise AdapterDataError("Letta source_refs must be a list when explicitly supplied")
    source_refs = tuple(SourceRef.model_validate(item) for item in refs_value)
    provenance_status = (
        ProvenanceStatus.VERIFIED
        if source_refs
        else ProvenanceStatus.KNOWN_ABSENT
        if refs_supplied
        else ProvenanceStatus.UNAVAILABLE
    )

    if "active" in source:
        active = source["active"]
    elif memory_type == "core":
        active = True
    else:
        active = not bool(source.get("is_deleted", False))

    memory_id = source.get("id")
    if memory_id is None:
        memory_id = deterministic_memory_id(
            "letta",
            content=content,
            created_at=source.get("created_at"),
            scope=normalized_scope,
            source_refs=refs_value,
        )

    try:
        return NormalizedMemory(
            id=str(memory_id),
            content=content,
            created_at=source.get("created_at"),
            updated_at=source.get("updated_at"),
            source_refs=source_refs,
            provenance_status=provenance_status,
            scope=normalized_scope,
            active=active,
            supersedes=(),
            embedding=source.get("embedding") if memory_type == "archival" else None,
            raw=source,
        )
    except ValidationError as error:
        raise AdapterDataError(f"invalid Letta record: {error}") from error
