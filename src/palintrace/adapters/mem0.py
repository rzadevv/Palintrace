"""Mem0 normalization and optional platform transport."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from palintrace.adapters.base import (
    AdapterAuthenticationError,
    AdapterDataError,
    AdapterDependencyError,
    MemoryAdapter,
    deterministic_memory_id,
    merge_scope,
    normalize_records,
    record_mapping,
    transport_error,
)
from palintrace.models import MemoryScope, NormalizedMemory, ProvenanceStatus, SourceRef
from palintrace.models.store import NormalizedStore


class Mem0Adapter(MemoryAdapter):
    """Export Mem0 platform records using the documented ``MemoryClient.get_all`` API."""

    name = "mem0"

    def __init__(
        self,
        *,
        records: Iterable[Any] | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        filters: Mapping[str, Any] | None = None,
        page_size: int = 100,
    ):
        self._records = records
        self._client = client
        self._api_key = api_key
        self._filters = dict(filters or {})
        self._page_size = page_size

    def dump(self) -> NormalizedStore:
        records = self._records if self._records is not None else self._fetch_records()
        query_scope = _mem0_scope(self._filters)
        return normalize_records(
            self.name,
            records,
            lambda record: normalize_mem0_record(record, scope=query_scope),
        )

    def _fetch_records(self) -> list[Any]:
        self._validate_live_arguments()
        client = self._client
        if client is None:
            api_key = self._api_key or os.getenv("MEM0_API_KEY")
            if not api_key:
                raise AdapterAuthenticationError(
                    "Mem0 live export requires MEM0_API_KEY or api_key=..."
                )
            try:
                from mem0 import MemoryClient  # type: ignore[import-not-found]
            except ImportError as error:
                raise AdapterDependencyError(
                    "Mem0 live export requires the optional dependency: "
                    "pip install 'palintrace[mem0]'"
                ) from error
            client = MemoryClient(api_key=api_key)

        records: list[Any] = []
        page_number = 1
        try:
            while True:
                response = client.get_all(
                    filters=self._filters,
                    page=page_number,
                    page_size=self._page_size,
                )
                if isinstance(response, list):
                    records.extend(response)
                    break
                if not isinstance(response, Mapping) or not isinstance(
                    response.get("results"), list
                ):
                    raise AdapterDataError(
                        "Mem0 get_all() must return a list or a paginated {'results': [...]} object"
                    )
                page_records = response["results"]
                records.extend(page_records)
                if not response.get("next") or not page_records:
                    break
                page_number += 1
        except AdapterDataError:
            raise
        except Exception as error:
            raise transport_error("Mem0", error) from error
        return records

    def _validate_live_arguments(self) -> None:
        if (
            isinstance(self._page_size, bool)
            or not isinstance(self._page_size, int)
            or not 1 <= self._page_size <= 200
        ):
            raise AdapterDataError(
                "Mem0 live export page_size must be an integer between 1 and 200"
            )
        if not any(
            self._filters.get(field) is not None and str(self._filters[field]).strip()
            for field in ("user_id", "agent_id", "run_id")
        ):
            raise AdapterDataError(
                "Mem0 live export requires at least one entity filter: user_id, agent_id, or run_id"
            )


def normalize_mem0_record(record: Any, *, scope: MemoryScope | None = None) -> NormalizedMemory:
    """Normalize one documented Mem0 memory response."""

    source = record_mapping(record)
    content = source.get("memory")
    if not isinstance(content, str):
        raise AdapterDataError("Mem0 record requires string field 'memory'")

    normalized_scope = merge_scope(
        MemoryScope(
            user_id=_optional_string(source.get("user_id")),
            agent_id=_optional_string(source.get("agent_id")),
            session_id=_optional_string(
                source.get("run_id")
                if source.get("run_id") is not None
                else source.get("session_id")
            ),
        ),
        scope,
    )
    refs_supplied = "source_refs" in source
    refs_value = source.get("source_refs", [])
    if not isinstance(refs_value, list):
        raise AdapterDataError("Mem0 source_refs must be a list when explicitly supplied")
    source_refs = tuple(SourceRef.model_validate(item) for item in refs_value)
    provenance_status = (
        ProvenanceStatus.DECLARED
        if source_refs
        else ProvenanceStatus.KNOWN_ABSENT
        if refs_supplied
        else ProvenanceStatus.UNAVAILABLE
    )
    memory_id = source.get("id")
    if memory_id is None:
        memory_id = deterministic_memory_id(
            "mem0",
            content=content,
            created_at=source.get("created_at"),
            scope=normalized_scope,
            source_refs=source_refs,
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
            active=source.get("active") if "active" in source else None,
            supersedes=tuple(str(item) for item in source.get("supersedes", [])),
            embedding=source.get("embedding"),
            raw=source,
        )
    except ValidationError as error:
        raise AdapterDataError(f"invalid Mem0 record: {error}") from error


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _mem0_scope(values: Mapping[str, Any]) -> MemoryScope:
    """Read only documented entity dimensions, never arbitrary metadata."""

    try:
        return MemoryScope(
            user_id=_optional_string(values.get("user_id")),
            agent_id=_optional_string(values.get("agent_id")),
            session_id=_optional_string(
                values.get("run_id")
                if values.get("run_id") is not None
                else values.get("session_id")
            ),
        )
    except ValidationError as error:
        raise AdapterDataError(f"invalid Mem0 query scope: {error}") from error
