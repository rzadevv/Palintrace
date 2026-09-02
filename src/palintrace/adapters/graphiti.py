"""Graphiti entity-edge normalization and optional read-only Neo4j transport."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from palintrace.adapters.base import (
    AdapterAuthenticationError,
    AdapterDataError,
    AdapterDependencyError,
    AdapterError,
    MemoryAdapter,
    deterministic_memory_id,
    normalize_records,
    record_mapping,
    transport_error,
)
from palintrace.models import MemoryScope, NormalizedMemory, ProvenanceStatus, SourceRef
from palintrace.models.store import NormalizedStore


class GraphitiAdapter(MemoryAdapter):
    """Export Graphiti ``EntityEdge`` facts without invoking its LLM ingestion path."""

    name = "graphiti"

    def __init__(
        self,
        *,
        records: Iterable[Any] | None = None,
        driver: Any | None = None,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        group_ids: Iterable[str] | None = None,
        scope: MemoryScope | None = None,
        episode_transcript_map: Mapping[str, SourceRef | Mapping[str, Any] | str] | None = None,
        include_embeddings: bool = False,
        page_size: int = 500,
    ):
        self._records = records
        self._driver = driver
        self._uri = uri
        self._user = user
        self._password = password
        self._group_ids = tuple(group_ids or ())
        self._scope = scope
        self._episode_transcript_map = dict(episode_transcript_map or {})
        self._include_embeddings = include_embeddings
        self._page_size = page_size

    def dump(self) -> NormalizedStore:
        if self._records is not None:
            return self._normalize(self._records)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_dump())
        raise AdapterError(
            "GraphitiAdapter.dump() cannot run inside an event loop; await async_dump() instead"
        )

    async def async_dump(self) -> NormalizedStore:
        records = self._records if self._records is not None else await self._fetch_records()
        return self._normalize(records)

    def _normalize(self, records: Iterable[Any]) -> NormalizedStore:
        return normalize_records(
            self.name,
            records,
            lambda record: normalize_graphiti_record(
                record,
                scope=self._scope,
                episode_transcript_map=self._episode_transcript_map,
            ),
        )

    async def _fetch_records(self) -> list[Any]:
        if not self._group_ids:
            raise AdapterDataError("Graphiti live export requires at least one group_id")
        driver = self._driver
        owns_driver = driver is None
        if driver is None:
            if not self._uri or not self._user or not self._password:
                raise AdapterAuthenticationError(
                    "Graphiti live export requires Neo4j uri, user, and password"
                )
            try:
                from graphiti_core.driver.neo4j_driver import (  # type: ignore[import-not-found]
                    Neo4jDriver,
                )
            except ImportError as error:
                raise AdapterDependencyError(
                    "Graphiti live export requires: pip install 'palintrace[graphiti]'"
                ) from error
            driver = Neo4jDriver(uri=self._uri, user=self._user, password=self._password)

        try:
            from graphiti_core.edges import EntityEdge  # type: ignore[import-not-found]
            from graphiti_core.errors import (  # type: ignore[import-not-found]
                GroupsEdgesNotFoundError,
            )
        except ImportError as error:
            if owns_driver:
                await driver.close()
            raise AdapterDependencyError(
                "Graphiti live export requires: pip install 'palintrace[graphiti]'"
            ) from error

        records: list[Any] = []
        cursor: str | None = None
        try:
            while True:
                try:
                    page = await EntityEdge.get_by_group_ids(
                        driver,
                        list(self._group_ids),
                        limit=self._page_size,
                        uuid_cursor=cursor,
                        with_embeddings=self._include_embeddings,
                    )
                except GroupsEdgesNotFoundError:
                    break
                records.extend(page)
                if len(page) < self._page_size:
                    break
                cursor = str(page[-1].uuid)
        except Exception as error:
            raise transport_error("Graphiti", error) from error
        finally:
            if owns_driver:
                await driver.close()
        return records


def normalize_graphiti_record(
    record: Any,
    *,
    scope: MemoryScope | None = None,
    episode_transcript_map: Mapping[str, SourceRef | Mapping[str, Any] | str] | None = None,
) -> NormalizedMemory:
    """Normalize one Graphiti ``EntityEdge`` fact."""

    source = record_mapping(record)
    content = source.get("fact")
    if not isinstance(content, str):
        raise AdapterDataError("Graphiti EntityEdge requires string field 'fact'")
    normalized_scope = scope or MemoryScope()

    episodes = source.get("episodes", [])
    if not isinstance(episodes, list):
        raise AdapterDataError("Graphiti episodes must be a list")
    source_refs = _mapped_episode_refs(episodes, episode_transcript_map or {})
    provenance_status = ProvenanceStatus.DECLARED if source_refs else ProvenanceStatus.UNAVAILABLE

    state_supplied = "invalid_at" in source or "expired_at" in source
    active = None
    if state_supplied:
        active = source.get("invalid_at") is None and source.get("expired_at") is None

    memory_id = source.get("uuid")
    if memory_id is None:
        memory_id = deterministic_memory_id(
            "graphiti",
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
            updated_at=None,
            source_refs=source_refs,
            provenance_status=provenance_status,
            scope=normalized_scope,
            active=active,
            supersedes=(),
            embedding=source.get("fact_embedding"),
            raw=source,
        )
    except ValidationError as error:
        raise AdapterDataError(f"invalid Graphiti record: {error}") from error


def _mapped_episode_refs(
    episodes: list[Any],
    episode_transcript_map: Mapping[str, SourceRef | Mapping[str, Any] | str],
) -> tuple[SourceRef, ...]:
    """Use only explicit episode-to-transcript mappings for normalized provenance."""

    refs: list[SourceRef] = []
    for episode in episodes:
        mapped = episode_transcript_map.get(str(episode))
        if mapped is None:
            continue
        if isinstance(mapped, SourceRef):
            refs.append(mapped)
        elif isinstance(mapped, str):
            refs.append(SourceRef(transcript_id=mapped))
        else:
            refs.append(SourceRef.model_validate(mapped))
    return tuple(refs)
