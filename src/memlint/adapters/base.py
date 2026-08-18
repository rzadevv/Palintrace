"""Small shared adapter contract and normalization utilities."""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import AwareDatetime, JsonValue, TypeAdapter, ValidationError

from memlint.models import MemoryScope, NormalizedMemory, NormalizedStore, SourceRef

_AWARE_DATETIME = TypeAdapter(AwareDatetime)


class AdapterError(RuntimeError):
    """Base class for actionable adapter failures."""


class AdapterDependencyError(AdapterError):
    """An optional backend SDK is required but not installed."""


class AdapterAuthenticationError(AdapterError):
    """Credentials are absent or rejected by a backend."""


class AdapterDataError(AdapterError):
    """A source record or export has an unsupported/malformed shape."""


@runtime_checkable
class AdapterContract(Protocol):
    """Structural interface available to future normalized consumers."""

    name: str

    def dump(self) -> NormalizedStore: ...


class MemoryAdapter(ABC):
    """Minimal base class: configured adapters produce one normalized store."""

    name: str

    @abstractmethod
    def dump(self) -> NormalizedStore:
        """Export backend records without performing downstream analysis."""


def json_safe(value: Any) -> JsonValue:
    """Convert SDK values to lossless JSON-compatible data or fail clearly."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdapterDataError("JSON values must not contain NaN or infinity")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    raise AdapterDataError(f"value of type {type(value).__name__} is not JSON serializable")


def record_mapping(record: Any) -> dict[str, Any]:
    """Convert a dict or SDK/Pydantic model to a plain JSON record."""

    if isinstance(record, Mapping):
        result = json_safe(record)
    elif callable(model_dump := getattr(record, "model_dump", None)):
        result = json_safe(model_dump(mode="json"))
    elif callable(to_dict := getattr(record, "to_dict", None)):
        result = json_safe(to_dict())
    else:
        raise AdapterDataError(
            f"expected a mapping or SDK model with model_dump()/to_dict(), got "
            f"{type(record).__name__}"
        )
    if not isinstance(result, dict):
        raise AdapterDataError("source record must convert to a JSON object")
    return dict(result)


def deterministic_memory_id(
    adapter: str,
    *,
    content: str,
    created_at: Any,
    scope: MemoryScope,
    source_refs: Any,
) -> str:
    """Derive a stable ID from portable identity fields when a source has none."""

    identity = {
        "adapter": adapter,
        "content": content,
        "created_at": _canonical_timestamp(created_at),
        "scope": scope.model_dump(mode="json"),
        "source_refs": _canonical_source_refs(source_refs),
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{adapter}:{digest}"


def _canonical_timestamp(value: Any) -> str | None:
    """Represent equivalent aware timestamps identically for generated IDs."""

    if value is None:
        return None
    try:
        parsed = _AWARE_DATETIME.validate_python(value)
    except ValidationError as error:
        raise AdapterDataError(f"generated ID requires an aware created_at: {error}") from error
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_source_refs(value: Any) -> list[dict[str, JsonValue]]:
    """Canonicalize transcript references as an order-independent collection."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AdapterDataError("generated ID source_refs must be a sequence")
    encoded: dict[str, dict[str, JsonValue]] = {}
    for item in value:
        try:
            source_ref = item if isinstance(item, SourceRef) else SourceRef.model_validate(item)
        except ValidationError as error:
            raise AdapterDataError(f"invalid source reference for generated ID: {error}") from error
        data = source_ref.model_dump(mode="json")
        key = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        encoded[key] = data
    return [encoded[key] for key in sorted(encoded)]


def merge_scope(primary: MemoryScope, fallback: MemoryScope | None) -> MemoryScope:
    """Fill only missing scope dimensions from adapter configuration."""

    if fallback is None:
        return primary
    return MemoryScope(
        user_id=primary.user_id or fallback.user_id,
        agent_id=primary.agent_id or fallback.agent_id,
        session_id=primary.session_id or fallback.session_id,
    )


def normalize_records(
    adapter_name: str,
    records: Iterable[Any],
    normalizer: Callable[[Any], NormalizedMemory],
) -> NormalizedStore:
    """Normalize records with indexed errors and store-level ID validation."""

    memories: list[NormalizedMemory] = []
    for index, record in enumerate(records):
        try:
            memories.append(normalizer(record))
        except AdapterDataError as error:
            raise AdapterDataError(
                f"{adapter_name} record {index} is malformed: {error}"
            ) from error
        except AdapterError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise AdapterDataError(
                f"{adapter_name} record {index} is malformed: {error}"
            ) from error
    try:
        return NormalizedStore(
            adapter=adapter_name,
            memories=tuple(sorted(memories, key=lambda memory: memory.id)),
        )
    except ValidationError as error:
        raise AdapterDataError(f"{adapter_name} export is invalid: {error}") from error


def page_items(page: Any) -> list[Any]:
    """Materialize SDK page/list responses without depending on an SDK type."""

    if isinstance(page, list | tuple):
        return list(page)
    for attribute in ("items", "data", "results"):
        value = getattr(page, attribute, None)
        if value is not None and not callable(value):
            return list(value)
    if isinstance(page, Iterable) and not isinstance(page, (str, bytes, Mapping)):
        return list(page)
    raise AdapterDataError(f"unsupported SDK page response: {type(page).__name__}")


def transport_error(backend: str, error: Exception) -> AdapterError:
    """Classify common authentication failures while retaining backend context."""

    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (401, 403) or "auth" in type(error).__name__.lower():
        return AdapterAuthenticationError(f"{backend} authentication failed: {error}")
    return AdapterError(f"{backend} export failed: {error}")
