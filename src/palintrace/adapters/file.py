"""Reference adapter for explicit JSON, JSONL, YAML, and Markdown inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from palintrace.adapters.base import (
    AdapterDataError,
    MemoryAdapter,
    deterministic_memory_id,
    json_safe,
    normalize_records,
    record_mapping,
)
from palintrace.models import MemoryScope, NormalizedMemory, ProvenanceStatus, SourceRef
from palintrace.models.store import NormalizedStore


class FileAdapter(MemoryAdapter):
    """Normalize a documented, non-guessing subset of common file formats."""

    name = "file"

    def __init__(self, source: str | Path):
        self.source = Path(source)

    def dump(self) -> NormalizedStore:
        records, source_format = self._load_records()
        return normalize_records(
            self.name,
            records,
            lambda record: normalize_file_record(record, source_format=source_format),
        )

    def _load_records(self) -> tuple[list[dict[str, Any]], str]:
        if not self.source.exists():
            raise AdapterDataError(f"file source does not exist: {self.source}")
        suffix = self.source.suffix.lower()
        text = self.source.read_text(encoding="utf-8")
        try:
            if suffix == ".json":
                return _structured_records(json.loads(text)), "json"
            if suffix == ".jsonl":
                return _jsonl_records(text), "jsonl"
            if suffix in {".yaml", ".yml"}:
                return _structured_records(yaml.safe_load(text)), "yaml"
            if suffix in {".md", ".markdown"}:
                return [_markdown_record(text)], "markdown"
        except AdapterDataError:
            raise
        except (json.JSONDecodeError, yaml.YAMLError, TypeError, ValueError) as error:
            raise AdapterDataError(f"could not parse {self.source}: {error}") from error
        raise AdapterDataError(
            f"unsupported file extension {suffix!r}; expected .json, .jsonl, .yaml, .yml, or .md"
        )


def _structured_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and "memories" in value:
        value = value["memories"]
    elif isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise AdapterDataError("structured file must be a record, a list, or {'memories': [...]}")
    if not all(isinstance(record, Mapping) for record in value):
        raise AdapterDataError("every memory entry must be an object")
    return [dict(record) for record in value]


def _jsonl_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdapterDataError(f"invalid JSONL on line {line_number}: {error}") from error
        if not isinstance(record, Mapping):
            raise AdapterDataError(f"JSONL line {line_number} must contain an object")
        records.append(dict(record))
    return records


def _markdown_record(text: str) -> dict[str, Any]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise AdapterDataError("Markdown memory must begin with YAML front matter ('---')")
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise AdapterDataError("Markdown front matter has no closing '---'")
    metadata = yaml.safe_load("".join(lines[1:closing])) or {}
    if not isinstance(metadata, Mapping):
        raise AdapterDataError("Markdown front matter must be a YAML object")
    body = "".join(lines[closing + 1 :]).strip("\r\n")
    record = dict(metadata)
    if "content" in record:
        raise AdapterDataError("Markdown content belongs in the body, not front matter")
    record["content"] = body
    return record


def normalize_file_record(record: Any, *, source_format: str) -> NormalizedMemory:
    """Normalize one explicit file record without rewriting its content."""

    source = record_mapping(record)
    content = source.get("content")
    if not isinstance(content, str):
        raise AdapterDataError("file memory requires string field 'content'")

    scope_value = source.get("scope", {})
    if not isinstance(scope_value, Mapping):
        raise AdapterDataError("scope must be an object")
    scope_data = dict(scope_value)
    for field in ("user_id", "agent_id", "session_id"):
        if field in source and field not in scope_data:
            scope_data[field] = source[field]
    scope = MemoryScope.model_validate(scope_data)

    refs_supplied = "source_refs" in source
    refs_value = source.get("source_refs", [])
    if not isinstance(refs_value, list):
        raise AdapterDataError("source_refs must be a list")
    source_refs = tuple(SourceRef.model_validate(item) for item in refs_value)
    explicit_status = source.get("provenance_status")
    if explicit_status is not None:
        provenance_status = ProvenanceStatus(explicit_status)
    elif source_refs:
        provenance_status = ProvenanceStatus.DECLARED
    elif refs_supplied:
        provenance_status = ProvenanceStatus.KNOWN_ABSENT
    else:
        provenance_status = ProvenanceStatus.UNAVAILABLE

    created_at = source.get("created_at")
    memory_id = source.get("id")
    if memory_id is None:
        memory_id = deterministic_memory_id(
            "file",
            content=content,
            created_at=created_at,
            scope=scope,
            source_refs=source_refs,
        )

    normalized_fields = {
        "id",
        "content",
        "created_at",
        "updated_at",
        "source_refs",
        "provenance_status",
        "scope",
        "user_id",
        "agent_id",
        "session_id",
        "active",
        "supersedes",
        "embedding",
        "raw",
    }
    supplied_raw = source.get("raw", {})
    if not isinstance(supplied_raw, Mapping):
        raise AdapterDataError("raw must be an object")
    raw = {
        "file_format": source_format,
        "source_metadata": {
            key: json_safe(value) for key, value in source.items() if key not in normalized_fields
        },
        "supplied_raw": json_safe(supplied_raw),
    }

    try:
        return NormalizedMemory(
            id=str(memory_id),
            content=content,
            created_at=created_at,
            updated_at=source.get("updated_at"),
            source_refs=source_refs,
            provenance_status=provenance_status,
            scope=scope,
            active=source.get("active"),
            supersedes=tuple(str(item) for item in source.get("supersedes", [])),
            embedding=source.get("embedding"),
            raw=raw,
        )
    except ValidationError as error:
        raise AdapterDataError(f"invalid normalized file record: {error}") from error
