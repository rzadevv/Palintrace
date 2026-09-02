from pathlib import Path

import pytest

from palintrace.adapters import AdapterDataError, FileAdapter
from palintrace.adapters.file import normalize_file_record
from palintrace.models import ProvenanceStatus

FIXTURES = Path("tests/fixtures")


@pytest.mark.parametrize(
    "filename",
    ["file_store.json", "file_store.yaml", "file_store.jsonl", "file_store.md"],
)
def test_supported_file_formats_normalize_equivalently(filename: str) -> None:
    reference = FileAdapter(FIXTURES / "file_store.json").dump().memories[0].semantic_dict()
    memory = FileAdapter(FIXTURES / filename).dump().memories[0]

    assert memory.semantic_dict() == reference
    expected_format = {".md": "markdown", ".yml": "yaml"}.get(
        Path(filename).suffix, Path(filename).suffix.lstrip(".")
    )
    assert memory.raw["file_format"] == expected_format


def test_missing_ids_are_deterministic_across_file_formats() -> None:
    record = {
        "content": "User prefers Python.",
        "created_at": "2026-08-10T14:20:00+02:00",
        "scope": {"user_id": "user-123"},
    }

    json_memory = normalize_file_record(record, source_format="json")
    yaml_memory = normalize_file_record(record, source_format="yaml")

    assert json_memory.id == yaml_memory.id
    assert json_memory.id.startswith("file:")
    assert json_memory.semantic_dict() == yaml_memory.semantic_dict()


def test_explicit_empty_provenance_is_known_absent() -> None:
    memory = normalize_file_record(
        {"id": "m1", "content": "A memory", "source_refs": []},
        source_format="json",
    )

    assert memory.provenance_status is ProvenanceStatus.KNOWN_ABSENT


def test_markdown_requires_explicit_front_matter(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.md"
    source.write_text("Just an arbitrary note.", encoding="utf-8")

    with pytest.raises(AdapterDataError, match="front matter"):
        FileAdapter(source).dump()


def test_malformed_record_is_not_silently_skipped(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"ok","content":"valid"}\n{"id":"bad"}\n', encoding="utf-8")

    with pytest.raises(AdapterDataError, match="record 1"):
        FileAdapter(source).dump()
