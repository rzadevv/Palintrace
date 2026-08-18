import pytest

from memlint.adapters import AdapterDataError
from memlint.adapters.base import deterministic_memory_id, json_safe
from memlint.adapters.file import normalize_file_record
from memlint.models import MemoryScope, SourceRef


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_json_safe_rejects_non_finite_floats_at_any_depth(invalid: float) -> None:
    with pytest.raises(AdapterDataError, match="NaN or infinity"):
        json_safe({"outer": [0.0, {"invalid": invalid}]})


def test_generated_ids_canonicalize_timestamps_and_source_ref_order() -> None:
    scope = MemoryScope(user_id="user-123")
    first_refs = (
        SourceRef(transcript_id="transcript-2", turn_idx=1),
        SourceRef(transcript_id="transcript-1", turn_idx=0, span=(0, 5)),
    )
    second_refs = tuple(reversed(first_refs))

    first = deterministic_memory_id(
        "file",
        content="A memory",
        created_at="2026-08-10T14:20:00+02:00",
        scope=scope,
        source_refs=first_refs,
    )
    second = deterministic_memory_id(
        "file",
        content="A memory",
        created_at="2026-08-10T12:20:00Z",
        scope=scope,
        source_refs=second_refs,
    )

    assert first == second


def test_source_provided_id_is_preserved_unchanged() -> None:
    memory = normalize_file_record(
        {"id": "Source-ID:01", "content": "A memory"}, source_format="json"
    )

    assert memory.id == "Source-ID:01"
