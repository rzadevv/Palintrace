from pathlib import Path

from memlint.models import MemoryScope, NormalizedMemory, NormalizedStore, TranscriptSet
from memlint.serialization import dumps_transcripts, load_store, loads_store, loads_transcripts


def test_store_serialization_round_trip_and_determinism(tmp_path: Path) -> None:
    store = NormalizedStore(
        adapter="file",
        memories=(
            NormalizedMemory(
                id="m1",
                content="User prefers Python.",
                scope=MemoryScope(user_id="user-123"),
                raw={"z": 1, "a": 2},
            ),
        ),
    )

    first = store.to_json()
    second = store.to_json()
    output = tmp_path / "normalized.json"
    store.to_json(output)

    assert first == second
    assert loads_store(first) == store
    assert load_store(output) == store
    assert '"exported_at": null' in first


def test_raw_can_be_excluded_only_by_explicit_serialization_choice() -> None:
    store = NormalizedStore(
        adapter="file",
        memories=(NormalizedMemory(id="m1", content="A memory", raw={"native": 1}),),
    )

    assert "raw" in store.to_dict()["memories"][0]
    assert "raw" not in store.to_dict(include_raw=False)["memories"][0]


def test_transcript_serialization_round_trip() -> None:
    fixture = Path("tests/fixtures/transcripts.json").read_text(encoding="utf-8")
    transcripts = loads_transcripts(fixture)
    serialized = dumps_transcripts(transcripts)

    assert loads_transcripts(serialized) == transcripts
    assert isinstance(transcripts, TranscriptSet)

