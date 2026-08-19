from pathlib import Path

import pytest

from memlint.checkers import CheckerResult
from memlint.cli import main
from memlint.mutations import MutationRequest, mutate
from memlint.serialization import dumps_transcripts, load_store, load_transcripts
from memlint.taxonomy import DefectClass


def test_audit_cli_writes_deterministic_checker_result(tmp_path: Path) -> None:
    base_store = load_store("examples/mutation-store.json")
    transcripts = load_transcripts("examples/mutation-transcripts.json")
    mutation_result = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.ORPHANED_PROVENANCE,
            subtype="missing_turn",
            target_memory_id="preference-python",
        ),
        transcripts,
    )
    store_path = tmp_path / "store.json"
    transcripts_path = tmp_path / "transcripts.json"
    output = tmp_path / "findings.json"
    mutation_result.mutated_store.to_json(store_path)
    transcripts_path.write_text(dumps_transcripts(transcripts), encoding="utf-8")
    before_store = store_path.read_bytes()
    before_transcripts = transcripts_path.read_bytes()

    result = main(
        [
            "audit",
            "--store",
            str(store_path),
            "--transcripts",
            str(transcripts_path),
            "--checker",
            "orphaned_provenance",
            "--output",
            str(output),
        ]
    )
    checker_result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))

    assert result == 0
    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == mutation_result.manifest.gold_label.memory_ids
    assert store_path.read_bytes() == before_store
    assert transcripts_path.read_bytes() == before_transcripts


def test_audit_cli_emits_clean_result_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "audit",
            "--store",
            "examples/mutation-store.json",
            "--transcripts",
            "examples/mutation-transcripts.json",
            "--checker",
            "orphaned_provenance",
        ]
    )
    checker_result = CheckerResult.model_validate_json(capsys.readouterr().out)

    assert result == 0
    assert checker_result.findings == ()


def test_audit_cli_surfaces_missing_required_transcript_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--checker",
                "orphaned_provenance",
            ]
        )

    assert "orphaned_provenance checker requires a TranscriptSet" in capsys.readouterr().err


@pytest.mark.parametrize("input_name", ["store", "transcripts"])
def test_audit_cli_rejects_overwriting_inputs(input_name: str, tmp_path: Path) -> None:
    store_path = Path("examples/mutation-store.json")
    transcripts_path = Path("examples/mutation-transcripts.json")
    output = store_path if input_name == "store" else transcripts_path

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                str(store_path),
                "--transcripts",
                str(transcripts_path),
                "--checker",
                "orphaned_provenance",
                "--output",
                str(output),
            ]
        )


def test_audit_cli_has_no_manifest_argument(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--transcripts",
                "examples/mutation-transcripts.json",
                "--checker",
                "orphaned_provenance",
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
