import json
from pathlib import Path

import pytest

from memlint.cli import main
from memlint.models import NormalizedStore
from memlint.mutations import MutationManifest


def test_mutate_cli_writes_separate_deterministic_outputs(tmp_path: Path) -> None:
    first_store = tmp_path / "first-store.json"
    first_manifest = tmp_path / "first-manifest.json"
    second_store = tmp_path / "second-store.json"
    second_manifest = tmp_path / "second-manifest.json"
    arguments = [
        "mutate",
        "--store",
        "examples/mutation-store.json",
        "--transcripts",
        "examples/mutation-transcripts.json",
        "--defect",
        "unsupported_claim",
        "--seed",
        "42",
        "--target-id",
        "preference-python",
        "--replace-from",
        "Python",
        "--replace-to",
        "Rust",
    ]

    assert main([*arguments, "--output", str(first_store), "--manifest", str(first_manifest)]) == 0
    assert (
        main([*arguments, "--output", str(second_store), "--manifest", str(second_manifest)])
        == 0
    )

    NormalizedStore.model_validate_json(first_store.read_text(encoding="utf-8"))
    MutationManifest.model_validate_json(first_manifest.read_text(encoding="utf-8"))
    assert first_store.read_bytes() == second_store.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def test_mutate_cli_rejects_shared_output_path(tmp_path: Path) -> None:
    output = tmp_path / "shared.json"

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "mutate",
                "--store",
                "examples/mutation-store.json",
                "--defect",
                "redundancy_bloat",
                "--target-id",
                "preference-python",
                "--output",
                str(output),
                "--manifest",
                str(output),
            ]
        )


def test_manifest_has_no_execution_timestamp(tmp_path: Path) -> None:
    output = tmp_path / "mutated.json"
    manifest = tmp_path / "manifest.json"

    main(
        [
            "mutate",
            "--store",
            "examples/mutation-store.json",
            "--defect",
            "redundancy_bloat",
            "--target-id",
            "preference-python",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert "created_at" not in payload
    assert "executed_at" not in payload
