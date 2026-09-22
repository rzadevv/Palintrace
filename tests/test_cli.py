import json
from pathlib import Path

import pytest

from palintrace.cli import main
from palintrace.command import main as command_main
from palintrace.models import MemoryScope, NormalizedStore


def test_file_dump_cli_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["dump", "--adapter", "file", "--source", "examples/store.yaml"])

    payload = capsys.readouterr().out
    store = NormalizedStore.model_validate_json(payload)
    assert result == 0
    assert store.adapter == "file"
    assert store.memories[0].id == "preference-python"


@pytest.mark.parametrize(
    ("adapter", "source", "extra"),
    [
        ("mem0", "tests/fixtures/mem0.json", []),
        ("graphiti", "tests/fixtures/graphiti.json", ["--user-id", "user-123"]),
        ("letta", "tests/fixtures/letta.json", ["--agent-id", "agent-1"]),
    ],
)
def test_external_fixture_dump_cli(
    adapter: str,
    source: str,
    extra: list[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / f"{adapter}.json"

    result = main(
        ["dump", "--adapter", adapter, "--source", source, "--output", str(output), *extra]
    )
    store = NormalizedStore.model_validate_json(output.read_text(encoding="utf-8"))

    assert result == 0
    assert store.adapter == adapter
    assert len(store) == 1


def _dump(*extra: str) -> list[str]:
    return [
        "dump",
        "--adapter",
        "graphiti",
        "--source",
        "tests/fixtures/graphiti.json",
        *extra,
    ]


@pytest.mark.parametrize("dimension", ["user_id", "agent_id", "session_id"])
def test_graphiti_group_scope_flag_maps_the_chosen_dimension(
    dimension: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert command_main(_dump("--graphiti-group-scope", dimension)) == 0

    store = NormalizedStore.model_validate_json(capsys.readouterr().out)
    scope = store.memories[0].scope
    assert getattr(scope, dimension) == "user-123-group"
    assert scope.model_dump(exclude={dimension}) == {
        field: None
        for field in ("user_id", "agent_id", "session_id")
        if field != dimension
    }


def test_graphiti_dump_without_the_flag_is_byte_identical_to_before(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert command_main(_dump()) == 0
    without_flag = capsys.readouterr().out

    store = NormalizedStore.model_validate_json(without_flag)
    assert store.memories[0].scope == MemoryScope()
    assert '"group_id"' not in without_flag


def test_graphiti_group_scope_flag_is_rejected_for_other_adapters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        command_main(
            [
                "dump",
                "--adapter",
                "file",
                "--source",
                "examples/store.yaml",
                "--graphiti-group-scope",
                "user_id",
            ]
        )

    assert (
        "--graphiti-group-scope is only valid for the graphiti adapter"
        in capsys.readouterr().err
    )


def test_graphiti_group_scope_adapter_errors_exit_two_with_a_readable_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing-group.json"
    missing.write_text(
        json.dumps({"edges": [{"uuid": "m1", "fact": "User prefers Python."}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        command_main(
            [
                "dump",
                "--adapter",
                "graphiti",
                "--source",
                str(missing),
                "--graphiti-group-scope",
                "user_id",
            ]
        )

    error = capsys.readouterr().err
    assert "non-blank string field 'group_id'" in error
    assert "Traceback" not in error


def test_graphiti_group_scope_conflict_exits_two_with_a_readable_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        command_main(
            _dump("--user-id", "user-123", "--graphiti-group-scope", "user_id")
        )

    error = capsys.readouterr().err
    assert "conflicts with the supplied scope user_id" in error
    assert "Traceback" not in error
