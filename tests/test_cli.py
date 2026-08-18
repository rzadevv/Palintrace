from pathlib import Path

import pytest

from memlint.cli import main
from memlint.models import NormalizedStore


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
