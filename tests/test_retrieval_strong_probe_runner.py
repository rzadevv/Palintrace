from __future__ import annotations

from pathlib import Path

import pytest

import tools.evaluate_retrieval_strong_probe as runner


def test_runner_exposes_only_optional_output_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--output" in output
    for forbidden in (
        "--fixture",
        "--retriever",
        "--top-k",
        "--policy",
        "--k1",
        "--b",
        "--tokenizer",
        "--stop-words",
        "--stemming",
        "--query-rewriting",
        "--boost",
    ):
        assert forbidden not in output


def test_bad_fixture_fails_before_retriever_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_fixture = tmp_path / "retrieval-probe.json"
    bad_fixture.write_bytes(runner.FROZEN_FIXTURE_PATH.read_bytes() + b"\n")
    constructed = False

    def build_retriever(_store: object) -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(runner, "FROZEN_FIXTURE_PATH", bad_fixture)
    monkeypatch.setattr(runner, "_build_retriever", build_retriever)
    with pytest.raises(SystemExit) as error:
        runner.main([])
    assert error.value.code == 2
    assert constructed is False
    assert "fixture SHA mismatch" in capsys.readouterr().err


def test_output_path_must_be_new_and_outside_repository(tmp_path: Path) -> None:
    assert runner._validate_output_path(None) is None
    output = tmp_path / "retrieval-result.json"
    assert runner._validate_output_path(output) == output.resolve()
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        runner._validate_output_path(output)
    with pytest.raises(ValueError, match="outside the repository"):
        runner._validate_output_path(runner.REPOSITORY_ROOT / "retrieval-result.json")


def test_store_construction_is_four_then_twelve_without_retrieval() -> None:
    spec = runner.probe.load_retrieval_strong_probe(runner.FROZEN_FIXTURE_PATH)
    case = spec.cases[0]
    baseline = runner._build_store(case, include_distractors=False)
    mutated = runner._build_store(case, include_distractors=True)
    assert len(baseline.memories) == 4
    assert len(mutated.memories) == 12
    assert baseline.memories == mutated.memories[:4]
    assert {memory.scope.user_id for memory in mutated.memories} == {
        case.scope_user_id
    }
