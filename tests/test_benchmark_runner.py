from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import palintrace.cli as cli
import palintrace.evaluation.preflight as preflight
import tools.run_benchmark_v0_1 as runner
from palintrace.evaluation import EvaluationInputError, load_benchmark_spec


class _JsonArtifact:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_json(self) -> str:
        return self.text


def test_runner_help_does_not_construct_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--output-dir" in output
    assert "--benchmark" in output
    assert "threshold" not in output
    assert "model" not in output
    assert "retriever" not in output


def test_preflight_failure_occurs_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructed = False

    def fail_preflight(**kwargs: object) -> object:
        del kwargs
        raise EvaluationInputError("frozen benchmark mismatch")

    def model_constructor(**kwargs: object) -> object:
        nonlocal constructed
        del kwargs
        constructed = True
        return object()

    monkeypatch.setattr(runner, "preflight_benchmark_v0_1", fail_preflight)
    monkeypatch.setattr(runner, "LocalNLISemanticJudge", model_constructor)
    output_dir = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        runner.main(["--output-dir", str(output_dir)])
    assert error.value.code == 2
    assert constructed is False
    assert not output_dir.exists()
    assert "frozen benchmark mismatch" in capsys.readouterr().err


def test_coordinated_fixture_and_manifest_tamper_fails_before_parser_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_root = tmp_path / "tests/fixtures/benchmark_v0.1"
    shutil.copytree(runner.FROZEN_FIXTURE_ROOT, fixture_root)
    manifest_path = tmp_path / preflight.DEFAULT_FIXTURE_HASH_MANIFEST_PATH
    shutil.copy2(preflight.DEFAULT_FIXTURE_HASH_MANIFEST_PATH, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_relative_path = next(
        relative_path
        for relative_path in sorted(manifest)
        if relative_path.endswith("_store.json")
    )
    fixture_path = tmp_path / fixture_relative_path
    fixture_path.write_bytes(fixture_path.read_bytes() + b"\n")
    modified_fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    manifest[fixture_relative_path] = modified_fixture_sha
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        fixture_relative_path
    ] == modified_fixture_sha

    parser_called = False
    model_constructed = False
    original_load_hash_manifest = preflight._load_hash_manifest

    def record_manifest_parse(path: Path) -> dict[str, str]:
        nonlocal parser_called
        parser_called = True
        return original_load_hash_manifest(path)

    def model_constructor(**kwargs: object) -> object:
        nonlocal model_constructed
        del kwargs
        model_constructed = True
        return object()

    monkeypatch.setattr(preflight, "_load_hash_manifest", record_manifest_parse)
    with pytest.raises(EvaluationInputError, match="hash-manifest SHA mismatch"):
        preflight.preflight_benchmark_v0_1(repository_root=tmp_path)
    assert parser_called is False

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "FROZEN_FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(runner, "LocalNLISemanticJudge", model_constructor)
    output_dir = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        runner.main(["--output-dir", str(output_dir)])
    assert error.value.code == 2
    assert parser_called is False
    assert model_constructed is False
    assert not output_dir.exists()
    assert "hash-manifest SHA mismatch" in capsys.readouterr().err


def test_runner_uses_fixed_model_only_after_preflight_and_writes_two_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    spec = load_benchmark_spec("tests/fixtures/benchmark_v0.1/benchmark.json")
    judge = object()

    def preflight(**kwargs: object) -> object:
        del kwargs
        order.append("preflight")
        return spec

    def construct_model(**kwargs: object) -> object:
        order.append("model")
        assert kwargs == {
            "model_id": "cross-encoder/nli-MiniLM2-L6-H768",
            "revision": "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
            "device": "cpu",
        }
        return judge

    def execute(**kwargs: object) -> _JsonArtifact:
        order.append("execute")
        assert kwargs["spec"] is spec
        assert kwargs["semantic_judge"] is judge
        return _JsonArtifact('{"artifact":"result"}\n')

    def provenance() -> _JsonArtifact:
        order.append("provenance")
        return _JsonArtifact('{"artifact":"provenance"}\n')

    monkeypatch.setattr(runner, "preflight_benchmark_v0_1", preflight)
    monkeypatch.setattr(runner, "LocalNLISemanticJudge", construct_model)
    monkeypatch.setattr(runner, "execute_benchmark_v0_1", execute)
    monkeypatch.setattr(runner, "build_execution_provenance", provenance)
    output_dir = tmp_path / "run"
    assert runner.main(["--output-dir", str(output_dir)]) == 0
    assert order == ["preflight", "model", "execute", "provenance"]
    assert (output_dir / "benchmark-result.json").read_bytes() == (
        b'{"artifact":"result"}\n'
    )
    assert (output_dir / "execution-provenance.json").read_bytes() == (
        b'{"artifact":"provenance"}\n'
    )


def test_runner_output_directory_is_fail_closed(tmp_path: Path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        runner._prepare_output_directory(nonempty)
    assert (nonempty / "existing.txt").read_text(encoding="utf-8") == "preserve"

    with pytest.raises(ValueError, match="frozen fixtures"):
        runner._prepare_output_directory(runner.FROZEN_FIXTURE_ROOT / "outputs")


def test_normal_cli_and_public_tree_have_no_benchmark_execution_command_or_results() -> None:
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "run_benchmark" not in cli_source
    assert "benchmark-audit" not in cli_source
    forbidden_names = {
        "benchmark-results.json",
        "benchmark-result.json",
        "results.csv",
        "performance.md",
        "leaderboard.md",
    }
    committed_or_working = {
        path.name
        for path in Path(".").rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    assert forbidden_names.isdisjoint(committed_or_working)
