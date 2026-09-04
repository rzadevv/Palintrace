from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest

import palintrace.command as command
from palintrace import __main__ as module_entrypoint
from palintrace import cli
from palintrace.adapters import (
    AdapterCapabilities,
    GraphitiAdapter,
    LettaAdapter,
    Mem0Adapter,
    adapter_capabilities,
)
from palintrace.checkers import CheckerResult, CheckerStats, EvidenceItem, Finding
from palintrace.models import MemoryScope, NormalizedMemory, NormalizedStore
from palintrace.retrieval import RetrievalHit, RetrievalObservation, RetrievalUsage
from palintrace.taxonomy import DefectClass


def _command_parser(name: str) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = command.build_parser()
    commands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return parser, commands.choices[name]


def _write_stale_store(path: Path, *, stale: bool = True) -> None:
    memories = [NormalizedMemory(id="old", content="Old value.", active=True)]
    if stale:
        memories.append(
            NormalizedMemory(
                id="new",
                content="New value.",
                active=True,
                supersedes=("old",),
            )
        )
    NormalizedStore(adapter="test", memories=tuple(memories)).to_json(path)


def _write_redundant_store(path: Path) -> None:
    scope = MemoryScope(user_id="user-1")
    NormalizedStore(
        adapter="test",
        memories=(
            NormalizedMemory(id="first", content="Same value.", scope=scope),
            NormalizedMemory(id="second", content="Same value.", scope=scope),
        ),
    ).to_json(path)


def _audit_arguments(
    store: Path,
    checker: str,
    *,
    fail_on: str | None = None,
    output: Path | None = None,
    sarif_output: Path | None = None,
) -> list[str]:
    arguments = ["audit", "--store", str(store), "--checker", checker]
    if fail_on is not None:
        arguments.extend(("--fail-on", fail_on))
    if output is not None:
        arguments.extend(("--output", str(output)))
    if sarif_output is not None:
        arguments.extend(("--sarif-output", str(sarif_output)))
    return arguments


def _write_observation(path: Path, *, sufficient: bool) -> None:
    hits = (RetrievalHit(memory_id="target", rank=1),) if sufficient else ()
    observation = RetrievalObservation(
        request_id="request-1",
        query_sha256=hashlib.sha256(b"Which memory?").hexdigest(),
        expected_memory_ids=("target",),
        top_k=1,
        retriever_id="recorded-retriever",
        retriever_version="1",
        hits=hits,
        usage=RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    )
    path.write_text(observation.to_json(), encoding="utf-8")


def _retrieval_arguments(
    observation: Path,
    *,
    fail_on: str | None = None,
    output: Path | None = None,
    sarif_output: Path | None = None,
) -> list[str]:
    arguments = [
        "retrieval-audit",
        "--observation",
        str(observation),
        "--policy",
        "all_expected",
    ]
    if fail_on is not None:
        arguments.extend(("--fail-on", fail_on))
    if output is not None:
        arguments.extend(("--output", str(output)))
    if sarif_output is not None:
        arguments.extend(("--sarif-output", str(sarif_output)))
    return arguments


def test_public_entrypoints_use_command_main() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["palintrace"] == "palintrace.command:main"
    assert module_entrypoint.main is command.main


@pytest.mark.parametrize("adapter", ["file", "mem0", "graphiti", "letta"])
def test_capabilities_parser_accepts_builtin_adapters(adapter: str) -> None:
    parser, command_parser = _command_parser("capabilities")

    args = parser.parse_args(["capabilities", "--adapter", adapter])
    adapter_action = next(
        action for action in command_parser._actions if action.dest == "adapter"
    )

    assert args.command == "capabilities"
    assert args.adapter == adapter
    assert tuple(adapter_action.choices) == ("file", "mem0", "graphiti", "letta")
    assert adapter_action.required is True


def test_capabilities_parser_requires_adapter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = command.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["capabilities"])

    assert raised.value.code == 2
    assert "the following arguments are required: --adapter" in capsys.readouterr().err


def test_capabilities_parser_rejects_unknown_adapter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = command.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["capabilities", "--adapter", "unknown"])

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert "invalid choice" in error
    assert "Traceback" not in error


def test_capabilities_parser_has_optional_path_output() -> None:
    parser, command_parser = _command_parser("capabilities")
    output_action = next(
        action for action in command_parser._actions if action.dest == "output"
    )

    args = parser.parse_args(
        ["capabilities", "--adapter", "file", "--output", "capabilities.json"]
    )

    assert args.output == Path("capabilities.json")
    assert output_action.type is Path
    assert output_action.default is None
    assert output_action.required is False


def test_capabilities_command_writes_json_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert command.main(["capabilities", "--adapter", "graphiti"]) == 0
    captured = capsys.readouterr()
    capabilities = AdapterCapabilities.model_validate_json(captured.out)

    assert capabilities.adapter == "graphiti"
    assert captured.out == adapter_capabilities("graphiti").to_json()
    assert captured.err == ""


def test_capabilities_command_writes_json_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "graphiti-capabilities.json"

    assert (
        command.main(
            [
                "capabilities",
                "--adapter",
                "graphiti",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    AdapterCapabilities.model_validate_json(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == adapter_capabilities("graphiti").to_json().encode("utf-8")
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("adapter", ["mem0", "graphiti", "letta"])
def test_capabilities_command_does_not_access_backends(
    adapter: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_backend_access(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("backend access attempted")

    for adapter_type in (Mem0Adapter, GraphitiAdapter, LettaAdapter):
        monkeypatch.setattr(adapter_type, "__init__", fail_backend_access)

    assert command.main(["capabilities", "--adapter", adapter]) == 0
    captured = capsys.readouterr()
    assert AdapterCapabilities.model_validate_json(captured.out).adapter == adapter
    assert captured.err == ""


def test_capabilities_filesystem_error_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "missing" / "capabilities.json"

    with pytest.raises(SystemExit) as raised:
        command.main(
            ["capabilities", "--adapter", "file", "--output", str(output)]
        )

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert "No such file or directory" in error
    assert "Traceback" not in error
    assert not output.exists()


@pytest.mark.parametrize("name", ["audit", "retrieval-audit"])
def test_audit_parsers_have_optional_fail_on(name: str) -> None:
    _, parser = _command_parser(name)
    action = next(action for action in parser._actions if action.dest == "fail_on")

    assert tuple(action.choices) == ("info", "warning", "error")
    assert action.default is None
    assert action.required is False


@pytest.mark.parametrize("name", ["audit", "retrieval-audit"])
def test_audit_parsers_have_optional_sarif_output(name: str) -> None:
    _, parser = _command_parser(name)
    action = next(action for action in parser._actions if action.dest == "sarif_output")

    assert action.type is Path
    assert action.default is None
    assert action.required is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["dump", "--adapter", "file", "--source", "store.json", "--fail-on", "error"],
        [
            "mutate",
            "--store",
            "store.json",
            "--defect",
            "stale_active",
            "--output",
            "mutated.json",
            "--manifest",
            "manifest.json",
            "--fail-on",
            "error",
        ],
    ],
)
def test_non_audit_parsers_reject_fail_on(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    parser = command.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(arguments)

    assert raised.value.code == 2
    assert "unrecognized arguments: --fail-on error" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "dump",
            "--adapter",
            "file",
            "--source",
            "store.json",
            "--sarif-output",
            "result.sarif",
        ],
        [
            "mutate",
            "--store",
            "store.json",
            "--defect",
            "stale_active",
            "--output",
            "mutated.json",
            "--manifest",
            "manifest.json",
            "--sarif-output",
            "result.sarif",
        ],
    ],
)
def test_non_audit_parsers_reject_sarif_output(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    parser = command.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(arguments)

    assert raised.value.code == 2
    assert "unrecognized arguments: --sarif-output result.sarif" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["dump", "--adapter", "file", "--source", "store.json"],
        [
            "mutate",
            "--store",
            "store.json",
            "--defect",
            "stale_active",
            "--output",
            "mutated.json",
            "--manifest",
            "manifest.json",
        ],
    ],
)
def test_non_audit_commands_delegate_to_frozen_cli(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Sequence[str] | None] = []

    def run_frozen_cli(argv: Sequence[str] | None = None) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(cli, "main", run_frozen_cli)

    assert command.main(arguments) == 0
    assert calls == [arguments]


def test_audit_with_findings_without_fail_on_returns_zero(tmp_path: Path) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / "result.json"
    _write_stale_store(store)

    assert command.main(_audit_arguments(store, "stale_active", output=output)) == 0
    assert len(
        CheckerResult.model_validate_json(output.read_text(encoding="utf-8")).findings
    ) == 1


@pytest.mark.parametrize("threshold", ["info", "warning", "error"])
def test_zero_findings_pass_every_threshold(tmp_path: Path, threshold: str) -> None:
    store = tmp_path / "clean.json"
    output = tmp_path / f"result-{threshold}.json"
    _write_stale_store(store, stale=False)

    assert (
        command.main(
            _audit_arguments(store, "stale_active", fail_on=threshold, output=output)
        )
        == 0
    )
    assert (
        CheckerResult.model_validate_json(output.read_text(encoding="utf-8")).findings
        == ()
    )


@pytest.mark.parametrize("threshold", ["info", "warning", "error"])
def test_error_result_fails_every_threshold(tmp_path: Path, threshold: str) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / f"result-{threshold}.json"
    _write_stale_store(store)

    assert (
        command.main(
            _audit_arguments(store, "stale_active", fail_on=threshold, output=output)
        )
        == 1
    )


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [("info", 1), ("warning", 1), ("error", 0)],
)
def test_warning_result_obeys_threshold(
    tmp_path: Path, threshold: str, expected: int
) -> None:
    store = tmp_path / "redundant.json"
    output = tmp_path / f"result-{threshold}.json"
    _write_redundant_store(store)

    assert (
        command.main(
            _audit_arguments(store, "redundancy_bloat", fail_on=threshold, output=output)
        )
        == expected
    )


def test_gate_preserves_output_bytes(tmp_path: Path) -> None:
    store = tmp_path / "stale.json"
    normal_output = tmp_path / "normal.json"
    gated_output = tmp_path / "gated.json"
    _write_stale_store(store)

    assert command.main(_audit_arguments(store, "stale_active", output=normal_output)) == 0
    assert (
        command.main(
            _audit_arguments(store, "stale_active", fail_on="error", output=gated_output)
        )
        == 1
    )
    CheckerResult.model_validate_json(gated_output.read_text(encoding="utf-8"))
    assert gated_output.read_bytes() == normal_output.read_bytes()


def test_gate_preserves_json_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = tmp_path / "stale.json"
    _write_stale_store(store)

    assert command.main(_audit_arguments(store, "stale_active", fail_on="error")) == 1
    captured = capsys.readouterr()
    result = CheckerResult.model_validate_json(captured.out)

    assert len(result.findings) == 1
    assert captured.out == result.to_json()
    assert captured.err == ""


def test_retrieval_audit_without_fail_on_returns_zero(tmp_path: Path) -> None:
    observation = tmp_path / "observation.json"
    output = tmp_path / "result.json"
    _write_observation(observation, sufficient=False)

    assert command.main(_retrieval_arguments(observation, output=output)) == 0
    assert len(
        CheckerResult.model_validate_json(output.read_text(encoding="utf-8")).findings
    ) == 1


@pytest.mark.parametrize(("sufficient", "expected"), [(False, 1), (True, 0)])
def test_retrieval_audit_obeys_error_threshold(
    tmp_path: Path, sufficient: bool, expected: int
) -> None:
    observation = tmp_path / "observation.json"
    output = tmp_path / "result.json"
    _write_observation(observation, sufficient=sufficient)

    assert (
        command.main(
            _retrieval_arguments(observation, fail_on="error", output=output)
        )
        == expected
    )
    result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert bool(result.findings) is not sufficient


def test_invalid_threshold_is_an_argument_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    _write_stale_store(store)

    with pytest.raises(SystemExit) as raised:
        command.main(_audit_arguments(store, "stale_active", fail_on="critical"))

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert "invalid choice" in error
    assert "Traceback" not in error


def test_existing_input_error_still_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as raised:
        command.main(_audit_arguments(missing, "stale_active", fail_on="error"))

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert "No such file or directory" in error
    assert "Traceback" not in error


def _result_with_confidence(confidence: float) -> CheckerResult:
    finding = Finding(
        finding_id=f"finding-confidence-{confidence}",
        defect_class=DefectClass.STALE_ACTIVE,
        memory_ids=("old",),
        confidence=confidence,
        evidence=(EvidenceItem(kind="test", message="Test evidence."),),
    )
    return CheckerResult(
        checker_id="stale_active",
        checker_version="1.0",
        defect_class=DefectClass.STALE_ACTIVE,
        findings=(finding,),
        stats=CheckerStats(memories_scanned=1, findings_emitted=1),
    )


def test_gate_does_not_inspect_finding_confidence() -> None:
    assert command._gate_triggered(_result_with_confidence(0.0), "error") is True
    assert command._gate_triggered(_result_with_confidence(1.0), "error") is True


def test_audit_writes_canonical_and_sarif_outputs(tmp_path: Path) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / "result.json"
    sarif_output = tmp_path / "result.sarif"
    _write_stale_store(store)

    assert (
        command.main(
            _audit_arguments(
                store,
                "stale_active",
                output=output,
                sarif_output=sarif_output,
            )
        )
        == 0
    )
    result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))
    sarif = json.loads(sarif_output.read_text(encoding="utf-8"))
    sarif_results = sarif["runs"][0]["results"]

    assert len(sarif_results) == len(result.findings) == 1
    assert sarif_results[0]["ruleId"] == result.rule_id
    assert (
        sarif_results[0]["fingerprints"]["palintraceFindingId"]
        == result.findings[0].finding_id
    )


def test_gated_audit_writes_both_outputs_before_exit_one(tmp_path: Path) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / "result.json"
    sarif_output = tmp_path / "result.sarif"
    _write_stale_store(store)

    assert (
        command.main(
            _audit_arguments(
                store,
                "stale_active",
                fail_on="error",
                output=output,
                sarif_output=sarif_output,
            )
        )
        == 1
    )
    CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert json.loads(sarif_output.read_text(encoding="utf-8"))["version"] == "2.1.0"


def test_sarif_output_does_not_change_canonical_json_bytes(tmp_path: Path) -> None:
    store = tmp_path / "stale.json"
    plain_output = tmp_path / "plain.json"
    projected_output = tmp_path / "projected.json"
    sarif_output = tmp_path / "result.sarif"
    _write_stale_store(store)

    assert command.main(_audit_arguments(store, "stale_active", output=plain_output)) == 0
    assert (
        command.main(
            _audit_arguments(
                store,
                "stale_active",
                output=projected_output,
                sarif_output=sarif_output,
            )
        )
        == 0
    )
    assert projected_output.read_bytes() == plain_output.read_bytes()


def test_sarif_file_preserves_canonical_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    sarif_output = tmp_path / "result.sarif"
    _write_stale_store(store)

    assert (
        command.main(
            _audit_arguments(store, "stale_active", sarif_output=sarif_output)
        )
        == 0
    )
    captured = capsys.readouterr()
    result = CheckerResult.model_validate_json(captured.out)

    assert captured.out == result.to_json()
    assert captured.err == ""
    assert "$schema" not in json.loads(captured.out)
    assert "$schema" in json.loads(sarif_output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("sufficient", [False, True])
def test_retrieval_audit_writes_matching_sarif(
    tmp_path: Path, sufficient: bool
) -> None:
    observation = tmp_path / "observation.json"
    output = tmp_path / "result.json"
    sarif_output = tmp_path / "result.sarif"
    _write_observation(observation, sufficient=sufficient)

    assert (
        command.main(
            _retrieval_arguments(
                observation,
                output=output,
                sarif_output=sarif_output,
            )
        )
        == 0
    )
    result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))
    sarif_results = json.loads(sarif_output.read_text(encoding="utf-8"))["runs"][0][
        "results"
    ]

    assert len(sarif_results) == len(result.findings) == (0 if sufficient else 1)


def test_sarif_output_cannot_overwrite_audit_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    _write_stale_store(store)
    original = store.read_bytes()

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(store, "stale_active", sarif_output=store)
        )

    assert raised.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert store.read_bytes() == original


def test_sarif_output_must_differ_from_canonical_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / "result.json"
    _write_stale_store(store)

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "stale_active",
                output=output,
                sarif_output=output,
            )
        )

    assert raised.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert not output.exists()


def test_sarif_output_cannot_overwrite_retrieval_observation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observation = tmp_path / "observation.json"
    _write_observation(observation, sufficient=False)
    original = observation.read_bytes()

    with pytest.raises(SystemExit) as raised:
        command.main(
            _retrieval_arguments(observation, sarif_output=observation)
        )

    assert raised.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert observation.read_bytes() == original


def test_sarif_filesystem_error_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / "result.json"
    sarif_output = tmp_path / "missing" / "result.sarif"
    _write_stale_store(store)

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "stale_active",
                output=output,
                sarif_output=sarif_output,
            )
        )

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert "No such file or directory" in error
    assert "Traceback" not in error
    CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert not sarif_output.exists()
