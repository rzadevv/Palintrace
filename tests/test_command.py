from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from palintrace.audit import AuditReport, SkippedChecker, SkipReason
from palintrace.checkers import (
    CheckerResult,
    CheckerStats,
    EvidenceItem,
    Finding,
    PrincipalBoundaryRule,
    ScopeDimension,
    ScopeIsolationPolicy,
)
from palintrace.models import (
    MemoryScope,
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from palintrace.retrieval import RetrievalHit, RetrievalObservation, RetrievalUsage
from palintrace.semantics import (
    SemanticJudgeError,
    SemanticJudgment,
    SemanticRelation,
)
from palintrace.serialization import dumps_transcripts
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
    transcripts: Path | None = None,
    scope_policy: Path | None = None,
    semantic_model_id: str | None = None,
    semantic_model_revision: str | None = None,
) -> list[str]:
    arguments = ["audit", "--store", str(store), "--checker", checker]
    if transcripts is not None:
        arguments.extend(("--transcripts", str(transcripts)))
    if scope_policy is not None:
        arguments.extend(("--scope-policy", str(scope_policy)))
    if semantic_model_id is not None:
        arguments.extend(("--semantic-model-id", semantic_model_id))
    if semantic_model_revision is not None:
        arguments.extend(("--semantic-model-revision", semantic_model_revision))
    if fail_on is not None:
        arguments.extend(("--fail-on", fail_on))
    if output is not None:
        arguments.extend(("--output", str(output)))
    if sarif_output is not None:
        arguments.extend(("--sarif-output", str(sarif_output)))
    return arguments


class _FakeAggregateJudge:
    def __init__(self, model_id: str, revision: str) -> None:
        self.judge_id = f"hf-nli:{model_id}"
        self.judge_version = revision
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        return SemanticJudgment(relation=SemanticRelation.ENTAILMENT, score=1.0)


def _write_complete_aggregate_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    store_path = tmp_path / "store.json"
    transcripts_path = tmp_path / "transcripts.json"
    policy_path = tmp_path / "scope-policy.json"
    NormalizedStore(
        adapter="test",
        memories=(
            NormalizedMemory(
                id="m1",
                content="User prefers Python.",
                source_refs=(SourceRef(transcript_id="t1", turn_idx=0),),
                provenance_status=ProvenanceStatus.DECLARED,
                scope=MemoryScope(user_id="user-a"),
                active=True,
            ),
        ),
    ).to_json(store_path)
    transcripts = TranscriptSet(
        transcripts=(
            Transcript(
                id="t1",
                turns=(
                    TranscriptTurn(index=0, role="user", content="User prefers Python."),
                ),
            ),
        )
    )
    transcripts_path.write_text(dumps_transcripts(transcripts), encoding="utf-8")
    ScopeIsolationPolicy(
        rules=(
            PrincipalBoundaryRule(
                dimension=ScopeDimension.USER_ID,
                authoritative_source_principal="user-a",
                prohibited_destination_principals=("user-b",),
            ),
        )
    ).to_json(policy_path)
    return store_path, transcripts_path, policy_path


def _aggregate_checker_ids(
    items: tuple[CheckerResult, ...] | tuple[SkippedChecker, ...],
) -> tuple[str, ...]:
    return tuple(item.checker_id for item in items)


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


def test_command_audit_parser_accepts_all_and_preserves_single_choices() -> None:
    parser, audit_parser = _command_parser("audit")
    checker_action = next(
        action for action in audit_parser._actions if action.dest == "checker"
    )

    for checker_id in (*cli.CHECKER_NAMES, "all"):
        args = parser.parse_args(
            ["audit", "--store", "store.json", "--checker", checker_id]
        )
        assert args.checker == checker_id

    assert tuple(checker_action.choices) == (*cli.CHECKER_NAMES, "all")
    assert cli.CHECKER_NAMES == (
        "orphaned_provenance",
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
        "unsupported_claim",
    )
    frozen_parser = cli.build_parser()
    frozen_commands = next(
        action
        for action in frozen_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    frozen_checker_action = next(
        action
        for action in frozen_commands.choices["audit"]._actions
        if action.dest == "checker"
    )
    assert "all" not in frozen_checker_action.choices


def test_command_audit_help_describes_and_displays_all() -> None:
    parser, audit_parser = _command_parser("audit")

    assert "run one checker or all checkers on normalized data" in parser.format_help()
    assert "run one checker or all checkers on normalized data" in audit_parser.format_help()
    assert "unsupported_claim,all" in audit_parser.format_help()


def test_command_audit_parser_still_rejects_unknown_checker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = command.build_parser()

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(
            ["audit", "--store", "store.json", "--checker", "unknown_checker"]
        )

    error = capsys.readouterr().err
    assert raised.value.code == 2
    assert "invalid choice" in error
    assert "Traceback" not in error


def test_aggregate_audit_writes_canonical_json_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = Path("examples/mutation-store.json")

    assert command.main(_audit_arguments(store, "all")) == 0
    captured = capsys.readouterr()
    report = AuditReport.model_validate_json(captured.out)

    assert report.schema_version == "0.1"
    assert _aggregate_checker_ids(report.results) == (
        "redundancy_bloat",
        "stale_active",
    )
    assert _aggregate_checker_ids(report.skipped) == (
        "orphaned_provenance",
        "privacy_scope_violation",
        "unsupported_claim",
    )
    assert report.skipped[0].reasons == (SkipReason.MISSING_TRANSCRIPTS,)
    assert report.skipped[1].reasons == (SkipReason.MISSING_SCOPE_POLICY,)
    assert report.skipped[2].reasons == (
        SkipReason.MISSING_TRANSCRIPTS,
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )
    assert captured.out == report.to_json()
    assert captured.err == ""


def test_aggregate_audit_file_output_is_quiet_and_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Path("examples/mutation-store.json")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert command.main(_audit_arguments(store, "all", output=first)) == 0
    assert command.main(_audit_arguments(store, "all", output=second)) == 0
    captured = capsys.readouterr()

    report = AuditReport.model_validate_json(first.read_text(encoding="utf-8"))
    assert report.schema_version == "0.1"
    assert first.read_bytes() == second.read_bytes()
    assert captured.out == ""
    assert captured.err == ""


def test_fully_configured_aggregate_loads_inputs_and_constructs_model_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, transcripts, policy = _write_complete_aggregate_inputs(tmp_path)
    output = tmp_path / "report.json"
    calls = {
        "store": 0,
        "transcripts": 0,
        "policy": 0,
        "judge": 0,
        "aggregate": 0,
    }
    judge_arguments: list[tuple[str, str, str]] = []
    original_load_store = command.load_store
    original_load_transcripts = command.load_transcripts
    original_load_scope_policy = command.load_scope_policy
    original_run_aggregate_audit = command.run_aggregate_audit

    def load_store_once(path: Path) -> NormalizedStore:
        calls["store"] += 1
        return original_load_store(path)

    def load_transcripts_once(path: Path) -> TranscriptSet:
        calls["transcripts"] += 1
        return original_load_transcripts(path)

    def load_policy_once(path: Path) -> ScopeIsolationPolicy:
        calls["policy"] += 1
        return original_load_scope_policy(path)

    def build_judge(*, model_id: str, revision: str, device: str) -> _FakeAggregateJudge:
        calls["judge"] += 1
        judge_arguments.append((model_id, revision, device))
        return _FakeAggregateJudge(model_id, revision)

    def run_once(
        store_value: NormalizedStore,
        **kwargs: object,
    ) -> AuditReport:
        calls["aggregate"] += 1
        return original_run_aggregate_audit(store_value, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(command, "load_store", load_store_once)
    monkeypatch.setattr(command, "load_transcripts", load_transcripts_once)
    monkeypatch.setattr(command, "load_scope_policy", load_policy_once)
    monkeypatch.setattr(command, "LocalNLISemanticJudge", build_judge)
    monkeypatch.setattr(command, "run_aggregate_audit", run_once)

    assert (
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                transcripts=transcripts,
                scope_policy=policy,
                semantic_model_id="test/model",
                semantic_model_revision="revision-1",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))

    assert calls == {
        "store": 1,
        "transcripts": 1,
        "policy": 1,
        "judge": 1,
        "aggregate": 1,
    }
    assert judge_arguments == [("test/model", "revision-1", "cpu")]
    assert _aggregate_checker_ids(report.results) == cli.CHECKER_NAMES
    assert report.skipped == ()
    assert captured.out == ""
    assert captured.err == ""


def test_complete_semantic_configuration_without_transcripts_does_not_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, policy = _write_complete_aggregate_inputs(tmp_path)
    output = tmp_path / "report.json"

    def fail_model_construction(**_kwargs: object) -> None:
        raise AssertionError("semantic model construction attempted")

    monkeypatch.setattr(command, "LocalNLISemanticJudge", fail_model_construction)

    assert (
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                scope_policy=policy,
                semantic_model_id="test/model",
                semantic_model_revision="revision-1",
            )
        )
        == 0
    )
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))
    unsupported = next(
        item for item in report.skipped if item.checker_id == "unsupported_claim"
    )

    assert unsupported.reasons == (SkipReason.MISSING_TRANSCRIPTS,)
    assert _aggregate_checker_ids(report.results) == (
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
    )


def test_aggregate_without_scope_policy_skips_only_privacy_checker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, transcripts, _ = _write_complete_aggregate_inputs(tmp_path)
    output = tmp_path / "report.json"

    def build_judge(*, model_id: str, revision: str, device: str) -> _FakeAggregateJudge:
        assert device == "cpu"
        return _FakeAggregateJudge(model_id, revision)

    monkeypatch.setattr(command, "LocalNLISemanticJudge", build_judge)

    assert (
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                transcripts=transcripts,
                semantic_model_id="test/model",
                semantic_model_revision="revision-1",
            )
        )
        == 0
    )
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))

    assert _aggregate_checker_ids(report.results) == tuple(
        checker_id
        for checker_id in cli.CHECKER_NAMES
        if checker_id != "privacy_scope_violation"
    )
    assert _aggregate_checker_ids(report.skipped) == ("privacy_scope_violation",)
    assert report.skipped[0].reasons == (SkipReason.MISSING_SCOPE_POLICY,)


def test_aggregate_without_semantic_config_skips_only_unsupported_checker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, transcripts, policy = _write_complete_aggregate_inputs(tmp_path)
    output = tmp_path / "report.json"

    def fail_model_construction(**_kwargs: object) -> None:
        raise AssertionError("semantic model construction attempted")

    monkeypatch.setattr(command, "LocalNLISemanticJudge", fail_model_construction)

    assert (
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                transcripts=transcripts,
                scope_policy=policy,
            )
        )
        == 0
    )
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))

    assert _aggregate_checker_ids(report.results) == cli.CHECKER_NAMES[:-1]
    assert _aggregate_checker_ids(report.skipped) == ("unsupported_claim",)
    assert report.skipped[0].reasons == (
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )


def test_configured_semantic_judge_marker_has_real_identity_and_fails_if_called() -> None:
    marker = command._ConfiguredSemanticJudge(
        model_id="test/model",
        revision="revision-1",
    )

    assert marker.judge_id == "hf-nli:test/model"
    assert marker.judge_version == "revision-1"
    with pytest.raises(AssertionError, match="must not be invoked"):
        marker.judge(premise="Evidence.", hypothesis="Claim.")


@pytest.mark.parametrize(
    ("model_id", "revision", "expected_option"),
    [
        ("test/model", None, "--semantic-model-revision"),
        (None, "revision-1", "--semantic-model-id"),
        ("", "revision-1", "--semantic-model-id"),
        ("test/model", "", "--semantic-model-revision"),
    ],
)
def test_aggregate_rejects_partial_or_blank_semantic_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_id: str | None,
    revision: str | None,
    expected_option: str,
) -> None:
    store = tmp_path / "store.json"
    output = tmp_path / "report.json"
    _write_stale_store(store, stale=False)

    def fail_model_construction(**_kwargs: object) -> None:
        raise AssertionError("semantic model construction attempted")

    monkeypatch.setattr(command, "LocalNLISemanticJudge", fail_model_construction)

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                semantic_model_id=model_id,
                semantic_model_revision=revision,
            )
        )

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert expected_option in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_aggregate_without_semantic_configuration_does_not_load_optional_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store.json"
    output = tmp_path / "report.json"
    _write_stale_store(store, stale=False)
    modules_before = {name: sys.modules.get(name) for name in ("torch", "transformers")}

    def fail_model_construction(**_kwargs: object) -> None:
        raise AssertionError("semantic model construction attempted")

    monkeypatch.setattr(command, "LocalNLISemanticJudge", fail_model_construction)

    assert command.main(_audit_arguments(store, "all", output=output)) == 0
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))

    unsupported = next(
        item for item in report.skipped if item.checker_id == "unsupported_claim"
    )
    assert unsupported.reasons == (
        SkipReason.MISSING_TRANSCRIPTS,
        SkipReason.MISSING_SEMANTIC_CONFIGURATION,
    )
    assert {name: sys.modules.get(name) for name in modules_before} == modules_before


@pytest.mark.parametrize("malformed_input", ["store", "transcripts", "policy"])
def test_aggregate_rejects_malformed_supplied_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    malformed_input: str,
) -> None:
    store = tmp_path / "store.json"
    transcripts = tmp_path / "transcripts.json"
    policy = tmp_path / "scope-policy.json"
    output = tmp_path / "report.json"
    _write_stale_store(store, stale=False)
    transcripts.write_text("{", encoding="utf-8")
    policy.write_text("{", encoding="utf-8")
    if malformed_input == "store":
        store.write_text("{", encoding="utf-8")
        transcripts_argument = None
        policy_argument = None
    else:
        transcripts_argument = transcripts if malformed_input == "transcripts" else None
        policy_argument = policy if malformed_input == "policy" else None

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                transcripts=transcripts_argument,
                scope_policy=policy_argument,
            )
        )

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_aggregate_semantic_model_error_is_a_handled_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, transcripts, policy = _write_complete_aggregate_inputs(tmp_path)
    output = tmp_path / "report.json"

    def fail_model_construction(**_kwargs: object) -> None:
        raise SemanticJudgeError("semantic model unavailable")

    monkeypatch.setattr(command, "LocalNLISemanticJudge", fail_model_construction)

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                transcripts=transcripts,
                scope_policy=policy,
                semantic_model_id="test/model",
                semantic_model_revision="revision-1",
            )
        )

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "semantic model unavailable" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_aggregate_checker_failure_is_not_serialized_as_partial_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "store.json"
    output = tmp_path / "report.json"
    _write_stale_store(store, stale=False)

    def fail_aggregate(*_args: object, **_kwargs: object) -> None:
        raise ValueError("checker failed")

    monkeypatch.setattr(command, "run_aggregate_audit", fail_aggregate)

    with pytest.raises(SystemExit) as raised:
        command.main(_audit_arguments(store, "all", output=output))

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "checker failed" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_aggregate_output_filesystem_error_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store.json"
    output = tmp_path / "missing" / "report.json"
    _write_stale_store(store, stale=False)

    with pytest.raises(SystemExit) as raised:
        command.main(_audit_arguments(store, "all", output=output))

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "No such file or directory" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(None, 0), ("info", 1), ("warning", 1), ("error", 0)],
)
def test_aggregate_warning_finding_obeys_threshold(
    tmp_path: Path,
    threshold: str | None,
    expected: int,
) -> None:
    store = tmp_path / "redundant.json"
    output = tmp_path / "report.json"
    _write_redundant_store(store)

    assert (
        command.main(
            _audit_arguments(store, "all", fail_on=threshold, output=output)
        )
        == expected
    )
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))
    redundancy = next(
        result for result in report.results if result.checker_id == "redundancy_bloat"
    )
    assert len(redundancy.findings) == 1


def test_aggregate_error_gate_preserves_complete_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    _write_stale_store(store)

    assert command.main(_audit_arguments(store, "all", fail_on="error")) == 1
    captured = capsys.readouterr()
    report = AuditReport.model_validate_json(captured.out)

    stale = next(result for result in report.results if result.checker_id == "stale_active")
    assert len(stale.findings) == 1
    assert captured.out == report.to_json()
    assert captured.err == ""


def test_aggregate_error_gate_preserves_complete_file_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    output = tmp_path / "report.json"
    _write_stale_store(store)

    assert (
        command.main(
            _audit_arguments(store, "all", fail_on="error", output=output)
        )
        == 1
    )
    captured = capsys.readouterr()
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))

    stale = next(result for result in report.results if result.checker_id == "stale_active")
    assert len(stale.findings) == 1
    assert captured.out == ""
    assert captured.err == ""


def test_aggregate_skips_do_not_trigger_info_gate(tmp_path: Path) -> None:
    store = tmp_path / "clean.json"
    output = tmp_path / "report.json"
    _write_stale_store(store, stale=False)

    assert (
        command.main(_audit_arguments(store, "all", fail_on="info", output=output))
        == 0
    )
    report = AuditReport.model_validate_json(output.read_text(encoding="utf-8"))
    assert all(not result.findings for result in report.results)
    assert report.skipped


def test_aggregate_gate_does_not_inspect_finding_confidence() -> None:
    report = AuditReport(
        results=(_result_with_confidence(0.0),),
        skipped=tuple(
            SkippedChecker(
                checker_id=checker_id,
                reasons=(SkipReason.MISSING_TRANSCRIPTS,),
            )
            for checker_id in cli.CHECKER_NAMES
            if checker_id != "stale_active"
        ),
    )

    assert command._aggregate_gate_triggered(report, "error") is True


def test_aggregate_sarif_is_rejected_before_writing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store.json"
    output = tmp_path / "report.json"
    sarif_output = tmp_path / "report.sarif"
    _write_stale_store(store, stale=False)

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                sarif_output=sarif_output,
            )
        )

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "--sarif-output" in captured.err
    assert "--checker all" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()
    assert not sarif_output.exists()


@pytest.mark.parametrize("collision", ["store", "transcripts", "policy"])
def test_aggregate_output_cannot_overwrite_any_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    collision: str,
) -> None:
    store, transcripts, policy = _write_complete_aggregate_inputs(tmp_path)
    paths = {"store": store, "transcripts": transcripts, "policy": policy}
    output = paths[collision]
    original = output.read_bytes()

    with pytest.raises(SystemExit) as raised:
        command.main(
            _audit_arguments(
                store,
                "all",
                output=output,
                transcripts=transcripts,
                scope_policy=policy,
            )
        )

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "must not overwrite input files" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert output.read_bytes() == original


def test_single_checker_command_still_emits_established_checker_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "stale.json"
    _write_stale_store(store)
    frozen_args = cli.build_parser().parse_args(
        ["audit", "--store", str(store), "--checker", "stale_active"]
    )
    expected = cli._run_audit(frozen_args)

    assert command.main(_audit_arguments(store, "stale_active")) == 0
    captured = capsys.readouterr()
    result = CheckerResult.model_validate_json(captured.out)

    assert result.schema_version == "0.3"
    assert captured.out == expected
    assert captured.err == ""
