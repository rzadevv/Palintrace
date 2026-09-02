import argparse
import ast
import hashlib
import json
from pathlib import Path

import pytest

import palintrace.cli as cli
import palintrace.retrieval as retrieval
from palintrace.checkers import CheckerResult
from palintrace.retrieval import (
    RetrievalHit,
    RetrievalObservation,
    RetrievalSufficiencyPolicy,
    RetrievalUsage,
)
from palintrace.taxonomy import DefectClass

QUERY_TEXT = "Which preference should the recorded retrieval have returned?"


def _observation(
    *,
    expected_memory_ids: tuple[str, ...] = ("m1",),
    hits: tuple[RetrievalHit, ...] = (),
    top_k: int = 3,
) -> RetrievalObservation:
    return RetrievalObservation(
        request_id="case-1",
        query_sha256=hashlib.sha256(QUERY_TEXT.encode("utf-8")).hexdigest(),
        expected_memory_ids=expected_memory_ids,
        top_k=top_k,
        retriever_id="recorded-retriever",
        retriever_version="1",
        hits=hits,
        usage=RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    )


def _write_observation(path: Path, observation: RetrievalObservation) -> bytes:
    payload = observation.to_json().encode("utf-8")
    path.write_bytes(payload)
    return payload


def _arguments(
    observation_path: Path,
    *,
    policy: str = "all_expected",
    output: Path | None = None,
) -> list[str]:
    arguments = [
        "retrieval-audit",
        "--observation",
        str(observation_path),
        "--policy",
        policy,
    ]
    if output is not None:
        arguments.extend(("--output", str(output)))
    return arguments


def test_retrieval_audit_parser_has_exact_required_shape() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    retrieval_parser = subparsers.choices["retrieval-audit"]
    actions = {action.dest: action for action in retrieval_parser._actions if action.dest != "help"}

    assert set(actions) == {"observation", "policy", "output"}
    assert actions["observation"].required is True
    assert actions["policy"].required is True
    assert actions["policy"].default is None
    assert actions["policy"].type is RetrievalSufficiencyPolicy
    assert tuple(actions["policy"].choices) == tuple(RetrievalSufficiencyPolicy)
    assert actions["output"].required is False


def test_insufficient_recorded_observation_writes_one_frozen_finding(
    tmp_path: Path,
) -> None:
    observation_path = tmp_path / "observation.json"
    output_path = tmp_path / "result.json"
    _write_observation(observation_path, _observation())

    exit_code = cli.main(_arguments(observation_path, output=output_path))
    result = CheckerResult.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert result.checker_id == "retrieval_shadowing"
    assert result.checker_version == "1.0"
    assert result.defect_class is DefectClass.RETRIEVAL_SHADOWING
    assert len(result.findings) == 1
    assert result.findings[0].memory_ids == ("m1",)
    assert result.findings[0].confidence == 1.0


def test_sufficient_recorded_observation_writes_zero_findings(tmp_path: Path) -> None:
    observation_path = tmp_path / "observation.json"
    output_path = tmp_path / "result.json"
    _write_observation(
        observation_path,
        _observation(hits=(RetrievalHit(memory_id="m1", rank=1),)),
    )

    assert cli.main(_arguments(observation_path, output=output_path)) == 0
    result = CheckerResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert result.checker_id == "retrieval_shadowing"
    assert result.findings == ()


def test_cli_preserves_explicit_all_vs_any_policy_divergence(tmp_path: Path) -> None:
    observation_path = tmp_path / "observation.json"
    all_output = tmp_path / "all.json"
    any_output = tmp_path / "any.json"
    _write_observation(
        observation_path,
        _observation(
            expected_memory_ids=("m1", "m2"),
            hits=(RetrievalHit(memory_id="m1", rank=1),),
        ),
    )

    assert cli.main(
        _arguments(observation_path, policy="all_expected", output=all_output)
    ) == 0
    assert cli.main(
        _arguments(observation_path, policy="any_expected", output=any_output)
    ) == 0
    all_result = CheckerResult.model_validate_json(all_output.read_text(encoding="utf-8"))
    any_result = CheckerResult.model_validate_json(any_output.read_text(encoding="utf-8"))

    assert len(all_result.findings) == 1
    assert all_result.findings[0].memory_ids == ("m2",)
    assert any_result.findings == ()


def test_stdout_is_only_json_and_matches_file_output_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    output_path = tmp_path / "result.json"
    _write_observation(observation_path, _observation())

    assert cli.main(_arguments(observation_path)) == 0
    captured_stdout = capsys.readouterr()
    stdout_bytes = captured_stdout.out.encode("utf-8")
    CheckerResult.model_validate_json(captured_stdout.out)
    assert captured_stdout.err == ""

    assert cli.main(_arguments(observation_path, output=output_path)) == 0
    captured_file = capsys.readouterr()
    assert captured_file.out == ""
    assert captured_file.err == ""
    assert output_path.read_bytes() == stdout_bytes


def test_stdout_result_contains_no_query_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    _write_observation(observation_path, _observation())

    assert cli.main(_arguments(observation_path)) == 0
    output = capsys.readouterr().out
    CheckerResult.model_validate_json(output)
    assert QUERY_TEXT not in output


@pytest.mark.parametrize("missing_argument", ["observation", "policy"])
def test_required_retrieval_audit_arguments_have_no_defaults(
    missing_argument: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    _write_observation(observation_path, _observation())
    arguments = ["retrieval-audit"]
    if missing_argument != "observation":
        arguments.extend(("--observation", str(observation_path)))
    if missing_argument != "policy":
        arguments.extend(("--policy", "all_expected"))

    with pytest.raises(SystemExit, match="2"):
        cli.main(arguments)

    error = capsys.readouterr().err
    assert f"--{missing_argument}" in error
    assert "Traceback" not in error


def test_invalid_json_fails_cleanly_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    output_path = tmp_path / "result.json"
    observation_path.write_text('{"broken":', encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        cli.main(_arguments(observation_path, output=output_path))

    error = capsys.readouterr().err
    assert "invalid RetrievalObservation JSON" in error
    assert "Traceback" not in error
    assert not output_path.exists()


def test_invalid_observation_schema_fails_cleanly_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    output_path = tmp_path / "result.json"
    payload = _observation().model_dump(mode="json")
    payload["top_k"] = 0
    observation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        cli.main(_arguments(observation_path, output=output_path))

    error = capsys.readouterr().err
    assert "invalid RetrievalObservation JSON" in error
    assert "Traceback" not in error
    assert not output_path.exists()


def test_invalid_policy_is_rejected_by_argparse(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    _write_observation(observation_path, _observation())

    with pytest.raises(SystemExit, match="2"):
        cli.main(_arguments(observation_path, policy="strict"))

    error = capsys.readouterr().err
    assert "invalid RetrievalSufficiencyPolicy value" in error
    assert "all_expected" in error
    assert "any_expected" in error
    assert "Traceback" not in error


def test_missing_observation_file_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(SystemExit, match="2"):
        cli.main(_arguments(missing_path))

    error = capsys.readouterr().err
    assert "No such file or directory" in error
    assert "Traceback" not in error


def test_output_cannot_overwrite_observation_and_input_is_preserved(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    original = _write_observation(observation_path, _observation())

    with pytest.raises(SystemExit, match="2"):
        cli.main(_arguments(observation_path, output=observation_path))

    error = capsys.readouterr().err
    assert "retrieval audit output must not overwrite the observation input" in error
    assert "Traceback" not in error
    assert observation_path.read_bytes() == original


@pytest.mark.parametrize(
    "forbidden_argument",
    [
        "--store",
        "--transcripts",
        "--query",
        "--expected-id",
        "--expected-memory-id",
        "--target-id",
        "--top-k",
        "--retriever-id",
        "--retriever-version",
        "--retrieval-calls",
        "--candidate-count",
        "--manifest",
        "--mutation-id",
        "--distractor-id",
        "--gold-label",
        "--semantic-model-id",
        "--semantic-model-revision",
    ],
)
def test_retrieval_audit_rejects_every_out_of_scope_argument(
    forbidden_argument: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation_path = tmp_path / "observation.json"
    _write_observation(observation_path, _observation())

    with pytest.raises(SystemExit, match="2"):
        cli.main([*_arguments(observation_path), forbidden_argument, "value"])

    error = capsys.readouterr().err
    assert "unrecognized arguments" in error
    assert forbidden_argument in error


def test_cli_calls_projection_once_with_validated_observation_and_enum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_path = tmp_path / "observation.json"
    observation = _observation()
    _write_observation(observation_path, observation)
    original_projection = cli.project_retrieval_shadowing_result
    calls: list[tuple[RetrievalObservation, RetrievalSufficiencyPolicy]] = []

    def record_projection(
        loaded_observation: RetrievalObservation,
        *,
        policy: RetrievalSufficiencyPolicy,
    ) -> CheckerResult:
        calls.append((loaded_observation, policy))
        return original_projection(loaded_observation, policy=policy)

    monkeypatch.setattr(cli, "project_retrieval_shadowing_result", record_projection)

    assert cli.main(_arguments(observation_path, policy="any_expected")) == 0
    assert calls == [(observation, RetrievalSufficiencyPolicy.ANY_EXPECTED)]


def test_cli_never_executes_runtime_retrieval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_path = tmp_path / "observation.json"
    _write_observation(observation_path, _observation())

    def unexpected_execution(**kwargs: object) -> None:
        pytest.fail(f"recorded retrieval CLI must not execute retrieval: {kwargs}")

    monkeypatch.setattr(retrieval, "run_retrieval_audit", unexpected_execution)

    assert cli.main(_arguments(observation_path)) == 0
    CheckerResult.model_validate_json(capsys.readouterr().out)


def test_retrieval_cli_handler_delegates_without_projection_logic() -> None:
    tree = ast.parse(Path("src/palintrace/cli.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_recorded_retrieval_audit"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    assert sum(
        isinstance(call.func, ast.Name)
        and call.func.id == "project_retrieval_shadowing_result"
        for call in calls
    ) == 1
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"hits", "expected_memory_ids", "retrieve"}
        for node in ast.walk(function)
    )


def test_static_audit_choices_remain_exact_and_retrieval_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.CHECKER_NAMES == (
        "orphaned_provenance",
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
        "unsupported_claim",
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--checker",
                "retrieval_shadowing",
            ]
        )

    assert "invalid choice" in capsys.readouterr().err
