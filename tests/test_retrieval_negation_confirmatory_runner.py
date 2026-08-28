from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tools.evaluate_retrieval_negation_confirmatory as runner


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
        "--condition",
    ):
        assert forbidden not in output


def test_bad_fixture_fails_before_retriever_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_fixture = tmp_path / "negation-confirmatory.json"
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
    output = tmp_path / "confirmatory-result.json"
    assert runner._validate_output_path(output) == output.resolve()
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        runner._validate_output_path(output)
    with pytest.raises(ValueError, match="outside the repository"):
        runner._validate_output_path(runner.REPOSITORY_ROOT / "confirmatory-result.json")


def test_store_construction_reuses_one_shared_four_memory_baseline() -> None:
    spec = runner.probe.load_retrieval_negation_confirmatory(
        runner.FROZEN_FIXTURE_PATH
    )
    scenario = spec.scenarios[0]
    baseline = runner._build_store(scenario, None)
    assert len(baseline.memories) == 4
    for condition in runner.probe.CONDITION_ORDER:
        mutated = runner._build_store(scenario, condition)
        assert len(mutated.memories) == 12
        assert mutated.memories[:4] == baseline.memories
        assert {memory.scope.user_id for memory in mutated.memories} == {
            scenario.scope_user_id
        }


def test_runner_preflights_before_future_execution() -> None:
    path = Path("tools/evaluate_retrieval_negation_confirmatory.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    preflight = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "preflight_retrieval_negation_confirmatory"
    )
    execute = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_execute_probe"
    )
    assert preflight.lineno < execute.lineno


def test_baseline_call_is_syntactically_outside_condition_loop() -> None:
    path = Path("tools/evaluate_retrieval_negation_confirmatory.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_scenario"
    )
    loop = next(node for node in execute.body if isinstance(node, ast.For))
    audit_calls = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_retrieval_audit"
    ]
    assert len(audit_calls) == 2
    assert sum(call in tuple(ast.walk(loop)) for call in audit_calls) == 1
