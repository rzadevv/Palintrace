from __future__ import annotations

import ast
from pathlib import Path

PROBE_MODULE = Path("src/memlint/evaluation/retrieval_strong_probe.py")
RUNNER = Path("tools/evaluate_retrieval_strong_probe.py")


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_production_runtime_checker_and_cli_do_not_import_probe() -> None:
    forbidden_roots = (
        Path("src/memlint/retrieval"),
        Path("src/memlint/checkers"),
        Path("src/memlint/semantics"),
    )
    for root in forbidden_roots:
        for path in root.glob("*.py"):
            assert "memlint.evaluation.retrieval_strong_probe" not in _imports(path)
    assert "memlint.evaluation.retrieval_strong_probe" not in _imports(
        Path("src/memlint/cli.py")
    )
    assert "retrieval_strong_probe" not in Path("src/memlint/__init__.py").read_text(
        encoding="utf-8"
    )


def test_probe_models_do_not_import_concrete_retriever_or_production_checker() -> None:
    imports = _imports(PROBE_MODULE)
    assert "memlint.evaluation.experimental_lexical" not in imports
    assert not any(
        module == "memlint.checkers" or module.startswith("memlint.checkers.")
        for module in imports
    )


def test_runner_is_only_concrete_retriever_construction_site() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ExperimentalLexicalRetriever"
    ]
    assert len(calls) == 1
    parents = [
        function
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and calls[0] in tuple(ast.walk(function))
    ]
    assert [function.name for function in parents] == ["_build_retriever"]


def test_runner_preflights_before_probe_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    preflight = next(
        node for node in calls if node.func.attr == "preflight_retrieval_strong_probe"
    )
    execute = next(node for node in calls if node.func.attr == "to_json")
    assert preflight.lineno < execute.lineno


def test_probe_is_not_part_of_benchmark_or_public_cli() -> None:
    benchmark = Path("tests/fixtures/benchmark_v0.1/benchmark.json").read_text(
        encoding="utf-8"
    )
    cli = Path("src/memlint/cli.py").read_text(encoding="utf-8")
    assert "retrieval-shadowing-strong-probe-v0.1" not in benchmark
    assert "retrieval-strong" not in cli
