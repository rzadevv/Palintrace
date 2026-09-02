from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import palintrace.cli as cli

EVALUATION_MODULE = Path("src/palintrace/evaluation/semantic_selectivity.py")
RUNNER = Path("tools/evaluate_semantic_selectivity.py")
SOURCE_ROOT = Path("src/palintrace")

FROZEN_HASHES = {
    Path("src/palintrace/semantics/local_nli.py"): (
        "41b853a6540b6bc18b1abd644c3477c3c05e28978c8e6ec3ef861ea9d37f8f19"
    ),
    Path("src/palintrace/semantics/composition.py"): (
        "5c710e879126f80153ee80be3b20392c7a3bcea3f1c8a7089fba761876cb87e3"
    ),
    Path("src/palintrace/checkers/unsupported_claim.py"): (
        "604d08b766cf901475be78258c31162d76d748759db2f205049bbac285fa6cdc"
    ),
    Path("src/palintrace/checkers/unsupported_claim_identity_grounded.py"): (
        "0bae646f781a63a90371c45411696c780a2ced6f6d1a0d2a3c6343b46aa06d98"
    ),
    Path("src/palintrace/semantics/identity.py"): (
        "23cd31a3d75fea5bd260dc5f8f9fd87c39dd930d851d1a44477c77a9423bd4f2"
    ),
    Path("src/palintrace/semantics/identity_source.py"): (
        "7fde1d795c74b2ef05b2fc9842047e50d075b886f9298c862f635080b8e34179"
    ),
}


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(imports)


def test_frozen_semantic_and_identity_predecessors_are_byte_identical() -> None:
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FROZEN_HASHES
    } == FROZEN_HASHES


def test_new_probe_is_evaluation_only_and_production_does_not_import_it() -> None:
    evaluation_imports = _imports(EVALUATION_MODULE)
    assert any(
        module == "palintrace.semantics" or module.startswith("palintrace.semantics.")
        for module in evaluation_imports
    )
    assert not any(
        module.startswith(("palintrace.checkers", "palintrace.adapters", "palintrace.retrieval"))
        for module in evaluation_imports
    )
    violations = [
        str(path)
        for root_name in ("semantics", "checkers", "adapters", "retrieval")
        for path in (SOURCE_ROOT / root_name).rglob("*.py")
        if any(
            module == "palintrace.evaluation.semantic_selectivity"
            for module in _imports(path)
        )
    ]
    assert violations == []


def test_runner_is_not_public_cli_and_exposes_no_scientific_override() -> None:
    parser = cli.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert set(command_action.choices) == {"dump", "mutate", "audit", "retrieval-audit"}
    runner_tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    option_strings = {
        argument.value
        for call in ast.walk(runner_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        for argument in call.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    assert option_strings == {"--output"}


def test_runner_preflights_before_model_construction() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
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
        and node.func.attr == "preflight_semantic_selectivity"
    )
    model = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_judge"
    )
    assert preflight.lineno < model.lineno
