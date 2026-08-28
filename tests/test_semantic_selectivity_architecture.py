from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import memlint.cli as cli

EVALUATION_MODULE = Path("src/memlint/evaluation/semantic_selectivity.py")
RUNNER = Path("tools/evaluate_semantic_selectivity.py")
SOURCE_ROOT = Path("src/memlint")

FROZEN_HASHES = {
    Path("src/memlint/semantics/local_nli.py"): (
        "aafe1e1a9d662879640285784704cdbfecefec4c25e402fae07101dd7ea087b1"
    ),
    Path("src/memlint/semantics/composition.py"): (
        "cd617221c65bb6a58de7164f7438143d661903f62f57076f4869d3e28d6a7629"
    ),
    Path("src/memlint/checkers/unsupported_claim.py"): (
        "04fd713308d9ed55e79501a31e99904939a2caf8ef90f2187e3fe1f594d09a8a"
    ),
    Path("src/memlint/checkers/unsupported_claim_identity_grounded.py"): (
        "6b742eeff6d4280661adba61ed201b67a6bc25a7d9a2b0c967ebbccd0c3210c5"
    ),
    Path("src/memlint/semantics/identity.py"): (
        "c6b54d0229cb6b87b5e23997685e9855b8b789ea2d68f6e6f07ee45a749f82f9"
    ),
    Path("src/memlint/semantics/identity_source.py"): (
        "058590ef7258b9f611de486b5130b866893f0e4a2c1091d5ecd0d0a465463e87"
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
        module == "memlint.semantics" or module.startswith("memlint.semantics.")
        for module in evaluation_imports
    )
    assert not any(
        module.startswith(("memlint.checkers", "memlint.adapters", "memlint.retrieval"))
        for module in evaluation_imports
    )
    violations = [
        str(path)
        for root_name in ("semantics", "checkers", "adapters", "retrieval")
        for path in (SOURCE_ROOT / root_name).rglob("*.py")
        if any(
            module == "memlint.evaluation.semantic_selectivity"
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
