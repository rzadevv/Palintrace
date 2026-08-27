import ast
from pathlib import Path

import memlint.cli as cli

SOURCE_ROOT = Path("src/memlint")
EVALUATION_ROOT = SOURCE_ROOT / "evaluation"
DETECTOR_ROOTS = tuple(
    SOURCE_ROOT / package
    for package in ("checkers", "mutations", "retrieval", "semantics")
)


def _absolute_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return tuple(imports)


def test_evaluation_package_contains_frozen_accounting_and_execution_layers() -> None:
    assert {path.name for path in EVALUATION_ROOT.glob("*.py")} == {
        "__init__.py",
        "benchmark.py",
        "clean_control.py",
        "execution.py",
        "execution_models.py",
        "experimental_lexical.py",
        "identity_grounded_heldout.py",
        "identity_probe.py",
        "models.py",
        "mutation.py",
        "preflight.py",
        "retrieval.py",
        "retrieval_strong_probe.py",
    }


def test_detector_and_runtime_packages_do_not_import_evaluation() -> None:
    violations = [
        f"{path}:{module}"
        for root in DETECTOR_ROOTS
        for path in root.rglob("*.py")
        for module in _absolute_imports(path)
        if module == "memlint.evaluation" or module.startswith("memlint.evaluation.")
    ]
    assert violations == []


def test_checker_interfaces_have_no_gold_evaluation_parameters_or_symbols() -> None:
    forbidden_names = {
        "MutationManifest",
        "GoldLabel",
        "base_store_status",
        "mutation_id",
        "created_memory_ids",
        "modified_memory_ids",
    }
    violations: list[str] = []
    for path in (SOURCE_ROOT / "checkers").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{path}:{node.lineno}:{node.id}")
            elif isinstance(node, ast.arg) and node.arg in forbidden_names:
                violations.append(f"{path}:{node.lineno}:{node.arg}")
            elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                violations.append(f"{path}:{node.lineno}:{node.attr}")
    assert violations == []


def test_evaluation_code_has_no_raw_access() -> None:
    violations: list[str] = []
    for path in EVALUATION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "raw":
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_retrieval_summary_module_has_no_manifest_or_mutation_dependency() -> None:
    path = EVALUATION_ROOT / "retrieval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_names = {"MutationManifest", "RetrievalProbe", "GoldLabel"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("memlint.mutations") for alias in node.names):
                violations.append(f"{node.lineno}:mutation import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("memlint.mutations"):
                violations.append(f"{node.lineno}:mutation import")
            if any(alias.name in forbidden_names for alias in node.names):
                violations.append(f"{node.lineno}:gold model import")
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            violations.append(f"{node.lineno}:{node.id}")
    assert violations == []


def test_no_evaluation_or_benchmark_cli_and_audit_accepts_no_manifest() -> None:
    parser = cli.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert set(command_action.choices) == {
        "dump",
        "mutate",
        "audit",
        "retrieval-audit",
    }
    audit_parser = command_action.choices["audit"]
    audit_arguments = {action.dest for action in audit_parser._actions}
    assert "manifest" not in audit_arguments
    assert "mutation_id" not in audit_arguments
    assert "gold_label" not in audit_arguments


def test_cli_has_no_evaluation_package_dependency() -> None:
    assert not any(
        module == "memlint.evaluation" or module.startswith("memlint.evaluation.")
        for module in _absolute_imports(SOURCE_ROOT / "cli.py")
    )
