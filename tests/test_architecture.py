import ast
from pathlib import Path

SOURCE_ROOT = Path("src/memlint")
CHECKER_ROOT = SOURCE_ROOT / "checkers"


def test_backend_sdks_are_imported_only_inside_their_adapters() -> None:
    backend_modules = {"mem0", "graphiti_core", "letta_client"}
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.parent.name == "adapters":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            if roots & backend_modules:
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_generic_code_does_not_read_raw_attributes() -> None:
    """Future generic/checker modules must use typed normalized fields, never ``.raw``."""

    allowed = {SOURCE_ROOT / "models" / "memory.py"}
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path.parent.name == "adapters" or path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "raw":
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_part_three_does_not_contain_later_phase_packages() -> None:
    forbidden = {"detectors", "repair", "benchmarks"}
    assert not any(path.name in forbidden for path in SOURCE_ROOT.rglob("*"))


def test_checker_package_cannot_import_mutation_code_or_gold_models() -> None:
    forbidden_names = {"MutationManifest", "GoldLabel", "MutationRequest"}
    violations: list[str] = []
    for path in CHECKER_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("memlint.mutations") for alias in node.names):
                    violations.append(f"{path}:{node.lineno}:mutation import")
            elif isinstance(node, ast.ImportFrom) and node.module:
                is_absolute_mutation_import = node.module.startswith("memlint.mutations")
                is_relative_mutation_import = node.level > 0 and node.module.startswith("mutations")
                imports_gold_model = any(alias.name in forbidden_names for alias in node.names)
                if (
                    is_absolute_mutation_import
                    or is_relative_mutation_import
                    or imports_gold_model
                ):
                    violations.append(f"{path}:{node.lineno}:mutation import")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{path}:{node.lineno}:{node.id}")
    assert violations == []


def test_checker_package_does_not_read_raw_attributes() -> None:
    violations: list[str] = []
    for path in CHECKER_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "raw":
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_part_three_has_only_the_orphaned_provenance_checker_module() -> None:
    modules = {path.name for path in CHECKER_ROOT.glob("*.py")}
    assert modules == {"__init__.py", "base.py", "models.py", "orphaned_provenance.py"}
