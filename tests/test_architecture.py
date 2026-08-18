import ast
from pathlib import Path

SOURCE_ROOT = Path("src/memlint")


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


def test_part_one_does_not_contain_detector_packages() -> None:
    forbidden = {"checkers", "detectors", "repair", "mutations", "benchmarks"}
    assert not any(path.name in forbidden for path in SOURCE_ROOT.rglob("*"))
