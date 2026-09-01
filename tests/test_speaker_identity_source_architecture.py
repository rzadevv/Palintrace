from __future__ import annotations

import ast
from pathlib import Path

import memlint.semantics as semantics

SOURCE_MODULE = Path("src/memlint/semantics/identity_source.py")
CANDIDATE_MODULE = Path("src/memlint/checkers/unsupported_claim_identity_grounded.py")


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_identity_source_contract_has_no_adapter_checker_or_evaluation_dependency() -> None:
    imports = _imports(SOURCE_MODULE)
    forbidden = ("memlint.adapters", "memlint.checkers", "memlint.evaluation")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in forbidden
    )


def test_no_adapter_or_candidate_depends_on_identity_source_admission() -> None:
    for path in Path("src/memlint/adapters").glob("*.py"):
        assert "memlint.semantics.identity_source" not in _imports(path)
    assert "memlint.semantics.identity_source" not in _imports(CANDIDATE_MODULE)


def test_identity_source_contract_is_available_without_exporting_candidate() -> None:
    assert (
        semantics.SpeakerIdentitySourceAssertion.__module__
        == "memlint.semantics.identity_source"
    )
    assert (
        semantics.SpeakerIdentitySourceAssertions.__module__
        == "memlint.semantics.identity_source"
    )
    assert semantics.SpeakerIdentityTrust.__module__ == "memlint.semantics.identity_source"
    import memlint.checkers as checkers

    assert not hasattr(checkers, "IdentityGroundedUnsupportedClaimChecker")
