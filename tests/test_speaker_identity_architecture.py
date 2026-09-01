from __future__ import annotations

import ast
from pathlib import Path

import memlint.semantics as semantics

IDENTITY_MODULE = Path("src/memlint/semantics/identity.py")
CANDIDATE_CHECKER_MODULE = Path(
    "src/memlint/checkers/unsupported_claim_identity_grounded.py"
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return tuple(found)


def test_identity_module_has_no_evaluation_checker_or_adapter_dependency() -> None:
    imports = _imports(IDENTITY_MODULE)
    forbidden = ("memlint.evaluation", "memlint.checkers", "memlint.adapters")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in forbidden
    )


def test_identity_resolution_has_no_claim_text_scope_raw_role_or_metadata_dependency() -> None:
    tree = ast.parse(IDENTITY_MODULE.read_text(encoding="utf-8"), filename=str(IDENTITY_MODULE))
    accessed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert accessed_attributes.isdisjoint(
        {
            "content",
            "raw",
            "scope",
            "role",
            "metadata",
            "user_id",
            "agent_id",
            "session_id",
        }
    )


def test_only_identity_grounded_candidate_checker_depends_on_identity_contract() -> None:
    forbidden_names = {
        "SpeakerIdentityBinding",
        "SpeakerIdentityBindings",
        "SpeakerIdentityError",
        "SpeakerIdentityResolution",
        "SpeakerIdentityResolutionStatus",
        "build_speaker_grounded_premise",
        "resolve_speaker_identity",
    }
    identity_importers: list[Path] = []
    for path in Path("src/memlint/checkers").glob("*.py"):
        imports = _imports(path)
        if "memlint.semantics.identity" in imports:
            identity_importers.append(path)
        if path == CANDIDATE_CHECKER_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert forbidden_names.isdisjoint(referenced_names)
    assert identity_importers == [CANDIDATE_CHECKER_MODULE]


def test_candidate_checker_has_no_evaluation_dependency() -> None:
    assert not any(
        module == "memlint.evaluation" or module.startswith("memlint.evaluation.")
        for module in _imports(CANDIDATE_CHECKER_MODULE)
    )


def test_adapters_do_not_depend_on_semantics() -> None:
    for path in Path("src/memlint/adapters").glob("*.py"):
        assert not any(
            module == "memlint.semantics" or module.startswith("memlint.semantics.")
            for module in _imports(path)
        )


def test_identity_contract_is_available_from_public_semantics_api() -> None:
    assert semantics.SpeakerIdentityBinding.__module__ == "memlint.semantics.identity"
    assert semantics.SpeakerIdentityBindings.__module__ == "memlint.semantics.identity"
    assert semantics.SpeakerIdentityResolution.__module__ == "memlint.semantics.identity"
    assert semantics.SpeakerIdentityResolutionStatus.__module__ == "memlint.semantics.identity"
