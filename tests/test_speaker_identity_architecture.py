from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import palintrace.semantics as semantics

IDENTITY_MODULE = Path("src/palintrace/semantics/identity.py")
CANDIDATE_CHECKER_MODULE = Path(
    "src/palintrace/checkers/unsupported_claim_identity_grounded.py"
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
    forbidden = ("palintrace.evaluation", "palintrace.checkers", "palintrace.adapters")
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
    for path in Path("src/palintrace/checkers").glob("*.py"):
        imports = _imports(path)
        if "palintrace.semantics.identity" in imports:
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
        module == "palintrace.evaluation" or module.startswith("palintrace.evaluation.")
        for module in _imports(CANDIDATE_CHECKER_MODULE)
    )


def test_part_6g_predecessor_modules_are_byte_frozen() -> None:
    expected = {
        Path("src/palintrace/checkers/unsupported_claim.py"): (
            "604d08b766cf901475be78258c31162d76d748759db2f205049bbac285fa6cdc"
        ),
        IDENTITY_MODULE: "23cd31a3d75fea5bd260dc5f8f9fd87c39dd930d851d1a44477c77a9423bd4f2",
        Path("src/palintrace/semantics/local_nli.py"): (
            "41b853a6540b6bc18b1abd644c3477c3c05e28978c8e6ec3ef861ea9d37f8f19"
        ),
        Path("src/palintrace/semantics/composition.py"): (
            "5c710e879126f80153ee80be3b20392c7a3bcea3f1c8a7089fba761876cb87e3"
        ),
    }
    actual = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    }
    assert actual == expected


def test_adapters_do_not_depend_on_semantics() -> None:
    for path in Path("src/palintrace/adapters").glob("*.py"):
        assert not any(
            module == "palintrace.semantics" or module.startswith("palintrace.semantics.")
            for module in _imports(path)
        )


def test_identity_contract_is_available_from_public_semantics_api() -> None:
    assert semantics.SpeakerIdentityBinding.__module__ == "palintrace.semantics.identity"
    assert semantics.SpeakerIdentityBindings.__module__ == "palintrace.semantics.identity"
    assert semantics.SpeakerIdentityResolution.__module__ == "palintrace.semantics.identity"
    assert semantics.SpeakerIdentityResolutionStatus.__module__ == "palintrace.semantics.identity"
