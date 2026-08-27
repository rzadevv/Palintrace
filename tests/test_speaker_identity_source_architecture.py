from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import memlint.semantics as semantics

SOURCE_MODULE = Path("src/memlint/semantics/identity_source.py")
CANDIDATE_MODULE = Path(
    "src/memlint/checkers/unsupported_claim_identity_grounded.py"
)


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


def test_cli_public_checker_and_transcript_contracts_remain_byte_frozen() -> None:
    expected = {
        Path("src/memlint/cli.py"): (
            "75379bfe370a8d56573bb3f6d022ed01c180cb00d84edf097309871df1dd51ca"
        ),
        Path("src/memlint/checkers/__init__.py"): (
            "b1a66c0ccc12182cc041f0baae0ad5f9267fb13a4f3be0a08c6dc11008a67b05"
        ),
        Path("src/memlint/models/transcript.py"): (
            "07af2300089581966435c844c4ab944184e11eeff1fb467ea72900ae07f6639a"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_all_scientific_predecessors_remain_byte_frozen() -> None:
    expected = {
        Path("src/memlint/checkers/unsupported_claim.py"): (
            "04fd713308d9ed55e79501a31e99904939a2caf8ef90f2187e3fe1f594d09a8a"
        ),
        CANDIDATE_MODULE: (
            "6b742eeff6d4280661adba61ed201b67a6bc25a7d9a2b0c967ebbccd0c3210c5"
        ),
        Path("src/memlint/semantics/identity.py"): (
            "c6b54d0229cb6b87b5e23997685e9855b8b789ea2d68f6e6f07ee45a749f82f9"
        ),
        Path("src/memlint/semantics/local_nli.py"): (
            "aafe1e1a9d662879640285784704cdbfecefec4c25e402fae07101dd7ea087b1"
        ),
        Path("src/memlint/semantics/composition.py"): (
            "cd617221c65bb6a58de7164f7438143d661903f62f57076f4869d3e28d6a7629"
        ),
        Path("tests/fixtures/unsupported_identity_probe_v0.1.json"): (
            "4cbc1dc77b1d6a315992c2b564e438f77d6ff6df3f5dfa621d0542cdf7ea7beb"
        ),
        Path("tests/fixtures/unsupported_identity_grounded_heldout_v0.1.json"): (
            "a0384e2d4e5d7764c45c87e1c729762cbd2714ced2faa3cb7e36a2b50283169b"
        ),
        Path("src/memlint/evaluation/identity_grounded_heldout.py"): (
            "0bd4746d72cd0b082a08d82000c2b31bc273fca286d957e47f5df57aff86ebf0"
        ),
        Path("tests/fixtures/benchmark_v0.1.sha256.json"): (
            "de4bb8c2076a2c89b7e2df95518ef5588934644b711119fccc8727e0e9ac73fb"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected
