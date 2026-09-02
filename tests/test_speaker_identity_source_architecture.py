from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import palintrace.semantics as semantics

SOURCE_MODULE = Path("src/palintrace/semantics/identity_source.py")
CANDIDATE_MODULE = Path(
    "src/palintrace/checkers/unsupported_claim_identity_grounded.py"
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
    forbidden = ("palintrace.adapters", "palintrace.checkers", "palintrace.evaluation")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imports
        for prefix in forbidden
    )


def test_no_adapter_or_candidate_depends_on_identity_source_admission() -> None:
    for path in Path("src/palintrace/adapters").glob("*.py"):
        assert "palintrace.semantics.identity_source" not in _imports(path)
    assert "palintrace.semantics.identity_source" not in _imports(CANDIDATE_MODULE)


def test_identity_source_contract_is_available_without_exporting_candidate() -> None:
    assert (
        semantics.SpeakerIdentitySourceAssertion.__module__
        == "palintrace.semantics.identity_source"
    )
    assert (
        semantics.SpeakerIdentitySourceAssertions.__module__
        == "palintrace.semantics.identity_source"
    )
    assert semantics.SpeakerIdentityTrust.__module__ == "palintrace.semantics.identity_source"
    import palintrace.checkers as checkers

    assert not hasattr(checkers, "IdentityGroundedUnsupportedClaimChecker")


def test_cli_public_checker_and_transcript_contracts_remain_byte_frozen() -> None:
    expected = {
        Path("src/palintrace/cli.py"): (
            "620aba148c80f6b6468156876eb3c9a4e6cc6d4b12b4d5325f8615823a2c9c57"
        ),
        Path("src/palintrace/checkers/__init__.py"): (
            "c2ff4192aef0d5830e323f476e3b22db66e1cf6283e35ec0ee88c7c80602379a"
        ),
        Path("src/palintrace/models/transcript.py"): (
            "07af2300089581966435c844c4ab944184e11eeff1fb467ea72900ae07f6639a"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_all_scientific_predecessors_remain_byte_frozen() -> None:
    expected = {
        Path("src/palintrace/checkers/unsupported_claim.py"): (
            "604d08b766cf901475be78258c31162d76d748759db2f205049bbac285fa6cdc"
        ),
        CANDIDATE_MODULE: (
            "0bae646f781a63a90371c45411696c780a2ced6f6d1a0d2a3c6343b46aa06d98"
        ),
        Path("src/palintrace/semantics/identity.py"): (
            "23cd31a3d75fea5bd260dc5f8f9fd87c39dd930d851d1a44477c77a9423bd4f2"
        ),
        Path("src/palintrace/semantics/local_nli.py"): (
            "41b853a6540b6bc18b1abd644c3477c3c05e28978c8e6ec3ef861ea9d37f8f19"
        ),
        Path("src/palintrace/semantics/composition.py"): (
            "5c710e879126f80153ee80be3b20392c7a3bcea3f1c8a7089fba761876cb87e3"
        ),
        Path("tests/fixtures/unsupported_identity_probe_v0.1.json"): (
            "4cbc1dc77b1d6a315992c2b564e438f77d6ff6df3f5dfa621d0542cdf7ea7beb"
        ),
        Path("tests/fixtures/unsupported_identity_grounded_heldout_v0.1.json"): (
            "a0384e2d4e5d7764c45c87e1c729762cbd2714ced2faa3cb7e36a2b50283169b"
        ),
        Path("src/palintrace/evaluation/identity_grounded_heldout.py"): (
            "da573daa40de94036f6916ab882e2ab61db52eba3a1a642f5a804fe650d8bb00"
        ),
        Path("tests/fixtures/benchmark_v0.1.sha256.json"): (
            "028fe4e096adc556b4d23bd89c6f5c79f635cbcfe327ad94a2a5a2e7794a659d"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected
