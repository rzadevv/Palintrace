from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import palintrace.evaluation.identity_grounded_heldout as heldout

EVALUATION_MODULE = Path("src/palintrace/evaluation/identity_grounded_heldout.py")
RUNNER = Path("tools/run_identity_grounded_heldout_v0_1.py")


def _imports(path: Path) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, node.lineno))
    return tuple(imports)


def test_phase_a_manifest_verifies_all_frozen_bytes() -> None:
    manifest = heldout.validate_phase_a_manifest(Path.cwd())
    assert len(manifest.files) == 10


def test_evaluation_is_isolated_from_public_and_cli_dispatch() -> None:
    public_checkers = Path("src/palintrace/checkers/__init__.py").read_text(encoding="utf-8")
    cli = Path("src/palintrace/cli.py").read_text(encoding="utf-8")
    benchmark = Path("src/palintrace/evaluation/benchmark.py").read_text(encoding="utf-8")
    candidate = "IdentityGroundedUnsupportedClaimChecker"
    assert candidate not in public_checkers
    assert candidate not in cli
    assert "unsupported_claim_identity_grounded" not in cli
    assert candidate not in benchmark


def test_local_nli_construction_is_lazy_and_runner_only() -> None:
    evaluation_source = EVALUATION_MODULE.read_text(encoding="utf-8")
    assert "LocalNLISemanticJudge" not in evaluation_source
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "LocalNLISemanticJudge" for alias in node.names)
    ]
    assert len(imports) == 1
    parent_functions = [
        function
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and imports[0] in tuple(ast.walk(function))
    ]
    assert [function.name for function in parent_functions] == ["_build_judge"]


def test_no_production_or_frozen_development_module_imports_heldout_evaluation() -> None:
    forbidden_roots = (
        Path("src/palintrace/checkers"),
        Path("src/palintrace/adapters"),
        Path("src/palintrace/semantics"),
    )
    for root in forbidden_roots:
        for path in root.glob("*.py"):
            assert all(
                module != "palintrace.evaluation.identity_grounded_heldout"
                for module, _line in _imports(path)
            )
    for path in (
        Path("src/palintrace/evaluation/benchmark.py"),
        Path("src/palintrace/evaluation/identity_probe.py"),
        Path("tools/run_benchmark_v0_1.py"),
        Path("tools/evaluate_identity_grounding.py"),
    ):
        assert all(
            module != "palintrace.evaluation.identity_grounded_heldout"
            for module, _line in _imports(path)
        )


def test_frozen_predecessor_and_candidate_hashes_remain_exact() -> None:
    expected = {
        Path("src/palintrace/checkers/unsupported_claim.py"): (
            heldout.FROZEN_UNSUPPORTED_CLAIM_SHA256
        ),
        Path("src/palintrace/checkers/unsupported_claim_identity_grounded.py"): (
            heldout.FROZEN_CANDIDATE_SHA256
        ),
        Path("src/palintrace/semantics/identity.py"): (
            heldout.FROZEN_IDENTITY_CONTRACT_SHA256
        ),
        Path("src/palintrace/semantics/local_nli.py"): heldout.FROZEN_LOCAL_NLI_SHA256,
        Path("src/palintrace/semantics/composition.py"): heldout.FROZEN_COMPOSITION_SHA256,
        Path("tests/fixtures/benchmark_v0.1.sha256.json"): (
            heldout.FROZEN_BENCHMARK_MANIFEST_SHA256
        ),
        Path("tests/fixtures/unsupported_identity_probe_v0.1.json"): (
            heldout.FROZEN_IDENTITY_PROBE_FIXTURE_SHA256
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected
