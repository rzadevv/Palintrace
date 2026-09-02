from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PROBE_MODULE = Path("src/palintrace/evaluation/retrieval_negation_confirmatory.py")
RUNNER = Path("tools/evaluate_retrieval_negation_confirmatory.py")


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_frozen_retrieval_and_part_6h_predecessors_are_byte_exact() -> None:
    expected = {
        Path("src/palintrace/retrieval/__init__.py"): (
            "d322710fade529108a8b9dc58b7fa9945a821a74d0df27548663f9574a681ff1"
        ),
        Path("src/palintrace/retrieval/base.py"): (
            "1af0cd7b4648ed2f9212fc042f19018ade298e80bd301c40af881b49ae1a7df2"
        ),
        Path("src/palintrace/retrieval/challenge.py"): (
            "d13a77bf5bf4120f03f2a4aaaac2d6fd48a8ff3fab09251f23e723a8e23e226e"
        ),
        Path("src/palintrace/retrieval/models.py"): (
            "781100bedafa2f6306968dae06019325a35f350e8ca525b11fb0cd74383a8057"
        ),
        Path("src/palintrace/retrieval/policy.py"): (
            "e124ce00b1fe9ce4b131813ed1e930b0f88c00ac0be69f949004ed501c512a74"
        ),
        Path("src/palintrace/evaluation/experimental_lexical.py"): (
            "3022262393253366349dc811fdbd822edf9e7b9517c9da7d26828a3c00ca94d4"
        ),
        Path("src/palintrace/evaluation/retrieval.py"): (
            "7876f02f9ff7ce1eb63da0a0dfe443c6cc98c415123c073d11202552e40b79f2"
        ),
        Path("src/palintrace/evaluation/retrieval_strong_probe.py"): (
            "9cda7aec925f03697f386520cfcdc1d73cd7108c8a6c794c3cdb82aaf0cf5a21"
        ),
        Path("tools/evaluate_retrieval_strong_probe.py"): (
            "05e5cfc908ceb05d447d348b35f27ce7a436ed22ecbbb7affbb4b45e775ff501"
        ),
        Path("tests/fixtures/retrieval_shadowing_strong_probe_v0.1.json"): (
            "98c2a6e1f1f5a38f34691918dd99a1eeb9096888ca2c02d63469595820c87748"
        ),
        Path("src/palintrace/checkers/retrieval_shadowing.py"): (
            "6f36d4606f489c3cf98ee29ac548bd4609b74f6c6977e3346b9bbb1d38aa783d"
        ),
        Path("src/palintrace/mutations/shadowing.py"): (
            "2350c9bf934124b9409e4b392363e41f52b972b0d620561fa9870928ba1cedce"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_frozen_identity_semantic_benchmark_and_cli_files_are_byte_exact() -> None:
    expected = {
        Path("src/palintrace/checkers/unsupported_claim.py"): (
            "604d08b766cf901475be78258c31162d76d748759db2f205049bbac285fa6cdc"
        ),
        Path("src/palintrace/checkers/unsupported_claim_identity_grounded.py"): (
            "0bae646f781a63a90371c45411696c780a2ced6f6d1a0d2a3c6343b46aa06d98"
        ),
        Path("src/palintrace/semantics/identity.py"): (
            "23cd31a3d75fea5bd260dc5f8f9fd87c39dd930d851d1a44477c77a9423bd4f2"
        ),
        Path("src/palintrace/semantics/identity_source.py"): (
            "7fde1d795c74b2ef05b2fc9842047e50d075b886f9298c862f635080b8e34179"
        ),
        Path("src/palintrace/semantics/local_nli.py"): (
            "41b853a6540b6bc18b1abd644c3477c3c05e28978c8e6ec3ef861ea9d37f8f19"
        ),
        Path("src/palintrace/semantics/composition.py"): (
            "5c710e879126f80153ee80be3b20392c7a3bcea3f1c8a7089fba761876cb87e3"
        ),
        Path("tests/fixtures/benchmark_v0.1.sha256.json"): (
            "028fe4e096adc556b4d23bd89c6f5c79f635cbcfe327ad94a2a5a2e7794a659d"
        ),
        Path("src/palintrace/cli.py"): (
            "620aba148c80f6b6468156876eb3c9a4e6cc6d4b12b4d5325f8615823a2c9c57"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_production_modules_do_not_depend_on_confirmatory_evaluation() -> None:
    forbidden_roots = (
        Path("src/palintrace/adapters"),
        Path("src/palintrace/retrieval"),
        Path("src/palintrace/checkers"),
        Path("src/palintrace/semantics"),
    )
    for root in forbidden_roots:
        for path in root.glob("*.py"):
            assert "palintrace.evaluation.retrieval_negation_confirmatory" not in _imports(
                path
            )
    assert "palintrace.evaluation.retrieval_negation_confirmatory" not in _imports(
        Path("src/palintrace/cli.py")
    )
    assert "retrieval_negation_confirmatory" not in Path(
        "src/palintrace/__init__.py"
    ).read_text(encoding="utf-8")


def test_evaluation_models_do_not_import_concrete_retriever_or_checker() -> None:
    imports = _imports(PROBE_MODULE)
    assert "palintrace.evaluation.experimental_lexical" not in imports
    assert not any(
        module == "palintrace.checkers" or module.startswith("palintrace.checkers.")
        for module in imports
    )


def test_runner_is_only_new_concrete_retriever_construction_site() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ExperimentalLexicalRetriever"
    ]
    assert len(calls) == 1
    parents = [
        function
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and calls[0] in tuple(ast.walk(function))
    ]
    assert [function.name for function in parents] == ["_build_retriever"]


def test_confirmatory_probe_is_not_in_benchmark_or_public_cli() -> None:
    benchmark = Path("tests/fixtures/benchmark_v0.1/benchmark.json").read_text(
        encoding="utf-8"
    )
    cli = Path("src/palintrace/cli.py").read_text(encoding="utf-8")
    assert "retrieval-negation-confirmatory-v0.1" not in benchmark
    assert "negation-confirmatory" not in cli
