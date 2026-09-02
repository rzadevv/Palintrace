from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PROBE_MODULE = Path("src/memlint/evaluation/retrieval_strong_probe.py")
RUNNER = Path("tools/evaluate_retrieval_strong_probe.py")


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_frozen_retrieval_implementation_and_part_five_contracts_are_byte_exact() -> None:
    expected = {
        Path("src/memlint/retrieval/__init__.py"): (
            "68d93a1d4349557b1a6f715935f07ebdca3501e05b04bc53146249d1b1c22af7"
        ),
        Path("src/memlint/retrieval/base.py"): (
            "85157b75a506b70973544a2e1aef41c36cedaca51cd5607899dfebd053575b6c"
        ),
        Path("src/memlint/retrieval/challenge.py"): (
            "b2e471c3ed14c6dd6619969a1b6df2db2ca66955fb5dd44763fc9d3448c72f7d"
        ),
        Path("src/memlint/retrieval/models.py"): (
            "781100bedafa2f6306968dae06019325a35f350e8ca525b11fb0cd74383a8057"
        ),
        Path("src/memlint/retrieval/policy.py"): (
            "3d954645236f1ed49671a1bcb48a947d1fde8723758738ac5fc55a9e77a14a85"
        ),
        Path("src/memlint/evaluation/experimental_lexical.py"): (
            "570f2a0c2be3561be031c20dfe397696cb0e07e64efcc61b34eacd30f88fddaf"
        ),
        Path("src/memlint/evaluation/retrieval.py"): (
            "e14c106599501ac2acacdc4454340e3ef77da774ee993fedb37466361608b897"
        ),
        Path("src/memlint/checkers/retrieval_shadowing.py"): (
            "d3e549812d071b9b516f21bfdaf4be2ca5f685d53090aa01d32913dd4cf64dca"
        ),
        Path("src/memlint/mutations/shadowing.py"): (
            "73197cae26979e939009252272b156d2f1e48d0d784fb07580cb47c2f6f3bbd1"
        ),
        Path("src/memlint/cli.py"): (
            "4a1f558e8e421f3cde965d5cdbe4c464082685e30e760249d0b9416503ddd942"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_identity_semantic_and_benchmark_predecessors_are_byte_exact() -> None:
    expected = {
        Path("src/memlint/checkers/unsupported_claim.py"): (
            "04fd713308d9ed55e79501a31e99904939a2caf8ef90f2187e3fe1f594d09a8a"
        ),
        Path("src/memlint/checkers/unsupported_claim_identity_grounded.py"): (
            "6b742eeff6d4280661adba61ed201b67a6bc25a7d9a2b0c967ebbccd0c3210c5"
        ),
        Path("src/memlint/semantics/identity.py"): (
            "c6b54d0229cb6b87b5e23997685e9855b8b789ea2d68f6e6f07ee45a749f82f9"
        ),
        Path("src/memlint/semantics/identity_source.py"): (
            "058590ef7258b9f611de486b5130b866893f0e4a2c1091d5ecd0d0a465463e87"
        ),
        Path("src/memlint/semantics/local_nli.py"): (
            "aafe1e1a9d662879640285784704cdbfecefec4c25e402fae07101dd7ea087b1"
        ),
        Path("src/memlint/semantics/composition.py"): (
            "cd617221c65bb6a58de7164f7438143d661903f62f57076f4869d3e28d6a7629"
        ),
        Path("tests/fixtures/benchmark_v0.1.sha256.json"): (
            "de4bb8c2076a2c89b7e2df95518ef5588934644b711119fccc8727e0e9ac73fb"
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
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_production_runtime_checker_and_cli_do_not_import_probe() -> None:
    forbidden_roots = (
        Path("src/memlint/retrieval"),
        Path("src/memlint/checkers"),
        Path("src/memlint/semantics"),
    )
    for root in forbidden_roots:
        for path in root.glob("*.py"):
            assert "memlint.evaluation.retrieval_strong_probe" not in _imports(path)
    assert "memlint.evaluation.retrieval_strong_probe" not in _imports(
        Path("src/memlint/cli.py")
    )
    assert "retrieval_strong_probe" not in Path("src/memlint/__init__.py").read_text(
        encoding="utf-8"
    )


def test_probe_models_do_not_import_concrete_retriever_or_production_checker() -> None:
    imports = _imports(PROBE_MODULE)
    assert "memlint.evaluation.experimental_lexical" not in imports
    assert not any(
        module == "memlint.checkers" or module.startswith("memlint.checkers.")
        for module in imports
    )


def test_runner_is_the_only_new_concrete_retriever_construction_site() -> None:
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


def test_runner_preflights_before_future_probe_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    preflight = next(
        node for node in calls if node.func.attr == "preflight_retrieval_strong_probe"
    )
    execute = next(node for node in calls if node.func.attr == "to_json")
    assert preflight.lineno < execute.lineno


def test_probe_is_not_part_of_benchmark_v01_or_public_cli() -> None:
    benchmark = Path("tests/fixtures/benchmark_v0.1/benchmark.json").read_text(
        encoding="utf-8"
    )
    cli = Path("src/memlint/cli.py").read_text(encoding="utf-8")
    assert "retrieval-shadowing-strong-probe-v0.1" not in benchmark
    assert "retrieval-strong" not in cli
