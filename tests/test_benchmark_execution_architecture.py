from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import palintrace.cli as cli

FROZEN_BENCHMARK_MODULE = Path("src/palintrace/evaluation/benchmark.py")
FROZEN_HASH_MANIFEST = Path("tests/fixtures/benchmark_v0.1.sha256.json")
FROZEN_FIXTURE_ROOT = Path("tests/fixtures/benchmark_v0.1")
PART6C_TESTS = (
    Path("tests/test_benchmark_execution.py"),
    Path("tests/test_benchmark_execution_models.py"),
    Path("tests/test_benchmark_runner.py"),
    Path("tests/test_evaluation_clean_control.py"),
    Path("tests/test_experimental_lexical.py"),
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


def test_part6b_benchmark_module_and_hash_manifest_are_byte_frozen() -> None:
    assert hashlib.sha256(FROZEN_BENCHMARK_MODULE.read_bytes()).hexdigest() == (
        "585cfcc0683edf9d9fc7bb692f350ae1b97920c7c224bbadf217d7f2107130c7"
    )
    assert hashlib.sha256(FROZEN_HASH_MANIFEST.read_bytes()).hexdigest() == (
        "028fe4e096adc556b4d23bd89c6f5c79f635cbcfe327ad94a2a5a2e7794a659d"
    )
    expected = json.loads(FROZEN_HASH_MANIFEST.read_text(encoding="utf-8"))
    actual = {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(FROZEN_FIXTURE_ROOT.iterdir())
        if path.is_file()
    }
    assert actual == expected


def test_experimental_lexical_exists_only_in_evaluation() -> None:
    assert Path("src/palintrace/evaluation/experimental_lexical.py").is_file()
    assert not Path("src/palintrace/retrieval/lexical.py").exists()
    assert {path.name for path in Path("src/palintrace/retrieval").glob("*.py")} == {
        "__init__.py",
        "base.py",
        "challenge.py",
        "models.py",
        "policy.py",
    }
    assert not any(
        module == "palintrace.evaluation" or module.startswith("palintrace.evaluation.")
        for path in Path("src/palintrace/retrieval").glob("*.py")
        for module in _imports(path)
    )


def test_part6c_tests_do_not_name_frozen_heldout_store_or_transcript_inputs() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PART6C_TESTS)
    forbidden = {
        "fixture_h1_store.json",
        "fixture_h2_store.json",
        "fixture_h3_store.json",
        "fixture_h1_transcripts.json",
        "fixture_h2_transcripts.json",
        "fixture_h3_transcripts.json",
    }
    assert not any(name in combined for name in forbidden)


def test_execution_models_have_no_inferential_or_generic_classification_metrics() -> None:
    source = Path("src/palintrace/evaluation/execution_models.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "false_positive_rate",
        "specificity",
        "confidence_interval",
        "p_value",
        "p-value",
    )
    assert not any(name in source for name in forbidden)


def test_normal_cli_remains_unchanged_and_has_no_benchmark_command() -> None:
    parser = cli.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert set(command_action.choices) == {
        "dump",
        "mutate",
        "audit",
        "retrieval-audit",
    }
