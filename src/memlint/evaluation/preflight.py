"""Fail-closed integrity checks performed before benchmark detector execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memlint.evaluation.benchmark import (
    BENCHMARK_ID,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_SPEC_SHA256,
    BenchmarkSpec,
    load_benchmark_spec,
)
from memlint.evaluation.models import EvaluationInputError

DEFAULT_BENCHMARK_PATH = Path("tests/fixtures/benchmark_v0.1/benchmark.json")
DEFAULT_FIXTURE_HASH_MANIFEST_PATH = Path("tests/fixtures/benchmark_v0.1.sha256.json")


def _sha256_bytes(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvaluationInputError(f"could not read frozen benchmark fixture: {path}") from error


def _load_hash_manifest(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationInputError("could not load the frozen fixture hash manifest") from error
    if not isinstance(payload, dict) or not payload:
        raise EvaluationInputError("fixture hash manifest must be a nonempty JSON object")
    hashes: dict[str, str] = {}
    for relative_path, digest in payload.items():
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise EvaluationInputError("fixture hash paths must be nonblank strings")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvaluationInputError("fixture hash values must be lowercase SHA-256 digests")
        hashes[relative_path] = digest
    return hashes


def preflight_benchmark_v0_1(
    *,
    repository_root: Path,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    hash_manifest_path: Path = DEFAULT_FIXTURE_HASH_MANIFEST_PATH,
) -> BenchmarkSpec:
    """Validate the exact benchmark and every fixture byte before any model is loaded."""

    if not isinstance(repository_root, Path):
        raise EvaluationInputError("repository_root must be a pathlib.Path")
    if not isinstance(benchmark_path, Path):
        raise EvaluationInputError("benchmark_path must be a pathlib.Path")
    if not isinstance(hash_manifest_path, Path):
        raise EvaluationInputError("hash_manifest_path must be a pathlib.Path")

    resolved_benchmark = (
        benchmark_path if benchmark_path.is_absolute() else repository_root / benchmark_path
    )
    spec = load_benchmark_spec(resolved_benchmark)
    canonical_sha = hashlib.sha256(spec.to_json(indent=None).encode("utf-8")).hexdigest()
    if canonical_sha != BENCHMARK_SPEC_SHA256:
        raise EvaluationInputError("benchmark canonical SHA does not match frozen v0.1")
    if spec.schema_version != BENCHMARK_SCHEMA_VERSION or spec.benchmark_id != BENCHMARK_ID:
        raise EvaluationInputError("benchmark schema or identity does not match frozen v0.1")

    resolved_manifest = (
        hash_manifest_path
        if hash_manifest_path.is_absolute()
        else repository_root / hash_manifest_path
    )
    expected_hashes = _load_hash_manifest(resolved_manifest)
    frozen_root = repository_root / "tests/fixtures/benchmark_v0.1"
    actual_fixture_paths = {
        path.relative_to(repository_root).as_posix()
        for path in frozen_root.iterdir()
        if path.is_file()
    }
    if set(expected_hashes) != actual_fixture_paths:
        raise EvaluationInputError("fixture hash manifest does not cover the frozen fixture set")
    mismatches = tuple(
        relative_path
        for relative_path, expected_sha in sorted(expected_hashes.items())
        if _sha256_bytes(repository_root / relative_path) != expected_sha
    )
    if mismatches:
        raise EvaluationInputError(
            "frozen benchmark fixture SHA mismatch: " + ", ".join(mismatches)
        )
    return spec
