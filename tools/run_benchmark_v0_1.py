"""Execute the frozen benchmark v0.1 after fail-closed integrity checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from palintrace.evaluation import (
    DEFAULT_BENCHMARK_PATH,
    UNSUPPORTED_MODEL_ID,
    UNSUPPORTED_MODEL_REVISION,
    EvaluationError,
    build_execution_provenance,
    execute_benchmark_v0_1,
    preflight_benchmark_v0_1,
)
from palintrace.semantics import LocalNLISemanticJudge

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/benchmark_v0.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Palintrace controlled benchmark v0.1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for canonical result and provenance JSON.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Benchmark JSON; canonical content must match frozen v0.1.",
    )
    return parser


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    frozen_root = FROZEN_FIXTURE_ROOT.resolve()
    if resolved == frozen_root or resolved.is_relative_to(frozen_root):
        raise ValueError("benchmark output must not be written into frozen fixtures")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("benchmark output path must be a directory")
        if any(resolved.iterdir()):
            raise ValueError("benchmark output directory must be empty")
    else:
        resolved.mkdir(parents=True)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """Run only after benchmark, fixture, method, and condition pre-flight succeeds."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        benchmark_path = args.benchmark
        if not benchmark_path.is_absolute():
            benchmark_path = REPOSITORY_ROOT / benchmark_path
        spec = preflight_benchmark_v0_1(
            repository_root=REPOSITORY_ROOT,
            benchmark_path=benchmark_path,
        )
        output_directory = _prepare_output_directory(args.output_dir)
    except (EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))

    judge = LocalNLISemanticJudge(
        model_id=UNSUPPORTED_MODEL_ID,
        revision=UNSUPPORTED_MODEL_REVISION,
        device="cpu",
    )
    result = execute_benchmark_v0_1(
        spec=spec,
        repository_root=REPOSITORY_ROOT,
        semantic_judge=judge,
    )
    provenance = build_execution_provenance()
    (output_directory / "benchmark-result.json").write_text(
        result.to_json(), encoding="utf-8"
    )
    (output_directory / "execution-provenance.json").write_text(
        provenance.to_json(), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    raise SystemExit(main())
