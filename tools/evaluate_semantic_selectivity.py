#!/usr/bin/env python3
"""Execute the frozen H3 semantic selectivity probe in one invocation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import memlint.evaluation.semantic_selectivity as probe
from memlint.semantics import SemanticJudge

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = REPOSITORY_ROOT / probe.SEMANTIC_SELECTIVITY_FIXTURE_PATH


def _validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    repository = REPOSITORY_ROOT.resolve()
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("semantic selectivity output must be outside the repository")
    if resolved.exists():
        raise ValueError("semantic selectivity output must not already exist")
    if not resolved.parent.exists():
        raise ValueError("semantic selectivity output parent must already exist")
    return resolved


def _build_judge() -> SemanticJudge:
    """Construct pinned MiniLM only after fixture and output preflight."""

    from memlint.semantics import LocalNLISemanticJudge

    return LocalNLISemanticJudge(
        model_id=probe.SEMANTIC_SELECTIVITY_MODEL_ID,
        revision=probe.SEMANTIC_SELECTIVITY_MODEL_REVISION,
        device=probe.SEMANTIC_SELECTIVITY_DEVICE,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight, run calibration and confirmation without a human selection gap, and write."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen H3 calibration and confirmatory synthetic probe once. "
            "The result path must be new and outside the repository."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = _validate_output_path(args.output)
        spec = probe.preflight_semantic_selectivity(FROZEN_FIXTURE_PATH)
    except (OSError, probe.SemanticSelectivityInputError, ValueError) as error:
        parser.error(str(error))

    judge = _build_judge()
    result = probe.execute_semantic_selectivity(
        spec=spec,
        semantic_judge=judge,
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(result.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - future one-shot execution entry point
    raise SystemExit(main())
