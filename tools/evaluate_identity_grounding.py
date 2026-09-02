#!/usr/bin/env python3
"""Execute the frozen speaker-identity development probe after fail-closed preflight."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import memlint.evaluation.identity_probe as identity_probe
from memlint.semantics import SemanticJudge

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = REPOSITORY_ROOT / identity_probe.IDENTITY_PROBE_FIXTURE_PATH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen post-v0.1 speaker-identity DEVELOPMENT probe; "
            "this is not held-out benchmark validation."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON result path; otherwise write deterministic JSON to stdout.",
    )
    return parser


def _build_judge() -> SemanticJudge:
    """Construct pinned MiniLM only after every fixture and method preflight check."""

    from memlint.semantics import LocalNLISemanticJudge

    return LocalNLISemanticJudge(
        model_id=identity_probe.IDENTITY_PROBE_MODEL_ID,
        revision=identity_probe.IDENTITY_PROBE_MODEL_REVISION,
        device=identity_probe.IDENTITY_PROBE_DEVICE,
    )


def _write_result(path: Path, text: str) -> None:
    resolved = path.resolve()
    fixture_root = (REPOSITORY_ROOT / "tests/fixtures").resolve()
    if resolved == fixture_root or resolved.is_relative_to(fixture_root):
        raise ValueError("identity-probe output must not be written into test fixtures")
    if resolved.exists():
        raise ValueError("identity-probe output path must be new")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the complete freeze before model construction or semantic inference."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec = identity_probe.preflight_identity_probe(FROZEN_FIXTURE_PATH)
        identity_probe.validate_identity_probe_model_identity()
    except (identity_probe.IdentityProbeInputError, OSError, ValueError) as error:
        parser.error(str(error))

    judge = _build_judge()
    result = identity_probe.execute_identity_probe(
        spec=spec,
        semantic_judge=judge,
    )
    text = result.to_json()
    if args.output is None:
        sys.stdout.write(text)
    else:
        try:
            _write_result(args.output, text)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
