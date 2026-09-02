#!/usr/bin/env python3
"""Execute the preregistered identity-grounded held-out evaluation exactly once."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from collections.abc import Sequence
from pathlib import Path

import palintrace.evaluation.identity_grounded_heldout as heldout
from palintrace.semantics import SemanticJudge

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FIXTURE_PATH = REPOSITORY_ROOT / heldout.HELDOUT_FIXTURE_PATH
DEVELOPMENT_FIXTURE_PATH = (
    REPOSITORY_ROOT / "tests/fixtures/unsupported_identity_probe_v0.1.json"
)
EXTERNAL_6F_RESULT_PATH = Path(
    "/home/chisste/Desktop/Projects/palintrace-identity-probe-v0.1-first-run/"
    "identity-probe-result.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Part 6G-C held-out evaluation. The output directory must "
            "be new and outside the repository."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _build_judge() -> SemanticJudge:
    """Construct pinned MiniLM only after complete fail-closed preflight."""

    from palintrace.semantics import LocalNLISemanticJudge

    return LocalNLISemanticJudge(
        model_id=heldout.HELDOUT_MODEL_ID,
        revision=heldout.HELDOUT_MODEL_REVISION,
        device=heldout.HELDOUT_DEVICE,
    )


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    repository = REPOSITORY_ROOT.resolve()
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("held-out result artifacts must remain outside the repository")
    if resolved.exists():
        raise ValueError("held-out output directory must be new")
    if not resolved.parent.exists():
        raise ValueError("held-out output parent directory must already exist")
    return resolved


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight frozen inputs, construct the model once, and write two safe artifacts."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output_dir = _validate_output_dir(args.output_dir)
        heldout.validate_phase_a_manifest(REPOSITORY_ROOT)
        spec = heldout.preflight_heldout_fixture(FROZEN_FIXTURE_PATH)
        heldout.validate_freshness(spec, DEVELOPMENT_FIXTURE_PATH)
        heldout.validate_coverage_contract(spec)
        protected, candidate_nonpublic, candidate_noncli = (
            heldout.validate_frozen_repository(
                REPOSITORY_ROOT,
                external_identity_result=EXTERNAL_6F_RESULT_PATH,
            )
        )
    except (heldout.HeldoutInputError, OSError, ValueError) as error:
        parser.error(str(error))

    judge = _build_judge()
    if (judge.judge_id, judge.judge_version) != (
        f"hf-nli:{heldout.HELDOUT_MODEL_ID}",
        heldout.HELDOUT_MODEL_REVISION,
    ):
        parser.error("constructed semantic judge does not match the preregistration")
    result = heldout.execute_heldout(
        spec=spec,
        semantic_judge=judge,
        protected_hashes_valid=protected,
        candidate_nonpublic=candidate_nonpublic,
        candidate_noncli=candidate_noncli,
    )
    result_text = result.to_json()
    result_sha = heldout.sha256_text(result_text)
    provenance = heldout.HeldoutExecutionProvenance(
        schema_version=heldout.HELDOUT_SCHEMA_VERSION,
        evaluation_id=heldout.HELDOUT_EVALUATION_ID,
        fixture_sha256=heldout.HELDOUT_FIXTURE_SHA256,
        evaluation_module_sha256=heldout.sha256_file(Path(heldout.__file__)),
        runner_sha256=heldout.sha256_file(Path(__file__)),
        candidate_sha256=heldout.FROZEN_CANDIDATE_SHA256,
        baseline_sha256=heldout.FROZEN_UNSUPPORTED_CLAIM_SHA256,
        identity_contract_sha256=heldout.FROZEN_IDENTITY_CONTRACT_SHA256,
        local_nli_sha256=heldout.FROZEN_LOCAL_NLI_SHA256,
        composition_sha256=heldout.FROZEN_COMPOSITION_SHA256,
        benchmark_sha256=heldout.FROZEN_BENCHMARK_SHA256,
        benchmark_manifest_sha256=heldout.FROZEN_BENCHMARK_MANIFEST_SHA256,
        identity_probe_fixture_sha256=heldout.FROZEN_IDENTITY_PROBE_FIXTURE_SHA256,
        identity_probe_result_sha256=heldout.FROZEN_IDENTITY_PROBE_RESULT_SHA256,
        result_sha256=result_sha,
        model_id=heldout.HELDOUT_MODEL_ID,
        model_revision=heldout.HELDOUT_MODEL_REVISION,
        device=heldout.HELDOUT_DEVICE,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
    )

    output_dir.mkdir()
    (output_dir / "heldout-result.json").write_text(result_text, encoding="utf-8")
    (output_dir / "execution-provenance.json").write_text(
        provenance.to_json(), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - real execution entry point
    raise SystemExit(main())
