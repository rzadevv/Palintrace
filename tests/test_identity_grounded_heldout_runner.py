from __future__ import annotations

import json
from pathlib import Path

import pytest

import palintrace.evaluation.identity_grounded_heldout as heldout
import palintrace.semantics as semantics
import tools.run_identity_grounded_heldout_v0_1 as runner
from palintrace.semantics import SemanticJudgment, SemanticRelation, SemanticUsage


class _FakePinnedJudge:
    judge_id = f"hf-nli:{heldout.HELDOUT_MODEL_ID}"
    judge_version = heldout.HELDOUT_MODEL_REVISION

    def __init__(self) -> None:
        self.call_count = 0

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        assert premise and hypothesis
        self.call_count += 1
        return SemanticJudgment(
            relation=SemanticRelation.ENTAILMENT,
            score=0.9,
            usage=SemanticUsage(model_calls=1, input_tokens=5, output_tokens=0),
        )


def test_runner_help_has_no_fixture_model_or_threshold_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--output-dir" in output
    assert "--fixture" not in output
    assert "--model" not in output
    assert "--revision" not in output
    assert "--threshold" not in output
    assert "--condition" not in output


def test_bad_fixture_stops_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_fixture = tmp_path / "heldout.json"
    bad_fixture.write_bytes(runner.FROZEN_FIXTURE_PATH.read_bytes() + b"\n")
    constructed = False

    def build() -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(runner, "FROZEN_FIXTURE_PATH", bad_fixture)
    monkeypatch.setattr(heldout, "validate_phase_a_manifest", lambda _root: None)
    monkeypatch.setattr(runner, "_build_judge", build)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output-dir", str(tmp_path / "result")])
    assert error.value.code == 2
    assert constructed is False
    assert "fixture SHA mismatch" in capsys.readouterr().err


def test_pinned_model_constructor_uses_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def construct(*, model_id: str, revision: str, device: str) -> object:
        assert model_id == "cross-encoder/nli-MiniLM2-L6-H768"
        assert revision == "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
        assert device == "cpu"
        return sentinel

    monkeypatch.setattr(semantics, "LocalNLISemanticJudge", construct)
    assert runner._build_judge() is sentinel


def test_runner_preflights_before_fake_model_and_writes_two_safe_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = _FakePinnedJudge()
    order: list[str] = []
    original_preflight = heldout.preflight_heldout_fixture

    def preflight(path: Path) -> heldout.HeldoutSpec:
        order.append("preflight")
        return original_preflight(path)

    def frozen_repository(
        repository_root: Path,
        *,
        external_identity_result: Path | None = None,
    ) -> tuple[bool, bool, bool]:
        del repository_root, external_identity_result
        order.append("repository")
        return True, True, True

    def build() -> _FakePinnedJudge:
        order.append("model")
        return judge

    monkeypatch.setattr(heldout, "preflight_heldout_fixture", preflight)
    monkeypatch.setattr(heldout, "validate_phase_a_manifest", lambda _root: None)
    monkeypatch.setattr(heldout, "validate_frozen_repository", frozen_repository)
    monkeypatch.setattr(runner, "_build_judge", build)
    output_dir = tmp_path / "heldout-result"
    assert runner.main(["--output-dir", str(output_dir)]) == 0
    assert order == ["preflight", "repository", "model"]
    assert judge.call_count == 162
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "execution-provenance.json",
        "heldout-result.json",
    ]
    payload = json.loads((output_dir / "heldout-result.json").read_text())
    assert payload["evaluation_id"] == heldout.HELDOUT_EVALUATION_ID
    assert len(payload["semantic_trials"]) == 160
    assert len(payload["coverage_trials"]) == 6
    assert payload["coverage"]["abstained_memories"] == 4
    provenance = json.loads((output_dir / "execution-provenance.json").read_text())
    assert provenance["result_sha256"] == heldout.sha256_file(
        output_dir / "heldout-result.json"
    )


def test_output_directory_must_be_new_and_outside_repository(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must be new"):
        runner._validate_output_dir(existing)
    with pytest.raises(ValueError, match="outside the repository"):
        runner._validate_output_dir(runner.REPOSITORY_ROOT / "heldout-result")
