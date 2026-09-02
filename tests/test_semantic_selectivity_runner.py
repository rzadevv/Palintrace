from __future__ import annotations

import json
from pathlib import Path

import pytest

import palintrace.evaluation.semantic_selectivity as probe
import palintrace.semantics as semantics
import tools.evaluate_semantic_selectivity as runner
from palintrace.semantics import SemanticJudgment, SemanticRelation, SemanticUsage


class _FakePinnedJudge:
    judge_id = f"hf-nli:{probe.SEMANTIC_SELECTIVITY_MODEL_ID}"
    judge_version = probe.SEMANTIC_SELECTIVITY_MODEL_REVISION

    def __init__(self, *, unsupported_supported: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.unsupported_supported = unsupported_supported

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        clean = len(self.calls) % 2 == 1
        relation = (
            SemanticRelation.ENTAILMENT
            if clean or self.unsupported_supported
            else SemanticRelation.CONTRADICTION
        )
        return SemanticJudgment(
            relation=relation,
            score=0.96,
            usage=SemanticUsage(model_calls=1, input_tokens=14, output_tokens=0),
        )


def test_runner_exposes_only_required_output_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--output" in output
    for forbidden in (
        "--fixture",
        "--threshold",
        "--model",
        "--revision",
        "--device",
        "--split",
        "--policy",
    ):
        assert forbidden not in output


def test_bad_fixture_fails_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_fixture = tmp_path / "probe.json"
    bad_fixture.write_bytes(runner.FROZEN_FIXTURE_PATH.read_bytes() + b"\n")
    constructed = False

    def construct() -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(runner, "FROZEN_FIXTURE_PATH", bad_fixture)
    monkeypatch.setattr(runner, "_build_judge", construct)
    with pytest.raises(SystemExit) as error:
        runner.main(["--output", str(tmp_path / "result.json")])
    assert error.value.code == 2
    assert constructed is False
    assert "fixture SHA mismatch" in capsys.readouterr().err


def test_frozen_model_constructor_uses_exact_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    def construct(*, model_id: str, revision: str, device: str) -> object:
        assert model_id == "cross-encoder/nli-MiniLM2-L6-H768"
        assert revision == "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
        assert device == "cpu"
        return sentinel

    monkeypatch.setattr(semantics, "LocalNLISemanticJudge", construct)
    assert runner._build_judge() is sentinel


def test_output_must_be_new_outside_repository_with_existing_parent(tmp_path: Path) -> None:
    output = tmp_path / "semantic-selectivity-result.json"
    assert runner._validate_output_path(output) == output.resolve()
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        runner._validate_output_path(output)
    assert output.read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ValueError, match="outside the repository"):
        runner._validate_output_path(runner.REPOSITORY_ROOT / "result.json")
    with pytest.raises(ValueError, match="parent must already exist"):
        runner._validate_output_path(tmp_path / "absent" / "result.json")


def test_runner_uses_one_fake_judge_after_preflight_and_writes_96_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = _FakePinnedJudge()
    order: list[str] = []
    original_preflight = probe.preflight_semantic_selectivity

    def preflight(path: Path) -> probe.SemanticSelectivitySpec:
        order.append("preflight")
        return original_preflight(path)

    def build() -> _FakePinnedJudge:
        order.append("model")
        return judge

    monkeypatch.setattr(probe, "preflight_semantic_selectivity", preflight)
    monkeypatch.setattr(runner, "_build_judge", build)
    output = tmp_path / "result.json"
    assert runner.main(["--output", str(output)]) == 0
    assert order == ["preflight", "model"]
    assert len(judge.calls) == 96
    result = probe.SemanticSelectivityExecutionResult.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    assert len(result.judgments) == 96
    assert result.calibration_selection.selected_threshold == 0.5
    assert (
        result.summary.interpretation
        is probe.SemanticSelectivityInterpretation.INCONCLUSIVE_BASELINE_TOO_EASY
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "premise" not in payload
    assert "clean_hypothesis" not in payload
    assert "unsupported_hypothesis" not in payload


def test_failed_calibration_stops_before_confirmatory_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = _FakePinnedJudge(unsupported_supported=True)
    monkeypatch.setattr(runner, "_build_judge", lambda: judge)
    output = tmp_path / "failed-calibration.json"
    assert runner.main(["--output", str(output)]) == 0
    assert len(judge.calls) == 48
    result = probe.SemanticSelectivityExecutionResult.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    assert result.calibration_selection.status is probe.CalibrationSelectionStatus.FAILED
    assert len(result.judgments) == 48
    assert (
        result.summary.interpretation
        is probe.SemanticSelectivityInterpretation.DOES_NOT_SUPPORT_H3
    )
