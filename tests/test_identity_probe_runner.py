from __future__ import annotations

import json
from pathlib import Path

import pytest

import memlint.evaluation.identity_probe as identity_probe
import memlint.semantics as semantics
import tools.evaluate_identity_grounding as runner
from memlint.semantics import SemanticJudgment, SemanticRelation, SemanticUsage


class _FakePinnedJudge:
    judge_id = f"hf-nli:{identity_probe.IDENTITY_PROBE_MODEL_ID}"
    judge_version = identity_probe.IDENTITY_PROBE_MODEL_REVISION

    def __init__(self) -> None:
        self.call_count = 0

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        assert premise and hypothesis
        self.call_count += 1
        return SemanticJudgment(
            relation=(
                SemanticRelation.ENTAILMENT
                if self.call_count % 2 == 1
                else SemanticRelation.CONTRADICTION
            ),
            score=0.9,
            usage=SemanticUsage(model_calls=1, input_tokens=8, output_tokens=0),
        )


def test_runner_help_exposes_no_method_or_fixture_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--output" in output
    assert "--fixture" not in output
    assert "--model" not in output
    assert "--revision" not in output
    assert "--condition" not in output
    assert "--threshold" not in output


def test_bad_fixture_fails_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_fixture = tmp_path / "probe.json"
    bad_fixture.write_bytes(runner.FROZEN_FIXTURE_PATH.read_bytes() + b"\n")
    model_constructed = False

    def construct() -> object:
        nonlocal model_constructed
        model_constructed = True
        return object()

    monkeypatch.setattr(runner, "FROZEN_FIXTURE_PATH", bad_fixture)
    monkeypatch.setattr(runner, "_build_judge", construct)
    with pytest.raises(SystemExit) as error:
        runner.main([])
    assert error.value.code == 2
    assert model_constructed is False
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


def test_runner_uses_fake_judge_only_after_preflight_and_writes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = _FakePinnedJudge()
    order: list[str] = []
    original_preflight = identity_probe.preflight_identity_probe

    def preflight(path: Path) -> identity_probe.IdentityProbeSpec:
        order.append("preflight")
        return original_preflight(path)

    def build() -> _FakePinnedJudge:
        order.append("model")
        return judge

    monkeypatch.setattr(identity_probe, "preflight_identity_probe", preflight)
    monkeypatch.setattr(runner, "_build_judge", build)
    output = tmp_path / "identity-probe-result.json"
    assert runner.main(["--output", str(output)]) == 0
    assert order == ["preflight", "model"]
    assert judge.call_count == 96
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "0.1"
    assert payload["probe_id"] == identity_probe.IDENTITY_PROBE_ID
    assert payload["fixture_sha256"] == identity_probe.IDENTITY_PROBE_FIXTURE_SHA256
    assert len(payload["judgments"]) == 96
    assert payload["interpretation"] == "INCONCLUSIVE_FAILURE_NOT_REPRODUCED"


def test_output_must_be_new_and_outside_fixtures(tmp_path: Path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="must be new"):
        runner._write_result(existing, "replacement")
    assert existing.read_text(encoding="utf-8") == "preserve"

    with pytest.raises(ValueError, match="test fixtures"):
        runner._write_result(
            runner.REPOSITORY_ROOT / "tests/fixtures/probe-output.json",
            "{}\n",
        )
