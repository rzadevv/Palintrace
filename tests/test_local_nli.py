from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

import memlint.semantics.local_nli as local_nli
from memlint.semantics import (
    LocalNLISemanticJudge,
    SemanticDependencyError,
    SemanticInputError,
    SemanticInputTooLongError,
    SemanticModelConfigError,
    SemanticRelation,
    semantic_judge_identity,
)


class FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, int],
        values: list[list[float]] | None = None,
    ) -> None:
        self.shape = shape
        self.values = values
        self.devices: list[str] = []

    def to(self, device: str) -> FakeTensor:
        self.devices.append(device)
        return self

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[list[float]] | None:
        return self.values


class FakeTorch:
    def __init__(self) -> None:
        self.inference_mode_active = False
        self.inference_mode_entries = 0

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        self.inference_mode_entries += 1
        self.inference_mode_active = True
        try:
            yield
        finally:
            self.inference_mode_active = False


class FakeTokenizer:
    def __init__(self, *, token_count: int = 7, model_max_length: int = 512) -> None:
        self.token_count = token_count
        self.model_max_length = model_max_length
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def __call__(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: object,
    ) -> dict[str, FakeTensor]:
        self.calls.append((premise, hypothesis, kwargs))
        return {
            "input_ids": FakeTensor(shape=(1, self.token_count)),
            "attention_mask": FakeTensor(shape=(1, self.token_count)),
        }


class FakeModel:
    def __init__(
        self,
        torch: FakeTorch,
        *,
        logits: tuple[float, float, float] = (0.0, 3.0, 1.0),
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        maximum_tokens: int = 512,
    ) -> None:
        self.torch = torch
        self.logits = logits
        self.config = SimpleNamespace(
            id2label=id2label
            or {0: "contradiction", 1: "entailment", 2: "neutral"},
            label2id=label2id
            or {"contradiction": 0, "entailment": 1, "neutral": 2},
            max_position_embeddings=maximum_tokens,
        )
        self.eval_calls = 0
        self.model_calls = 0
        self.devices: list[str] = []
        self.inference_modes: list[bool] = []

    def to(self, device: str) -> FakeModel:
        self.devices.append(device)
        return self

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def __call__(self, **encoded: FakeTensor) -> SimpleNamespace:
        assert encoded
        self.model_calls += 1
        self.inference_modes.append(self.torch.inference_mode_active)
        return SimpleNamespace(
            logits=FakeTensor(shape=(1, 3), values=[list(self.logits)])
        )


def _judge(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token_count: int = 7,
    tokenizer_limit: int = 512,
    model_limit: int = 512,
    logits: tuple[float, float, float] = (0.0, 3.0, 1.0),
) -> tuple[LocalNLISemanticJudge, FakeTokenizer, FakeModel, FakeTorch]:
    torch = FakeTorch()
    tokenizer = FakeTokenizer(token_count=token_count, model_max_length=tokenizer_limit)
    model = FakeModel(torch, logits=logits, maximum_tokens=model_limit)
    backend = local_nli._LocalNLIBackend(  # noqa: SLF001
        torch=torch,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        model=model,
    )
    monkeypatch.setattr(local_nli, "_load_backend", lambda **kwargs: backend)
    judge = LocalNLISemanticJudge(model_id="example/nli", revision="a" * 40)
    return judge, tokenizer, model, torch


def test_importing_public_packages_does_not_import_optional_libraries() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import memlint; import memlint.semantics; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_optional_dependency_has_clear_install_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(local_nli, "import_module", missing_import)
    with pytest.raises(SemanticDependencyError, match=r"memlint\[semantic-local\]"):
        LocalNLISemanticJudge(model_id="example/nli", revision="a" * 40)


@pytest.mark.parametrize(
    ("model_id", "revision", "message"),
    [
        ("", "a" * 40, "model_id"),
        ("   ", "a" * 40, "model_id"),
        ("example/nli", "", "revision"),
        ("example/nli", "   ", "revision"),
    ],
)
def test_model_identity_fields_must_be_nonblank(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    revision: str,
    message: str,
) -> None:
    def unexpected_load(**kwargs: object) -> None:
        pytest.fail(f"backend should not load: {kwargs}")

    monkeypatch.setattr(local_nli, "_load_backend", unexpected_load)
    with pytest.raises(SemanticModelConfigError, match=message):
        LocalNLISemanticJudge(model_id=model_id, revision=revision)


def test_judge_identity_is_stable_and_uses_exact_declared_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge, _, _, _ = _judge(monkeypatch)
    assert semantic_judge_identity(judge) == (
        "hf-nli:example/nli",
        "a" * 40,
    )


def test_semantic_label_mapping_uses_names_not_numerical_order() -> None:
    config = SimpleNamespace(
        id2label={0: "NEUTRAL", 1: "Contradiction", 2: "entailment"},
        label2id={"NEUTRAL": 0, "Contradiction": 1, "entailment": 2},
    )
    assert local_nli._validated_label_mapping(config) == {  # noqa: SLF001
        0: SemanticRelation.NEUTRAL,
        1: SemanticRelation.CONTRADICTION,
        2: SemanticRelation.ENTAILMENT,
    }


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment"},
            label2id={"contradiction": 0, "entailment": 1},
        ),
        SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "entailment"},
            label2id={},
        ),
        SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "unknown"},
            label2id={},
        ),
        SimpleNamespace(
            id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
            label2id={},
        ),
        SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
            label2id={"contradiction": 1, "entailment": 0, "neutral": 2},
        ),
    ],
    ids=("missing", "duplicate", "unknown", "generic", "disagreeing"),
)
def test_malformed_semantic_label_mapping_is_rejected(config: object) -> None:
    with pytest.raises(SemanticModelConfigError):
        local_nli._validated_label_mapping(config)  # noqa: SLF001


def test_premise_hypothesis_direction_and_nontruncating_tokenization_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge, tokenizer, _, _ = _judge(monkeypatch)
    judge.judge(premise="source evidence", hypothesis="stored claim")
    assert tokenizer.calls == [
        (
            "source evidence",
            "stored claim",
            {
                "add_special_tokens": True,
                "truncation": False,
                "return_tensors": "pt",
            },
        )
    ]


@pytest.mark.parametrize(
    ("premise", "hypothesis", "message"),
    [
        ("", "claim", "premise"),
        ("   ", "claim", "premise"),
        ("evidence", "", "hypothesis"),
        ("evidence", "   ", "hypothesis"),
    ],
)
def test_blank_semantic_input_is_rejected_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    premise: str,
    hypothesis: str,
    message: str,
) -> None:
    judge, _, model, _ = _judge(monkeypatch)
    with pytest.raises(SemanticInputError, match=message):
        judge.judge(premise=premise, hypothesis=hypothesis)
    assert model.model_calls == 0


def test_complete_over_limit_pair_is_rejected_without_leaking_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    premise = "private premise marker"
    hypothesis = "private hypothesis marker"
    judge, tokenizer, model, _ = _judge(
        monkeypatch,
        token_count=9,
        tokenizer_limit=12,
        model_limit=8,
    )
    with pytest.raises(SemanticInputTooLongError) as caught:
        judge.judge(premise=premise, hypothesis=hypothesis)
    assert caught.value.observed_tokens == 9
    assert caught.value.maximum_tokens == 8
    assert premise not in str(caught.value)
    assert hypothesis not in str(caught.value)
    assert tokenizer.calls[0][2]["truncation"] is False
    assert model.model_calls == 0


def test_argmax_score_usage_eval_mode_and_no_grad_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge, _, model, torch = _judge(
        monkeypatch,
        token_count=11,
        logits=(0.0, 2.0, 1.0),
    )
    judgment = judge.judge(premise="evidence", hypothesis="claim")
    expected_score = math.exp(2.0) / (math.exp(0.0) + math.exp(2.0) + math.exp(1.0))
    assert judgment.relation is SemanticRelation.ENTAILMENT
    assert judgment.score == pytest.approx(expected_score)
    assert judgment.usage.model_calls == 1
    assert judgment.usage.input_tokens == 11
    assert judgment.usage.output_tokens == 0
    assert model.eval_calls == 1
    assert model.model_calls == 1
    assert model.inference_modes == [True]
    assert torch.inference_mode_entries == 1


def test_repeated_fake_backend_inference_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge, _, _, _ = _judge(monkeypatch, logits=(3.0, 1.0, 0.0))
    first = judge.judge(premise="evidence", hypothesis="claim")
    second = judge.judge(premise="evidence", hypothesis="claim")
    assert first == second
    assert first.relation is SemanticRelation.CONTRADICTION


def test_loader_disables_remote_code_and_requires_safetensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}
    torch = FakeTorch()
    model = FakeModel(torch)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (args, kwargs)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            calls["model"] = (args, kwargs)
            return model

    transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForSequenceClassification=FakeAutoModel,
    )
    monkeypatch.setattr(
        local_nli,
        "import_module",
        lambda name: torch if name == "torch" else transformers,
    )
    backend = local_nli._load_backend(  # noqa: SLF001
        model_id="example/nli",
        revision="a" * 40,
        device="cpu",
    )
    assert backend.model is model
    assert calls["tokenizer"][1] == {
        "revision": "a" * 40,
        "trust_remote_code": False,
    }
    assert calls["model"][1] == {
        "revision": "a" * 40,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert model.devices == ["cpu"]
