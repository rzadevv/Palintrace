from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memlint.mutations.injection import TEMPLATES as PART_2_INJECTION_TEMPLATES
from tools.evaluate_injected_instruction_models import (
    CANDIDATES,
    DEVICE,
    INJECTION_PROBE_SHA256,
    MAXIMUM_BENIGN_FALSE_POSITIVES,
    MAXIMUM_MODEL_INPUT_CONFIG_ERRORS,
    MAXIMUM_QUOTED_FALSE_POSITIVES,
    MAXIMUM_TOTAL_SAFE_FALSE_POSITIVES,
    MINIMUM_INJECTIONS_DETECTED,
    InjectionClass,
    InjectionProbeCase,
    InjectionReport,
    InputTooLongError,
    LabelMappingSource,
    LocalInjectionClassifier,
    ModelCandidate,
    ModelConfigError,
    ModelEvaluation,
    _load_with_factories,
    effective_input_limit,
    evaluate_cases,
    load_cases,
    passes_readiness_gate,
    select_ready_model,
    softmax,
    validate_candidate_definitions,
    validate_fixture_hash,
    validated_label_mapping,
)


def _report(
    *,
    detected: int = 7,
    quoted_false: int = 0,
    benign_false: int = 1,
    errors: int = 0,
    latency_ms: float = 1.0,
) -> InjectionReport:
    return InjectionReport(
        injection_detected_ids=tuple(f"I{index}" for index in range(1, detected + 1)),
        missed_injection_ids=tuple(
            f"I{index}" for index in range(detected + 1, 9)
        ),
        quoted_false_positive_ids=tuple(
            f"Q{index}" for index in range(1, quoted_false + 1)
        ),
        benign_false_positive_ids=tuple(
            f"B{index}" for index in range(1, benign_false + 1)
        ),
        correct_ids=(),
        model_input_config_errors=tuple(
            f"E{index}:input" for index in range(1, errors + 1)
        ),
        median_cpu_latency_ms=latency_ms,
        total_model_calls=24,
        total_input_tokens=1,
    )


def _evaluation(
    candidate: ModelCandidate,
    report: InjectionReport,
) -> ModelEvaluation:
    return ModelEvaluation(candidate=candidate, results=(), report=report)


def _protectai_config(**overrides: Any) -> SimpleNamespace:
    values = {
        "id2label": {0: "SAFE", 1: "INJECTION"},
        "label2id": {"SAFE": 0, "INJECTION": 1},
        "max_position_embeddings": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _neuralchemy_config(**overrides: Any) -> SimpleNamespace:
    values = {
        "id2label": {0: "LABEL_0", 1: "LABEL_1"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "max_position_embeddings": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exact_two_candidate_definitions_and_pinned_revisions() -> None:
    assert tuple(
        (
            candidate.name,
            candidate.model_id,
            candidate.revision,
            candidate.license,
            candidate.safetensors_bytes,
            candidate.label_mapping_source,
        )
        for candidate in CANDIDATES
    ) == (
        (
            "protectai",
            "protectai/deberta-v3-base-prompt-injection-v2",
            "e6535ca4ce3ba852083e75ec585d7c8aeb4be4c5",
            "Apache-2.0",
            737_719_272,
            LabelMappingSource.MODEL_CONFIG,
        ),
        (
            "neuralchemy",
            "neuralchemy/prompt-injection-deberta",
            "e271c829b9fd9ce9031f12ab29350261f3ed9457",
            "Apache-2.0",
            283_805_636,
            LabelMappingSource.PINNED_MODEL_CARD,
        ),
    )
    assert tuple(InjectionClass) == (
        InjectionClass.SAFE,
        InjectionClass.INJECTION,
    )
    validate_candidate_definitions(CANDIDATES)


def test_mutable_main_and_abbreviated_revisions_are_forbidden() -> None:
    with pytest.raises(ValueError, match="mutable 'main'"):
        validate_candidate_definitions((replace(CANDIDATES[0], revision="main"),))
    with pytest.raises(ValueError, match="full immutable"):
        validate_candidate_definitions((replace(CANDIDATES[0], revision="e6535ca"),))


def test_protectai_semantic_config_mapping_is_required() -> None:
    assert validated_label_mapping(CANDIDATES[0], _protectai_config()) == {
        0: InjectionClass.SAFE,
        1: InjectionClass.INJECTION,
    }

    with pytest.raises(ModelConfigError, match="0=SAFE and 1=INJECTION"):
        validated_label_mapping(
            CANDIDATES[0],
            _protectai_config(
                id2label={0: "INJECTION", 1: "SAFE"},
                label2id={"INJECTION": 0, "SAFE": 1},
            ),
        )
    with pytest.raises(ModelConfigError, match="disagree"):
        validated_label_mapping(
            CANDIDATES[0],
            _protectai_config(label2id={"SAFE": 1, "INJECTION": 0}),
        )


def test_neuralchemy_generic_mapping_is_candidate_specific() -> None:
    assert validated_label_mapping(CANDIDATES[1], _neuralchemy_config()) == {
        0: InjectionClass.SAFE,
        1: InjectionClass.INJECTION,
    }

    unknown = replace(
        CANDIDATES[1],
        name="unknown",
        model_id="example/unknown-generic-classifier",
    )
    with pytest.raises(ModelConfigError, match="only for the pinned Neuralchemy"):
        validated_label_mapping(unknown, _neuralchemy_config())
    with pytest.raises(ModelConfigError, match="0=SAFE and 1=INJECTION"):
        validated_label_mapping(CANDIDATES[0], _neuralchemy_config())


def test_label2id_is_optional_but_must_be_equivalent_when_present() -> None:
    config = SimpleNamespace(
        id2label={"0": "SAFE", "1": "INJECTION"},
        max_position_embeddings=512,
    )
    assert validated_label_mapping(CANDIDATES[0], config) == {
        0: InjectionClass.SAFE,
        1: InjectionClass.INJECTION,
    }


class _Factory:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
        self.calls.append((model_id, kwargs))
        return self.result


class _LoadTokenizer:
    model_max_length = 512


class _LoadModel:
    def __init__(self) -> None:
        self.config = _protectai_config()
        self.device: str | None = None
        self.eval_called = False

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        self.eval_called = True


def test_safe_pinned_cpu_loading_uses_sequence_classifier_and_eval() -> None:
    tokenizer = _LoadTokenizer()
    model = _LoadModel()
    tokenizer_factory = _Factory(tokenizer)
    model_factory = _Factory(model)

    loaded = _load_with_factories(
        CANDIDATES[0],
        tokenizer_factory,
        model_factory,
        object(),
    )

    assert isinstance(loaded, LocalInjectionClassifier)
    assert tokenizer_factory.calls == [
        (
            "protectai/deberta-v3-base-prompt-injection-v2",
            {
                "revision": "e6535ca4ce3ba852083e75ec585d7c8aeb4be4c5",
                "trust_remote_code": False,
            },
        )
    ]
    assert model_factory.calls == [
        (
            "protectai/deberta-v3-base-prompt-injection-v2",
            {
                "revision": "e6535ca4ce3ba852083e75ec585d7c8aeb4be4c5",
                "trust_remote_code": False,
                "use_safetensors": True,
            },
        )
    ]
    assert model.device == DEVICE == "cpu"
    assert model.eval_called


class _Tensor:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> _Tensor:
        self.device = device
        return self


class _InputIds(_Tensor):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.shape = (1, token_count)


class _Tokenizer:
    model_max_length = 512

    def __init__(self, token_count: int = 11) -> None:
        self.token_count = token_count
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, content: str, **kwargs: Any) -> dict[str, _Tensor]:
        self.calls.append((content, kwargs))
        return {
            "input_ids": _InputIds(self.token_count),
            "attention_mask": _Tensor(),
        }


class _Logits:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def detach(self) -> _Logits:
        return self

    def cpu(self) -> _Logits:
        return self

    def tolist(self) -> list[list[float]]:
        return self.values


class _InferenceContext:
    def __init__(self, torch: _Torch) -> None:
        self.torch = torch

    def __enter__(self) -> None:
        self.torch.active = True
        self.torch.entries += 1

    def __exit__(self, *args: object) -> None:
        self.torch.active = False


class _Torch:
    def __init__(self) -> None:
        self.active = False
        self.entries = 0

    def inference_mode(self) -> _InferenceContext:
        return _InferenceContext(self)


class _InferenceModel:
    def __init__(self, torch: _Torch, logits: list[list[float]]) -> None:
        self.config = _protectai_config()
        self.torch = torch
        self.logits = logits
        self.calls = 0
        self.encoded: dict[str, _Tensor] | None = None

    def __call__(self, **encoded: _Tensor) -> SimpleNamespace:
        assert self.torch.active
        self.calls += 1
        self.encoded = encoded
        return SimpleNamespace(logits=_Logits(self.logits))


def test_complete_input_argmax_softmax_and_inference_mode() -> None:
    torch = _Torch()
    tokenizer = _Tokenizer()
    model = _InferenceModel(torch, [[1.0, 3.0]])
    classifier = LocalInjectionClassifier(CANDIDATES[0], tokenizer, model, torch)

    result = classifier.classify("complete memory content")

    assert tokenizer.calls == [
        (
            "complete memory content",
            {
                "add_special_tokens": True,
                "truncation": False,
                "return_tensors": "pt",
            },
        )
    ]
    assert torch.entries == 1
    assert not torch.active
    assert model.calls == 1
    assert model.encoded is not None
    assert all(tensor.device == "cpu" for tensor in model.encoded.values())
    assert result.predicted is InjectionClass.INJECTION
    assert result.selected_class_score == pytest.approx(0.8807970779778823)
    assert result.input_tokens == 11


def test_argmax_uses_label_mapping_without_threshold() -> None:
    torch = _Torch()
    classifier = LocalInjectionClassifier(
        CANDIDATES[0],
        _Tokenizer(),
        _InferenceModel(torch, [[0.0001, 0.0]]),
        torch,
    )

    result = classifier.classify("small safe argmax")

    assert result.predicted is InjectionClass.SAFE
    assert result.selected_class_score == pytest.approx(0.5000249999999792)


def test_softmax_is_stable_and_diagnostic_only() -> None:
    assert softmax((1001.0, 1000.0)) == pytest.approx(
        (0.7310585786300049, 0.2689414213699951)
    )
    assert math.isclose(sum(softmax((-1000.0, -1001.0))), 1.0)


def test_model_limit_is_enforced_without_truncation_or_model_call() -> None:
    torch = _Torch()
    tokenizer = _Tokenizer(token_count=513)
    model = _InferenceModel(torch, [[0.0, 1.0]])
    classifier = LocalInjectionClassifier(CANDIDATES[0], tokenizer, model, torch)

    with pytest.raises(InputTooLongError) as raised:
        classifier.classify("too long")

    assert raised.value.observed_tokens == 513
    assert raised.value.maximum_tokens == 512
    assert tokenizer.calls[0][1]["truncation"] is False
    assert model.calls == 0
    assert torch.entries == 0


def test_over_limit_case_is_recorded_as_unassessable_probe_error() -> None:
    torch = _Torch()
    classifier = LocalInjectionClassifier(
        CANDIDATES[0],
        _Tokenizer(token_count=513),
        _InferenceModel(torch, [[0.0, 1.0]]),
        torch,
    )
    case = InjectionProbeCase(
        case_id="I1",
        group="I",
        content="complete but over-limit memory",
        expected=InjectionClass.INJECTION,
    )

    evaluation = evaluate_cases(CANDIDATES[0], classifier, (case,))

    result = evaluation.results[0]
    assert result.predicted is None
    assert result.selected_class_score is None
    assert result.input_tokens == 513
    assert result.error == "input-too-long"
    assert not result.model_called
    assert evaluation.report.model_input_config_errors == ("I1:input-too-long",)
    assert evaluation.report.total_model_calls == 0
    assert not passes_readiness_gate(evaluation.report)


def test_effective_limit_respects_smallest_declared_limit() -> None:
    tokenizer = SimpleNamespace(model_max_length=256)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    assert effective_input_limit(tokenizer=tokenizer, model=model) == 256

    tokenizer.model_max_length = 1024
    assert effective_input_limit(tokenizer=tokenizer, model=model) == 512

    with pytest.raises(ModelConfigError, match="positive supported input limit"):
        effective_input_limit(
            tokenizer=SimpleNamespace(model_max_length=None),
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=None)),
        )


def test_exact_fixture_hash_counts_ids_and_groups() -> None:
    assert INJECTION_PROBE_SHA256 == (
        "e33961cfa4fe3eafdda4c6cd9f0d0096551f3469bc41957a93f39ed68c54eb3d"
    )
    validate_fixture_hash()
    cases = load_cases()

    assert len(cases) == 24
    assert tuple(case.case_id for case in cases) == tuple(
        f"{group}{index}" for group in ("I", "Q", "B") for index in range(1, 9)
    )
    assert tuple(sum(case.group == group for case in cases) for group in ("I", "Q", "B")) == (
        8,
        8,
        8,
    )
    assert sum(case.expected is InjectionClass.INJECTION for case in cases) == 8
    assert sum(case.expected is InjectionClass.SAFE for case in cases) == 16


def test_exact_part_2_injection_templates_are_absent_from_selection_fixture() -> None:
    fixture_contents = {case.content for case in load_cases()}
    assert set(PART_2_INJECTION_TEMPLATES.values()).isdisjoint(fixture_contents)


def test_readiness_gate_is_pre_frozen_at_every_boundary() -> None:
    assert MINIMUM_INJECTIONS_DETECTED == 7
    assert MAXIMUM_QUOTED_FALSE_POSITIVES == 0
    assert MAXIMUM_BENIGN_FALSE_POSITIVES == 1
    assert MAXIMUM_TOTAL_SAFE_FALSE_POSITIVES == 1
    assert MAXIMUM_MODEL_INPUT_CONFIG_ERRORS == 0
    assert passes_readiness_gate(_report())

    assert not passes_readiness_gate(_report(detected=6))
    assert not passes_readiness_gate(_report(quoted_false=1, benign_false=0))
    assert not passes_readiness_gate(_report(benign_false=2))
    assert not passes_readiness_gate(_report(errors=1))


def test_selection_considers_only_gate_passing_candidates() -> None:
    failing_smaller_faster = _evaluation(
        CANDIDATES[1],
        _report(detected=6, benign_false=0, latency_ms=0.01),
    )
    passing = _evaluation(CANDIDATES[0], _report(latency_ms=100.0))

    assert select_ready_model((failing_smaller_faster, passing)) is passing


def test_selection_returns_none_when_neither_candidate_passes() -> None:
    evaluations = (
        _evaluation(CANDIDATES[0], _report(quoted_false=1, benign_false=0)),
        _evaluation(CANDIDATES[1], _report(detected=6)),
    )
    assert select_ready_model(evaluations) is None


def test_selection_ranking_uses_recall_then_size_then_latency_after_precision() -> None:
    one_benign_false = _evaluation(CANDIDATES[1], _report(benign_false=1))
    zero_false = _evaluation(CANDIDATES[0], _report(benign_false=0))
    assert select_ready_model((one_benign_false, zero_false)) is zero_false

    detected_seven = _evaluation(CANDIDATES[1], _report(detected=7, benign_false=0))
    detected_eight = _evaluation(CANDIDATES[0], _report(detected=8, benign_false=0))
    assert select_ready_model((detected_seven, detected_eight)) is detected_eight

    protectai = _evaluation(CANDIDATES[0], _report(benign_false=0))
    neuralchemy = _evaluation(CANDIDATES[1], _report(benign_false=0))
    assert select_ready_model((protectai, neuralchemy)) is neuralchemy

    slower = _evaluation(
        replace(CANDIDATES[1], name="slower"),
        _report(benign_false=0, latency_ms=2.0),
    )
    faster = _evaluation(
        replace(CANDIDATES[1], name="faster"),
        _report(benign_false=0, latency_ms=1.0),
    )
    assert select_ready_model((slower, faster)) is faster


def test_all_prior_probe_hashes_are_unchanged() -> None:
    expected = {
        Path("tests/fixtures/semantic_probe_v0.1.json"): (
            "e277c04b9b18d5717f94b524e65467b0240ec515961abed49398132dc8777fb4"
        ),
        Path("tests/fixtures/evidence_composition_probe_v0.1.json"): (
            "84f824548b1ae2ee2d75fc04e5069bb1d8e45580092515a6c1aaa5d656675237"
        ),
        Path("tests/fixtures/contradiction_pair_probe_v0.1.json"): (
            "0744755a747164a9ff646a094b78fdf132e2b89de09556cf17f0189054d72744"
        ),
        Path("tests/fixtures/instruction_contradiction_probe_v0.1.json"): (
            "e381b9f0091b3277602ea30d9eef07766e49823c56c7747128e1936bd6202c50"
        ),
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_tool_has_no_production_import_leakage_or_raw_access() -> None:
    path = Path("tools/evaluate_injected_instruction_models.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules: list[str] = []
    raw_attributes: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Attribute) and node.attr == "raw":
            raw_attributes.append(node.lineno)

    assert not any(module.startswith("memlint.checkers") for module in imported_modules)
    assert not any(module.startswith("memlint.mutations") for module in imported_modules)
    assert raw_attributes == []
    source = path.read_text(encoding="utf-8")
    assert "MutationManifest" not in source
    assert "AutoModelForCausalLM" not in source
    assert ".generate(" not in source
