"""Optional local CPU NLI implementation of the semantic judge contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

from palintrace.semantics.models import (
    SemanticJudgment,
    SemanticRelation,
    SemanticUsage,
)

_SEMANTIC_LABELS = frozenset(relation.value for relation in SemanticRelation)
_LOCAL_INPUT_SAFETY_CAP = 4096


class SemanticJudgeError(RuntimeError):
    """Base error raised by a concrete semantic judge."""


class SemanticDependencyError(SemanticJudgeError):
    """The optional dependencies required by a semantic judge are unavailable."""


class SemanticModelConfigError(SemanticJudgeError):
    """A model or tokenizer configuration is unsafe or incompatible."""


class SemanticInputError(SemanticJudgeError, ValueError):
    """Semantic input cannot be judged as supplied."""


class SemanticInputTooLongError(SemanticInputError):
    """A complete premise/hypothesis token pair exceeds the model input limit."""

    def __init__(self, *, observed_tokens: int, maximum_tokens: int) -> None:
        self.observed_tokens = observed_tokens
        self.maximum_tokens = maximum_tokens
        super().__init__(
            "semantic input exceeds the supported token limit: "
            f"observed_tokens={observed_tokens}, maximum_tokens={maximum_tokens}"
        )


@dataclass(frozen=True)
class _LocalNLIBackend:
    torch: ModuleType
    tokenizer: Any
    model: Any


def _load_backend(*, model_id: str, revision: str, device: str) -> _LocalNLIBackend:
    """Load optional libraries and one pinned model behind a testable boundary."""

    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
    except ImportError as exc:
        raise SemanticDependencyError(
            "local NLI requires the 'semantic-local' optional dependencies; "
            "install palintrace[semantic-local]"
        ) from exc

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            use_safetensors=True,
        )
        model.to(device)
    except ImportError as exc:
        raise SemanticDependencyError(
            "local NLI tokenizer dependencies are unavailable; "
            "install palintrace[semantic-local]"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise SemanticJudgeError(
            f"failed to load pinned local NLI model {model_id!r} at revision {revision!r}"
        ) from exc
    return _LocalNLIBackend(torch=torch, tokenizer=tokenizer, model=model)


def _mapping_from_id2label(value: object) -> dict[int, SemanticRelation]:
    if not isinstance(value, Mapping) or len(value) != len(_SEMANTIC_LABELS):
        raise SemanticModelConfigError(
            "model id2label must expose exactly contradiction, entailment, and neutral"
        )

    mapping: dict[int, SemanticRelation] = {}
    for raw_class_id, raw_label in value.items():
        if isinstance(raw_class_id, bool):
            raise SemanticModelConfigError("model id2label class IDs must be integers")
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError) as exc:
            raise SemanticModelConfigError("model id2label class IDs must be integers") from exc
        if isinstance(raw_class_id, float) and not raw_class_id.is_integer():
            raise SemanticModelConfigError("model id2label class IDs must be integers")
        if not isinstance(raw_label, str) or raw_label.casefold() not in _SEMANTIC_LABELS:
            raise SemanticModelConfigError(
                "model labels must be exactly contradiction, entailment, and neutral"
            )
        relation = SemanticRelation(raw_label.casefold())
        if relation in mapping.values() or class_id in mapping:
            raise SemanticModelConfigError(
                "model label mapping contains a duplicate relation or ID"
            )
        mapping[class_id] = relation

    if set(mapping.values()) != set(SemanticRelation):
        raise SemanticModelConfigError(
            "model label mapping is missing contradiction, entailment, or neutral"
        )
    return mapping


def _mapping_from_label2id(value: object) -> dict[int, SemanticRelation]:
    if not isinstance(value, Mapping) or len(value) != len(_SEMANTIC_LABELS):
        raise SemanticModelConfigError(
            "model label2id must expose exactly contradiction, entailment, and neutral"
        )
    reversed_mapping = {raw_class_id: raw_label for raw_label, raw_class_id in value.items()}
    if len(reversed_mapping) != len(value):
        raise SemanticModelConfigError("model label mapping contains a duplicate relation or ID")
    return _mapping_from_id2label(reversed_mapping)


def _validated_label_mapping(config: object) -> dict[int, SemanticRelation]:
    """Validate semantic names and consistency without inferring numerical order."""

    id2label = getattr(config, "id2label", None)
    label2id = getattr(config, "label2id", None)
    id_mapping = _mapping_from_id2label(id2label) if id2label else None
    label_mapping = _mapping_from_label2id(label2id) if label2id else None
    if id_mapping is None and label_mapping is None:
        raise SemanticModelConfigError("model configuration has no semantic label mapping")
    if id_mapping is not None and label_mapping is not None and id_mapping != label_mapping:
        raise SemanticModelConfigError("model id2label and label2id mappings disagree")
    if id_mapping is not None:
        return id_mapping
    assert label_mapping is not None
    return label_mapping


def _positive_limit(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _effective_input_limit(*, tokenizer: object, model: object) -> int:
    tokenizer_limit = _positive_limit(getattr(tokenizer, "model_max_length", None))
    config = getattr(model, "config", None)
    model_limit = _positive_limit(getattr(config, "max_position_embeddings", None))
    configured_limits = [
        limit for limit in (tokenizer_limit, model_limit) if limit is not None
    ]
    if not configured_limits:
        raise SemanticModelConfigError(
            "model and tokenizer do not declare a positive supported input limit"
        )
    return min(*configured_limits, _LOCAL_INPUT_SAFETY_CAP)


def _single_pair_token_count(input_ids: object) -> int:
    shape = getattr(input_ids, "shape", None)
    if not isinstance(shape, Sequence) or len(shape) != 2 or shape[0] != 1:
        raise SemanticJudgeError("tokenizer did not return one encoded input pair")
    token_count = shape[1]
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise SemanticJudgeError("tokenizer returned an invalid input token count")
    return token_count


def _three_logits(logits: Any) -> tuple[float, float, float]:
    try:
        rows = logits.detach().cpu().tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise SemanticJudgeError("local NLI model returned invalid logits") from exc
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise SemanticJudgeError("local NLI model must return one row of three logits")
    row = rows[0]
    if len(row) != len(_SEMANTIC_LABELS):
        raise SemanticJudgeError("local NLI model must return exactly three logits")
    values = tuple(float(value) for value in row)
    if not all(math.isfinite(value) for value in values):
        raise SemanticJudgeError("local NLI model returned non-finite logits")
    return values[0], values[1], values[2]


def _softmax(values: tuple[float, float, float]) -> tuple[float, float, float]:
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    total = sum(exponentials)
    return (
        exponentials[0] / total,
        exponentials[1] / total,
        exponentials[2] / total,
    )


class LocalNLISemanticJudge:
    """Classify one premise/hypothesis pair with a pinned local CPU NLI model."""

    def __init__(self, *, model_id: str, revision: str, device: str = "cpu") -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise SemanticModelConfigError("model_id must be a nonblank string")
        if not isinstance(revision, str) or not revision.strip():
            raise SemanticModelConfigError("revision must be a nonblank string")
        if device != "cpu":
            raise SemanticModelConfigError("Part 4B local NLI supports only device='cpu'")

        self.judge_id = f"hf-nli:{model_id}"
        self.judge_version = revision
        self.device = device
        self._backend = _load_backend(model_id=model_id, revision=revision, device=device)
        self._class_relations = _validated_label_mapping(self._backend.model.config)
        if set(self._class_relations) != set(range(len(_SEMANTIC_LABELS))):
            raise SemanticModelConfigError("model class IDs must form the contiguous range 0..2")
        self.maximum_tokens = _effective_input_limit(
            tokenizer=self._backend.tokenizer,
            model=self._backend.model,
        )
        self._backend.model.eval()

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        """Return deterministic three-class NLI inference for one complete text pair."""

        if not isinstance(premise, str) or not premise.strip():
            raise SemanticInputError("premise must be a nonblank string")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise SemanticInputError("hypothesis must be a nonblank string")

        try:
            encoded = self._backend.tokenizer(
                premise,
                hypothesis,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            token_count = _single_pair_token_count(encoded["input_ids"])
        except SemanticJudgeError:
            raise
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise SemanticJudgeError("failed to tokenize semantic input pair") from exc
        if token_count > self.maximum_tokens:
            raise SemanticInputTooLongError(
                observed_tokens=token_count,
                maximum_tokens=self.maximum_tokens,
            )

        try:
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with self._backend.torch.inference_mode():
                logits = self._backend.model(**encoded).logits
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise SemanticJudgeError("local NLI inference failed") from exc

        values = _three_logits(logits)
        scores = _softmax(values)
        selected_class = max(range(len(scores)), key=scores.__getitem__)
        return SemanticJudgment(
            relation=self._class_relations[selected_class],
            score=scores[selected_class],
            usage=SemanticUsage(
                model_calls=1,
                input_tokens=token_count,
                output_tokens=0,
            ),
        )
