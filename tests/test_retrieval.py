import ast
import hashlib
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from palintrace import cli
from palintrace.retrieval import RetrievalHit, RetrievalObservation, RetrievalUsage

RETRIEVAL_ROOT = Path("src/palintrace/retrieval")
QUERY_SHA256 = hashlib.sha256(b"Which memory is relevant?").hexdigest()


def _payload(*hits: RetrievalHit, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "audit-case-1",
        "query_sha256": QUERY_SHA256,
        "expected_memory_ids": ("m1",),
        "top_k": 2,
        "retriever_id": "fake-retriever",
        "retriever_version": "1",
        "hits": hits,
        "usage": RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    }
    payload.update(overrides)
    return payload


def _observation(*hits: RetrievalHit, **overrides: object) -> RetrievalObservation:
    return RetrievalObservation.model_validate(_payload(*hits, **overrides))


@pytest.mark.parametrize("rank", [0, -1, True, 1.0, "1"])
def test_hit_rank_is_a_strict_positive_integer(rank: object) -> None:
    with pytest.raises(ValidationError):
        RetrievalHit.model_validate({"memory_id": "m1", "rank": rank})


@pytest.mark.parametrize(
    "score",
    [True, "0.5", Decimal("0.5"), float("nan"), float("inf"), -float("inf")],
)
def test_hit_score_rejects_non_python_or_nonfinite_values(score: object) -> None:
    with pytest.raises(ValidationError):
        RetrievalHit.model_validate({"memory_id": "m1", "rank": 1, "score": score})


def test_hit_score_accepts_absent_float_and_integer_as_float() -> None:
    assert RetrievalHit(memory_id="m1", rank=1).score is None
    assert RetrievalHit(memory_id="m1", rank=1, score=0.25).score == 0.25
    integer_score = RetrievalHit(memory_id="m1", rank=1, score=2)
    assert integer_score.score == 2.0
    assert isinstance(integer_score.score, float)


def test_hit_has_no_content_target_or_defect_fields() -> None:
    assert set(RetrievalHit.model_fields) == {"memory_id", "rank", "score"}


@pytest.mark.parametrize("field", ["retrieval_calls", "candidate_count"])
@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_usage_fields_are_strict_nonnegative_integers(field: str, value: object) -> None:
    payload: dict[str, object] = {"retrieval_calls": 1, "candidate_count": 1}
    payload[field] = value

    with pytest.raises(ValidationError):
        RetrievalUsage.model_validate(payload)


def test_observation_has_no_query_text_or_decision_fields() -> None:
    assert set(RetrievalObservation.model_fields) == {
        "request_id",
        "query_sha256",
        "expected_memory_ids",
        "top_k",
        "retriever_id",
        "retriever_version",
        "hits",
        "usage",
    }


def test_observation_canonicalizes_targets_and_hits() -> None:
    observation = _observation(
        RetrievalHit(memory_id="m2", rank=2),
        RetrievalHit(memory_id="m1", rank=1, score=0.1),
        expected_memory_ids=("m2", "m1"),
    )

    assert observation.expected_memory_ids == ("m1", "m2")
    assert [hit.memory_id for hit in observation.hits] == ["m1", "m2"]


def test_observation_accepts_empty_hits() -> None:
    assert _observation().hits == ()


def test_observation_rejects_duplicate_ranks() -> None:
    with pytest.raises(ValidationError, match="ranks must be unique"):
        _observation(RetrievalHit(memory_id="m1", rank=1), RetrievalHit(memory_id="m2", rank=1))


def test_observation_rejects_duplicate_memory_ids() -> None:
    with pytest.raises(ValidationError, match="memory IDs must be unique"):
        _observation(RetrievalHit(memory_id="m1", rank=1), RetrievalHit(memory_id="m1", rank=2))


def test_observation_rejects_more_hits_than_top_k() -> None:
    with pytest.raises(ValidationError, match="must not exceed top_k"):
        _observation(
            RetrievalHit(memory_id="m1", rank=1),
            RetrievalHit(memory_id="m2", rank=2),
            top_k=1,
        )


@pytest.mark.parametrize("expected_memory_ids", [(), ("m1", "m1"), ("m1", " ")])
def test_observation_rejects_invalid_expected_memory_ids(
    expected_memory_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="expected_memory_ids"):
        _observation(expected_memory_ids=expected_memory_ids)


@pytest.mark.parametrize("field", ["request_id", "retriever_id", "retriever_version"])
def test_observation_rejects_blank_identity_strings(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        _observation(**{field: " "})


@pytest.mark.parametrize("query_sha256", ["abc", QUERY_SHA256.upper(), "g" * 64])
def test_observation_requires_a_lowercase_sha256_query_digest(query_sha256: str) -> None:
    with pytest.raises(ValidationError, match="query_sha256"):
        _observation(query_sha256=query_sha256)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.0, "1"])
def test_observation_top_k_is_a_strict_positive_integer(top_k: object) -> None:
    with pytest.raises(ValidationError):
        _observation(top_k=top_k)


def test_observation_serialization_is_byte_deterministic() -> None:
    hits = (RetrievalHit(memory_id="m2", rank=2), RetrievalHit(memory_id="m1", rank=1, score=1))
    first = _observation(*hits).to_json()
    second = _observation(*reversed(hits)).to_json()

    assert first == second
    assert RetrievalObservation.model_validate_json(first).to_json() == first


def test_retrieval_package_has_no_forbidden_dependencies_or_raw_access() -> None:
    forbidden_modules = {
        "palintrace.checkers",
        "palintrace.mutations",
        "palintrace.semantics",
    }
    forbidden_names = {
        "GoldLabel",
        "MutationManifest",
        "MutationRequest",
        "RetrievalProbe",
    }
    violations: list[str] = []
    for path in RETRIEVAL_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = {alias.name for alias in node.names}
                if modules & forbidden_modules:
                    violations.append(f"{path}:{node.lineno}:forbidden import")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_modules or any(
                    alias.name in forbidden_names for alias in node.names
                ):
                    violations.append(f"{path}:{node.lineno}:forbidden import")
            elif isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{path}:{node.lineno}:{node.id}")
            elif isinstance(node, ast.Attribute) and node.attr == "raw":
                violations.append(f"{path}:{node.lineno}:raw")
    assert violations == []


def test_retrieval_shadowing_is_not_a_store_checker() -> None:
    assert "retrieval_shadowing" not in cli.CHECKER_NAMES
