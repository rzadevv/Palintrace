import ast
import hashlib
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from memlint import cli
from memlint.models import NormalizedMemory, NormalizedStore
from memlint.mutations import DistractorFamily, MutationRequest, mutate
from memlint.retrieval import (
    RetrievalAuditRequest,
    RetrievalHit,
    RetrievalInputError,
    RetrievalObservation,
    RetrievalObservationError,
    RetrievalResponse,
    RetrievalUsage,
    Retriever,
    retriever_identity,
    run_retrieval_audit,
    validate_retrieval_audit_request,
)
from memlint.taxonomy import DefectClass

RETRIEVAL_ROOT = Path("src/memlint/retrieval")


def _store(*memory_ids: str) -> NormalizedStore:
    return NormalizedStore(
        adapter="test",
        memories=tuple(
            NormalizedMemory(id=memory_id, content=f"Visible memory {memory_id}.")
            for memory_id in memory_ids
        ),
    )


def _request(
    *,
    expected_memory_ids: tuple[str, ...] = ("m1",),
    top_k: int = 2,
    query: str = "Which memory is relevant?",
) -> RetrievalAuditRequest:
    return RetrievalAuditRequest(
        request_id="audit-case-1",
        query=query,
        expected_memory_ids=expected_memory_ids,
        top_k=top_k,
    )


def _response(*hits: RetrievalHit) -> RetrievalResponse:
    return RetrievalResponse(
        hits=hits,
        usage=RetrievalUsage(retrieval_calls=1, candidate_count=len(hits)),
    )


class FakeRetriever:
    retriever_id = "fake-retriever"
    retriever_version = "1"

    def __init__(self, response: RetrievalResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def retrieve(self, *, query: str, top_k: int) -> RetrievalResponse:
        self.calls.append({"query": query, "top_k": top_k})
        return self.response


class DynamicIdentityRetriever(FakeRetriever):
    def __init__(
        self,
        response: RetrievalResponse,
        *,
        retriever_id: object,
        retriever_version: object,
    ) -> None:
        super().__init__(response)
        self.retriever_id = retriever_id  # type: ignore[assignment]
        self.retriever_version = retriever_version  # type: ignore[assignment]


def test_request_has_exact_minimal_fields_and_preserves_exact_text() -> None:
    request = RetrievalAuditRequest(
        request_id=" audit-case ",
        query="  Exact query?\n",
        expected_memory_ids=("m2", "m1"),
        top_k=3,
    )

    assert set(RetrievalAuditRequest.model_fields) == {
        "request_id",
        "query",
        "expected_memory_ids",
        "top_k",
    }
    assert request.request_id == " audit-case "
    assert request.query == "  Exact query?\n"
    assert request.expected_memory_ids == ("m1", "m2")
    with pytest.raises(ValidationError):
        RetrievalAuditRequest.model_validate(
            {
                **request.model_dump(),
                "mutation_id": "forbidden",
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("request_id", ""), ("request_id", "  "), ("query", ""), ("query", "\t")],
)
def test_request_rejects_blank_required_strings(field: str, value: str) -> None:
    payload = _request().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match="must not be blank"):
        RetrievalAuditRequest.model_validate(payload)


@pytest.mark.parametrize(
    "expected_memory_ids",
    [(), ("m1", "m1"), ("m1", " ")],
)
def test_request_rejects_invalid_expected_memory_ids(
    expected_memory_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="expected_memory_ids"):
        _request(expected_memory_ids=expected_memory_ids)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.0, "1"])
def test_request_top_k_is_a_strict_positive_integer(top_k: object) -> None:
    payload = _request().model_dump()
    payload["top_k"] = top_k

    with pytest.raises(ValidationError):
        RetrievalAuditRequest.model_validate(payload)


def test_missing_expected_target_is_invalid_before_retrieval() -> None:
    request = _request(expected_memory_ids=("missing",))
    retriever = FakeRetriever(_response())

    with pytest.raises(RetrievalInputError, match="absent from the audited store"):
        run_retrieval_audit(store=_store("m1"), request=request, retriever=retriever)

    assert retriever.calls == []


def test_request_validation_accepts_every_target_in_store() -> None:
    validate_retrieval_audit_request(
        _request(expected_memory_ids=("m2", "m1")),
        _store("m1", "m2"),
    )


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


def test_hit_and_response_have_no_content_target_or_defect_fields() -> None:
    assert set(RetrievalHit.model_fields) == {"memory_id", "rank", "score"}
    assert set(RetrievalResponse.model_fields) == {"hits", "usage"}


def test_response_canonicalizes_hits_by_authoritative_rank() -> None:
    response = _response(
        RetrievalHit(memory_id="m3", rank=3, score=0.9),
        RetrievalHit(memory_id="m1", rank=1, score=0.1),
        RetrievalHit(memory_id="m2", rank=2),
    )

    assert [hit.memory_id for hit in response.hits] == ["m1", "m2", "m3"]


def test_response_rejects_duplicate_ranks() -> None:
    with pytest.raises(ValidationError, match="ranks must be unique"):
        _response(
            RetrievalHit(memory_id="m1", rank=1),
            RetrievalHit(memory_id="m2", rank=1),
        )


def test_response_rejects_duplicate_memory_ids() -> None:
    with pytest.raises(ValidationError, match="memory IDs must be unique"):
        _response(
            RetrievalHit(memory_id="m1", rank=1),
            RetrievalHit(memory_id="m1", rank=2),
        )


@pytest.mark.parametrize("field", ["retrieval_calls", "candidate_count"])
@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_usage_fields_are_strict_nonnegative_integers(field: str, value: object) -> None:
    payload: dict[str, object] = {"retrieval_calls": 1, "candidate_count": 1}
    payload[field] = value

    with pytest.raises(ValidationError):
        RetrievalUsage.model_validate(payload)


def test_unknown_returned_memory_id_is_an_observation_error() -> None:
    retriever = FakeRetriever(_response(RetrievalHit(memory_id="unknown", rank=1)))

    with pytest.raises(RetrievalObservationError, match="audited store snapshot"):
        run_retrieval_audit(
            store=_store("m1"),
            request=_request(),
            retriever=retriever,
        )


def test_empty_and_fewer_than_top_k_results_are_valid_observations() -> None:
    store = _store("m1", "m2")
    empty = run_retrieval_audit(
        store=store,
        request=_request(top_k=2),
        retriever=FakeRetriever(_response()),
    )
    fewer = run_retrieval_audit(
        store=store,
        request=_request(top_k=2),
        retriever=FakeRetriever(_response(RetrievalHit(memory_id="m2", rank=1))),
    )

    assert empty.hits == ()
    assert [hit.memory_id for hit in fewer.hits] == ["m2"]


def test_more_than_top_k_results_are_rejected() -> None:
    retriever = FakeRetriever(
        _response(
            RetrievalHit(memory_id="m1", rank=1),
            RetrievalHit(memory_id="m2", rank=2),
        )
    )

    with pytest.raises(RetrievalObservationError, match="more hits than requested"):
        run_retrieval_audit(
            store=_store("m1", "m2"),
            request=_request(top_k=1),
            retriever=retriever,
        )


def test_retriever_identity_is_validated_and_preserved() -> None:
    retriever = DynamicIdentityRetriever(
        _response(),
        retriever_id=" retriever ",
        retriever_version=" version ",
    )
    assert retriever_identity(cast(Retriever, retriever)) == (
        " retriever ",
        " version ",
    )


@pytest.mark.parametrize(
    ("retriever_id", "retriever_version"),
    [("", "1"), (" ", "1"), ("id", ""), ("id", " "), (1, "1"), ("id", 1)],
)
def test_invalid_retriever_identity_fails_before_call(
    retriever_id: object,
    retriever_version: object,
) -> None:
    retriever = DynamicIdentityRetriever(
        _response(),
        retriever_id=retriever_id,
        retriever_version=retriever_version,
    )

    with pytest.raises(RetrievalInputError, match="nonblank string"):
        run_retrieval_audit(
            store=_store("m1"),
            request=_request(),
            retriever=cast(Retriever, retriever),
        )

    assert retriever.calls == []


def test_retriever_receives_only_query_and_top_k() -> None:
    request = _request(expected_memory_ids=("m1",), top_k=4, query="Exact query")
    retriever = FakeRetriever(_response())

    run_retrieval_audit(store=_store("m1"), request=request, retriever=retriever)

    assert retriever.calls == [{"query": "Exact query", "top_k": 4}]
    assert not ({"expected_memory_ids", "request_id", "gold", "distractor_ids"} & set(
        retriever.calls[0]
    ))


def test_observation_hashes_the_exact_utf8_query_without_storing_it() -> None:
    query = "  Welchen Editor bevorzugt die Nutzerin? ü\n"
    observation = run_retrieval_audit(
        store=_store("m1"),
        request=_request(query=query),
        retriever=FakeRetriever(_response()),
    )

    expected_sha = hashlib.sha256(query.encode("utf-8")).hexdigest()
    serialized = observation.to_json()
    assert observation.query_sha256 == expected_sha
    assert query not in serialized
    assert "query" not in RetrievalObservation.model_fields


def test_observation_records_targets_and_actual_hits_without_a_decision() -> None:
    observation = run_retrieval_audit(
        store=_store("m1", "m2"),
        request=_request(expected_memory_ids=("m1",), top_k=2),
        retriever=FakeRetriever(_response(RetrievalHit(memory_id="m2", rank=1))),
    )

    assert observation.expected_memory_ids == ("m1",)
    assert [hit.memory_id for hit in observation.hits] == ["m2"]
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


def test_observation_serialization_is_byte_deterministic() -> None:
    store = _store("m1", "m2")
    request = _request(expected_memory_ids=("m2", "m1"), top_k=2)
    response = _response(
        RetrievalHit(memory_id="m2", rank=2),
        RetrievalHit(memory_id="m1", rank=1, score=1),
    )

    first = run_retrieval_audit(
        store=store,
        request=request,
        retriever=FakeRetriever(response),
    ).to_json()
    second = run_retrieval_audit(
        store=store,
        request=request,
        retriever=FakeRetriever(response),
    ).to_json()

    assert first == second
    assert RetrievalObservation.model_validate_json(first).to_json() == first


def test_retrieval_package_has_no_forbidden_dependencies_or_raw_access() -> None:
    forbidden_modules = {
        "memlint.checkers",
        "memlint.mutations",
        "memlint.semantics",
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


def test_retrieval_architecture_has_contract_only() -> None:
    assert {path.name for path in RETRIEVAL_ROOT.glob("*.py")} == {
        "__init__.py",
        "base.py",
        "models.py",
    }
    assert "RetrievalShadowingChecker" not in {
        node.name
        for path in Path("src/memlint").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert "retrieval_shadowing" not in cli.CHECKER_NAMES


def test_part_two_probe_can_be_translated_only_at_the_test_boundary() -> None:
    base_store = NormalizedStore(
        adapter="test",
        memories=(
            NormalizedMemory(
                id="editor-neovim",
                content="User's favorite editor is Neovim.",
            ),
        ),
    )
    mutation = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.RETRIEVAL_SHADOWING,
            target_memory_id="editor-neovim",
            query="Which editor does the user prefer?",
            distractor_family=DistractorFamily.EDITOR,
        ),
    )
    probe = mutation.manifest.retrieval_probe
    assert probe is not None

    audit_request = RetrievalAuditRequest(
        request_id="test-cross-check",
        query=probe.query,
        expected_memory_ids=probe.expected_memory_ids,
        top_k=3,
    )
    retriever = FakeRetriever(_response())
    observation = run_retrieval_audit(
        store=mutation.mutated_store,
        request=audit_request,
        retriever=retriever,
    )

    assert observation.expected_memory_ids == probe.expected_memory_ids
    assert observation.hits == ()
    assert retriever.calls == [{"query": probe.query, "top_k": 3}]
    assert "distractor_memory_ids" not in audit_request.model_dump()
    assert not ({"gold_label", "manifest", "mutation_id"} & set(
        observation.model_dump()
    ))
