"""Provider-independent retrieval protocol and audit execution boundary."""

from __future__ import annotations

import hashlib
from typing import Protocol

from palintrace.models import NormalizedStore
from palintrace.retrieval.models import (
    RetrievalAuditRequest,
    RetrievalObservation,
    RetrievalResponse,
)


class RetrievalError(RuntimeError):
    """Base error raised when a retrieval audit cannot produce valid evidence."""


class RetrievalInputError(RetrievalError, ValueError):
    """The audit request, store snapshot, or retriever identity is invalid."""


class RetrievalObservationError(RetrievalError):
    """A runtime retrieval result cannot be reconciled with the audit inputs."""


class Retriever(Protocol):
    """Retrieve by query and limit without visibility into evaluator targets."""

    retriever_id: str
    retriever_version: str

    def retrieve(self, *, query: str, top_k: int) -> RetrievalResponse:
        """Return minimal target-blind runtime retrieval evidence."""


def retriever_identity(retriever: Retriever) -> tuple[str, str]:
    """Return a retriever's declared identity after validating runtime values."""

    try:
        retriever_id = retriever.retriever_id
        retriever_version = retriever.retriever_version
    except AttributeError as exc:
        raise RetrievalInputError(
            "retriever must declare nonblank retriever_id and retriever_version strings"
        ) from exc
    if not isinstance(retriever_id, str) or not retriever_id.strip():
        raise RetrievalInputError("retriever_id must be a nonblank string")
    if not isinstance(retriever_version, str) or not retriever_version.strip():
        raise RetrievalInputError("retriever_version must be a nonblank string")
    return retriever_id, retriever_version


def validate_retrieval_audit_request(
    request: RetrievalAuditRequest,
    store: NormalizedStore,
) -> None:
    """Require every declared relevance target to exist in the audited store snapshot."""

    if not isinstance(request, RetrievalAuditRequest):
        raise RetrievalInputError("request must be a RetrievalAuditRequest")
    if not isinstance(store, NormalizedStore):
        raise RetrievalInputError("store must be a NormalizedStore")
    missing_ids = tuple(
        memory_id
        for memory_id in request.expected_memory_ids
        if store.get(memory_id) is None
    )
    if missing_ids:
        raise RetrievalInputError(
            f"expected memory IDs are absent from the audited store: {missing_ids!r}"
        )


def _validate_runtime_response(
    *,
    response: object,
    request: RetrievalAuditRequest,
    store: NormalizedStore,
) -> RetrievalResponse:
    if not isinstance(response, RetrievalResponse):
        raise RetrievalObservationError("retriever must return a RetrievalResponse")
    unknown_ids = tuple(
        hit.memory_id for hit in response.hits if store.get(hit.memory_id) is None
    )
    if unknown_ids:
        raise RetrievalObservationError(
            "returned memory IDs are absent from the audited store snapshot: "
            f"{unknown_ids!r}"
        )
    if len(response.hits) > request.top_k:
        raise RetrievalObservationError(
            "retriever returned more hits than requested: "
            f"hits={len(response.hits)}, top_k={request.top_k}"
        )
    return response


def run_retrieval_audit(
    *,
    store: NormalizedStore,
    request: RetrievalAuditRequest,
    retriever: Retriever,
) -> RetrievalObservation:
    """Execute one target-blind retrieval and join it to the declared audit specification."""

    validate_retrieval_audit_request(request, store)
    retriever_id, retriever_version = retriever_identity(retriever)
    try:
        runtime_result = retriever.retrieve(query=request.query, top_k=request.top_k)
    except Exception:
        raise RetrievalObservationError("retriever failed to return runtime evidence") from None
    response = _validate_runtime_response(
        response=runtime_result,
        request=request,
        store=store,
    )
    return RetrievalObservation(
        request_id=request.request_id,
        query_sha256=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
        expected_memory_ids=request.expected_memory_ids,
        top_k=request.top_k,
        retriever_id=retriever_id,
        retriever_version=retriever_version,
        hits=response.hits,
        usage=response.usage,
    )
