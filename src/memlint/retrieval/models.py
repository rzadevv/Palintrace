"""Provider-independent retrieval audit and runtime observation models."""

from __future__ import annotations

import json
import string
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]


def _canonical_memory_ids(
    value: tuple[str, ...],
    *,
    field_name: str,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if require_nonempty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(not memory_id.strip() for memory_id in value):
        raise ValueError(f"{field_name} must contain only nonblank IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must be unique")
    return tuple(sorted(value))


def _canonical_hits(value: tuple[RetrievalHit, ...]) -> tuple[RetrievalHit, ...]:
    ranks = [hit.rank for hit in value]
    if len(set(ranks)) != len(ranks):
        raise ValueError("retrieval hit ranks must be unique")
    memory_ids = [hit.memory_id for hit in value]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("retrieval hit memory IDs must be unique")
    return tuple(sorted(value, key=lambda hit: hit.rank))


class RetrievalAuditRequest(BaseModel):
    """A caller-declared query and visible relevance targets for one retrieval audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    query: str
    expected_memory_ids: tuple[str, ...]
    top_k: StrictPositiveInt

    @field_validator("request_id", "query")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_id and query must not be blank")
        return value

    @field_validator("expected_memory_ids")
    @classmethod
    def expected_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_memory_ids(
            value,
            field_name="expected_memory_ids",
            require_nonempty=True,
        )


class RetrievalHit(BaseModel):
    """One minimal retrieval result, ordered authoritatively by one-based rank."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    rank: StrictPositiveInt
    score: FiniteScore | None = None

    @field_validator("memory_id")
    @classmethod
    def memory_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory_id must not be blank")
        return value

    @field_validator("score", mode="before")
    @classmethod
    def score_must_be_a_python_number(cls, value: object) -> int | float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be a Python int, float, or null")
        return value


class RetrievalUsage(BaseModel):
    """Implementation-reported retrieval work without prices, timing, or token counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_calls: StrictNonNegativeInt
    candidate_count: StrictNonNegativeInt


class RetrievalResponse(BaseModel):
    """A retriever's minimal runtime result without audit targets or defect semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: tuple[RetrievalHit, ...]
    usage: RetrievalUsage

    @field_validator("hits")
    @classmethod
    def hits_are_canonical(cls, value: tuple[RetrievalHit, ...]) -> tuple[RetrievalHit, ...]:
        return _canonical_hits(value)


class RetrievalObservation(BaseModel):
    """Deterministic joined audit specification and target-blind runtime evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    query_sha256: str
    expected_memory_ids: tuple[str, ...]
    top_k: StrictPositiveInt
    retriever_id: str
    retriever_version: str
    hits: tuple[RetrievalHit, ...]
    usage: RetrievalUsage

    @field_validator("request_id", "retriever_id", "retriever_version")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("observation identity strings must not be blank")
        return value

    @field_validator("query_sha256")
    @classmethod
    def query_sha256_must_be_canonical(cls, value: str) -> str:
        if len(value) != 64 or any(character not in string.hexdigits for character in value):
            raise ValueError("query_sha256 must contain exactly 64 hexadecimal characters")
        if value != value.lower():
            raise ValueError("query_sha256 must use lowercase hexadecimal")
        return value

    @field_validator("expected_memory_ids")
    @classmethod
    def expected_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_memory_ids(
            value,
            field_name="expected_memory_ids",
            require_nonempty=True,
        )

    @field_validator("hits")
    @classmethod
    def hits_are_canonical(cls, value: tuple[RetrievalHit, ...]) -> tuple[RetrievalHit, ...]:
        return _canonical_hits(value)

    @model_validator(mode="after")
    def hit_count_must_not_exceed_top_k(self) -> RetrievalObservation:
        if len(self.hits) > self.top_k:
            raise ValueError("observation hits must not exceed top_k")
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically without the audit query text."""

        text = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        if indent is not None:
            text += "\n"
        return text
