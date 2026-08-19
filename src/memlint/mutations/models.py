"""Typed requests, manifests, and results for deterministic mutations."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PositiveInt, field_validator

from memlint.models import NormalizedStore
from memlint.taxonomy import TAXONOMY_VERSION, DefectClass

MUTATION_SCHEMA_VERSION = "1.0"


class BaseStoreStatus(StrEnum):
    """Whether records outside the injected positive have verified clean labels."""

    UNKNOWN = "unknown"
    CURATED_CLEAN = "curated_clean"


class MutationTargetRole(StrEnum):
    """A target's role in the injected defect or its supporting context."""

    PRIMARY = "primary"
    CONFLICTING = "conflicting"
    SUPERSEDING = "superseding"
    SOURCE = "source"
    DUPLICATE = "duplicate"


class MutationTarget(BaseModel):
    """One memory involved in a mutation, described only in the gold manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    role: MutationTargetRole
    receives_gold_label: bool


class RetrievalProbe(BaseModel):
    """A later runtime retrieval experiment, without an observed result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    expected_memory_ids: tuple[str, ...]
    distractor_memory_ids: tuple[str, ...]

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class MutationRequest(BaseModel):
    """Explicit mutation configuration; no semantic values are guessed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    defect_class: DefectClass
    subtype: str | None = None
    seed: int = 0
    target_memory_id: str | None = None
    replace_from: str | None = None
    replace_to: str | None = None
    destination_user_id: str | None = None
    destination_agent_id: str | None = None
    query: str | None = None
    distractor_count: PositiveInt = Field(default=3, le=5)
    base_store_status: BaseStoreStatus = BaseStoreStatus.UNKNOWN

    @field_validator(
        "subtype",
        "target_memory_id",
        "replace_from",
        "replace_to",
        "destination_user_id",
        "destination_agent_id",
        "query",
    )
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("mutation request strings must not be blank")
        return value


class MutationManifest(BaseModel):
    """Gold mutation data kept separate from the detector-visible store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = MUTATION_SCHEMA_VERSION
    taxonomy_version: str = TAXONOMY_VERSION
    mutation_id: str
    defect_class: DefectClass
    subtype: str
    seed: int
    base_store_digest: str
    mutated_store_digest: str
    transcript_digest: str | None = None
    target_memory_ids: tuple[str, ...]
    targets: tuple[MutationTarget, ...]
    created_memory_ids: tuple[str, ...] = ()
    modified_memory_ids: tuple[str, ...] = ()
    removed_memory_ids: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    requires_runtime_validation: bool = False
    base_store_status: BaseStoreStatus = BaseStoreStatus.UNKNOWN
    retrieval_probe: RetrievalProbe | None = None

    def to_json(self, output: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Serialize the manifest deterministically, without execution-time fields."""

        text = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        if indent is not None:
            text += "\n"
        if output is not None:
            Path(output).write_text(text, encoding="utf-8")
        return text


class MutationResult(BaseModel):
    """A new normalized store paired with its separate gold manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mutated_store: NormalizedStore
    manifest: MutationManifest
