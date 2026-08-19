"""Typed requests, manifests, and results for deterministic mutations."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    StrictBool,
    field_validator,
    model_validator,
)

from memlint.models import NormalizedStore
from memlint.taxonomy import TAXONOMY_VERSION, DefectClass

MUTATION_SCHEMA_VERSION = "1.1"


class BaseStoreStatus(StrEnum):
    """Whether records outside the injected positive have verified clean labels."""

    UNKNOWN = "unknown"
    CURATED_CLEAN = "curated_clean"


class ConflictRelation(StrEnum):
    """Explicit semantic contract for a controlled contradiction fixture."""

    EXCLUSIVE_VALUE = "exclusive_value"


class DistractorFamily(StrEnum):
    """Fixed topic family used by a controlled retrieval challenge."""

    EDITOR = "editor"


class GoldLabelUnit(StrEnum):
    """The evaluation object that receives one gold label."""

    MEMORY = "memory"
    MEMORY_PAIR = "memory_pair"
    RETRIEVAL_CASE = "retrieval_case"


GOLD_LABEL_UNIT_BY_DEFECT = {
    DefectClass.UNSUPPORTED_CLAIM: GoldLabelUnit.MEMORY,
    DefectClass.INTERNAL_CONTRADICTION: GoldLabelUnit.MEMORY_PAIR,
    DefectClass.STALE_ACTIVE: GoldLabelUnit.MEMORY,
    DefectClass.ORPHANED_PROVENANCE: GoldLabelUnit.MEMORY,
    DefectClass.RETRIEVAL_SHADOWING: GoldLabelUnit.RETRIEVAL_CASE,
    DefectClass.INJECTED_INSTRUCTION: GoldLabelUnit.MEMORY,
    DefectClass.PRIVACY_SCOPE_VIOLATION: GoldLabelUnit.MEMORY,
    DefectClass.REDUNDANCY_BLOAT: GoldLabelUnit.MEMORY_PAIR,
}


class MutationTargetRole(StrEnum):
    """A target's role in the injected defect or its supporting context."""

    PRIMARY = "primary"
    CONFLICTING = "conflicting"
    SUPERSEDING = "superseding"
    SOURCE = "source"
    DUPLICATE = "duplicate"


class MutationTarget(BaseModel):
    """One contextual memory role, described only in the gold manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    role: MutationTargetRole

    @field_validator("memory_id")
    @classmethod
    def memory_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("target memory_id must not be blank")
        return value


class GoldLabel(BaseModel):
    """One evaluation unit, including relational IDs kept together."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: GoldLabelUnit
    memory_ids: tuple[str, ...]
    observed_positive: StrictBool

    @field_validator("memory_ids")
    @classmethod
    def memory_ids_must_be_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("gold label memory_ids must not be empty")
        if any(not memory_id.strip() for memory_id in value):
            raise ValueError("gold label memory IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("gold label memory IDs must be unique")
        return value

    @model_validator(mode="after")
    def unit_has_correct_arity(self) -> GoldLabel:
        expected = 2 if self.unit is GoldLabelUnit.MEMORY_PAIR else 1
        if len(self.memory_ids) != expected:
            raise ValueError(f"{self.unit.value} requires exactly {expected} memory ID(s)")
        return self


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

    @field_validator("expected_memory_ids")
    @classmethod
    def expected_ids_must_be_nonempty_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError("expected_memory_ids must not be empty")
        if any(not memory_id.strip() for memory_id in value):
            raise ValueError("expected memory IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("expected memory IDs must be unique")
        return value

    @field_validator("distractor_memory_ids")
    @classmethod
    def distractor_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not memory_id.strip() for memory_id in value):
            raise ValueError("distractor memory IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("distractor memory IDs must be unique")
        return value

    @model_validator(mode="after")
    def expected_and_distractor_ids_must_be_disjoint(self) -> RetrievalProbe:
        if set(self.expected_memory_ids) & set(self.distractor_memory_ids):
            raise ValueError("expected and distractor memory IDs must be disjoint")
        return self


class MutationRequest(BaseModel):
    """Explicit mutation configuration; no semantic values are guessed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    defect_class: DefectClass
    subtype: str | None = None
    seed: int = 0
    target_memory_id: str | None = None
    replace_from: str | None = None
    replace_to: str | None = None
    conflict_relation: ConflictRelation | None = None
    destination_user_id: str | None = None
    destination_agent_id: str | None = None
    query: str | None = None
    distractor_family: DistractorFamily | None = None
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
    gold_label: GoldLabel
    created_memory_ids: tuple[str, ...] = ()
    modified_memory_ids: tuple[str, ...] = ()
    removed_memory_ids: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    requires_runtime_validation: bool = False
    base_store_status: BaseStoreStatus = BaseStoreStatus.UNKNOWN
    retrieval_probe: RetrievalProbe | None = None

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != MUTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported mutation schema_version: {value!r}")
        return value

    @field_validator("taxonomy_version")
    @classmethod
    def taxonomy_version_must_match(cls, value: str) -> str:
        if value != TAXONOMY_VERSION:
            raise ValueError(f"unsupported taxonomy_version: {value!r}")
        return value

    @field_validator("mutation_id", "subtype", "base_store_digest", "mutated_store_digest")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("manifest identifiers and digests must not be blank")
        return value

    @field_validator("transcript_digest")
    @classmethod
    def transcript_digest_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("transcript_digest must not be blank")
        return value

    @field_validator("target_memory_ids")
    @classmethod
    def target_ids_must_be_nonempty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("target_memory_ids must not be empty")
        return cls._validate_id_collection(value)

    @field_validator("created_memory_ids", "modified_memory_ids", "removed_memory_ids")
    @classmethod
    def changed_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return cls._validate_id_collection(value)

    @field_validator("targets")
    @classmethod
    def targets_must_be_nonempty_unique(
        cls, value: tuple[MutationTarget, ...]
    ) -> tuple[MutationTarget, ...]:
        if not value:
            raise ValueError("targets must not be empty")
        ids = tuple(target.memory_id for target in value)
        if len(set(ids)) != len(ids):
            raise ValueError("target structures must have unique memory IDs")
        return value

    @model_validator(mode="after")
    def validate_research_invariants(self) -> MutationManifest:
        changed = {
            "created": set(self.created_memory_ids),
            "modified": set(self.modified_memory_ids),
            "removed": set(self.removed_memory_ids),
        }
        pairs = (("created", "modified"), ("created", "removed"), ("modified", "removed"))
        for left, right in pairs:
            if changed[left] & changed[right]:
                raise ValueError(f"{left} and {right} memory IDs must be disjoint")

        declared_target_ids = {target.memory_id for target in self.targets}
        if not set(self.target_memory_ids) <= declared_target_ids:
            raise ValueError("target_memory_ids must correspond to target structures")
        if self.gold_label.memory_ids != self.target_memory_ids:
            raise ValueError("gold label memory_ids must equal target_memory_ids")

        expected_unit = GOLD_LABEL_UNIT_BY_DEFECT[self.defect_class]
        if self.gold_label.unit is not expected_unit:
            raise ValueError(
                f"{self.defect_class.value} requires gold label unit {expected_unit.value}"
            )
        has_probe = self.retrieval_probe is not None
        if self.requires_runtime_validation != has_probe:
            raise ValueError("runtime validation and retrieval_probe must be present together")

        if self.defect_class is DefectClass.RETRIEVAL_SHADOWING:
            if self.gold_label.observed_positive:
                raise ValueError("retrieval challenge cannot contain an observed positive")
            if self.retrieval_probe is None:
                raise ValueError("retrieval challenge requires retrieval_probe")
            if self.retrieval_probe.expected_memory_ids != self.target_memory_ids:
                raise ValueError("retrieval expected IDs must equal target_memory_ids")
            if self.retrieval_probe.distractor_memory_ids != self.created_memory_ids:
                raise ValueError("retrieval distractor IDs must equal created_memory_ids")
        elif not self.gold_label.observed_positive:
            raise ValueError("static controlled mutations require an observed positive")
        return self

    @staticmethod
    def _validate_id_collection(value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not memory_id.strip() for memory_id in value):
            raise ValueError("memory IDs must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("memory ID collections must not contain duplicates")
        return value

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
