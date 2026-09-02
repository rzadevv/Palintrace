"""Versioned pre-execution benchmark specification contracts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from palintrace.evaluation.models import EvaluationInputError
from palintrace.models import NormalizedMemory, NormalizedStore, TranscriptSet
from palintrace.mutations import BaseStoreStatus, DistractorFamily, MutationRequest
from palintrace.retrieval import RetrievalSufficiencyPolicy
from palintrace.serialization import load_store, load_transcripts
from palintrace.taxonomy import DefectClass

BENCHMARK_SCHEMA_VERSION = "0.1"
BENCHMARK_ID = "palintrace-controlled-v0.1"
BENCHMARK_SPEC_SHA256 = "90793984a6e191cad4cd276c185dfe8a1eac814035493a101768a96051a240f1"

StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]

STATIC_BENCHMARK_DEFECTS = (
    DefectClass.ORPHANED_PROVENANCE,
    DefectClass.REDUNDANCY_BLOAT,
    DefectClass.STALE_ACTIVE,
    DefectClass.PRIVACY_SCOPE_VIOLATION,
    DefectClass.UNSUPPORTED_CLAIM,
)
STATIC_HELD_OUT_COUNTS = {
    DefectClass.ORPHANED_PROVENANCE: 9,
    DefectClass.REDUNDANCY_BLOAT: 6,
    DefectClass.STALE_ACTIVE: 6,
    DefectClass.PRIVACY_SCOPE_VIOLATION: 6,
    DefectClass.UNSUPPORTED_CLAIM: 12,
}
REQUIRED_DEVELOPMENT_ARTIFACTS = (
    "examples/mutation-store.json",
    "examples/mutation-transcripts.json",
    "tests/fixtures/contradiction_pair_probe_v0.1.json",
    "tests/fixtures/evidence_composition_probe_v0.1.json",
    "tests/fixtures/injected_instruction_probe_v0.1.json",
    "tests/fixtures/instruction_contradiction_probe_v0.1.json",
    "tests/fixtures/semantic_probe_v0.1.json",
)


class BenchmarkSplit(StrEnum):
    """Exact pre-execution development/held-out partition."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class BenchmarkCaseKind(StrEnum):
    """Semantically distinct benchmark specification unit types."""

    STATIC_MUTATION = "static_mutation"
    CLEAN_CONTROL = "clean_control"
    RETRIEVAL_CHALLENGE = "retrieval_challenge"


def _nonblank(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _repository_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError("benchmark fixture paths must be canonical repository-relative paths")
    return _nonblank(value, field_name="benchmark fixture path")


class _BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkFixture(_BenchmarkModel):
    """Public paths and declared cleanliness for one synthetic fixture bundle."""

    fixture_id: str
    store_path: str
    transcripts_path: str | None = None
    base_store_status: BaseStoreStatus
    scope_policy_path: str | None = None

    @field_validator("fixture_id")
    @classmethod
    def fixture_id_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="fixture_id")

    @field_validator("store_path")
    @classmethod
    def store_path_must_be_relative(cls, value: str) -> str:
        return _repository_relative_path(value)

    @field_validator("transcripts_path", "scope_policy_path")
    @classmethod
    def optional_paths_must_be_relative(cls, value: str | None) -> str | None:
        return None if value is None else _repository_relative_path(value)


class StaticMutationBenchmarkCase(_BenchmarkModel):
    """One frozen static challenge-generation request without detector output."""

    case_id: str
    kind: BenchmarkCaseKind
    split: BenchmarkSplit
    defect_class: DefectClass
    subtype: str
    base_fixture_id: str
    transcript_fixture_id: str | None = None
    mutation_request: MutationRequest
    semantic_domain: str | None = None

    @field_validator("case_id", "subtype", "base_fixture_id")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="static benchmark identifiers")

    @field_validator("transcript_fixture_id", "semantic_domain")
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, field_name="static case metadata")

    @model_validator(mode="after")
    def request_matches_declared_static_case(self) -> StaticMutationBenchmarkCase:
        if self.kind is not BenchmarkCaseKind.STATIC_MUTATION:
            raise ValueError("static mutation case requires kind='static_mutation'")
        if self.defect_class not in STATIC_BENCHMARK_DEFECTS:
            raise ValueError("static benchmark cases require an implemented detector class")
        if self.mutation_request.defect_class is not self.defect_class:
            raise ValueError("mutation request defect class must match its benchmark case")
        if self.mutation_request.subtype != self.subtype:
            raise ValueError("mutation request subtype must match its benchmark case")
        if (
            self.split is BenchmarkSplit.HELD_OUT
            and self.mutation_request.base_store_status is not BaseStoreStatus.CURATED_CLEAN
        ):
            raise ValueError("held-out mutation requests require curated-clean base status")
        if (
            self.defect_class is DefectClass.UNSUPPORTED_CLAIM
            and self.semantic_domain is None
        ):
            raise ValueError("unsupported-claim cases require a semantic_domain")
        return self


class CleanControlBenchmarkCase(_BenchmarkModel):
    """One unmutated curated-clean control specification."""

    case_id: str
    kind: BenchmarkCaseKind
    split: BenchmarkSplit
    defect_class: DefectClass
    base_fixture_id: str
    transcript_fixture_id: str | None = None
    scope_policy_fixture_id: str | None = None

    @field_validator("case_id", "base_fixture_id")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="clean-control identifiers")

    @field_validator("transcript_fixture_id", "scope_policy_fixture_id")
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, field_name="clean-control fixture ID")

    @model_validator(mode="after")
    def kind_and_defect_must_be_supported(self) -> CleanControlBenchmarkCase:
        if self.kind is not BenchmarkCaseKind.CLEAN_CONTROL:
            raise ValueError("clean-control case requires kind='clean_control'")
        if self.defect_class not in STATIC_BENCHMARK_DEFECTS:
            raise ValueError("clean controls require an implemented static detector class")
        if self.defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION:
            if self.scope_policy_fixture_id is None:
                raise ValueError("privacy clean controls require an explicit scope policy fixture")
        elif self.scope_policy_fixture_id is not None:
            raise ValueError("scope policy fixtures are only valid for privacy clean controls")
        return self


class RetrievalCondition(_BenchmarkModel):
    """One intended experimental retrieval condition without runtime output."""

    condition_id: str
    policy: RetrievalSufficiencyPolicy
    top_k: StrictPositiveInt
    retriever_kind: str
    retriever_config_version: str

    @field_validator("condition_id", "retriever_kind", "retriever_config_version")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="retrieval condition identity")


class RetrievalBenchmarkCase(_BenchmarkModel):
    """One paired retrieval challenge-generation specification without observations."""

    case_id: str
    kind: BenchmarkCaseKind
    split: BenchmarkSplit
    base_fixture_id: str
    mutation_request: MutationRequest
    policy: RetrievalSufficiencyPolicy
    top_k: StrictPositiveInt
    retrieval_condition_id: str

    @field_validator("case_id", "base_fixture_id", "retrieval_condition_id")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="retrieval benchmark identifiers")

    @model_validator(mode="after")
    def request_is_a_frozen_retrieval_challenge(self) -> RetrievalBenchmarkCase:
        request = self.mutation_request
        if self.kind is not BenchmarkCaseKind.RETRIEVAL_CHALLENGE:
            raise ValueError("retrieval case requires kind='retrieval_challenge'")
        if request.defect_class is not DefectClass.RETRIEVAL_SHADOWING:
            raise ValueError("retrieval case requires retrieval_shadowing mutation request")
        if request.subtype != "distractor_crowding":
            raise ValueError("retrieval case requires subtype='distractor_crowding'")
        if request.distractor_family is not DistractorFamily.EDITOR:
            raise ValueError("retrieval case requires the frozen editor distractor family")
        if request.target_memory_id is None or request.query is None:
            raise ValueError("retrieval cases require explicit target and query")
        if (
            self.split is BenchmarkSplit.HELD_OUT
            and request.base_store_status is not BaseStoreStatus.CURATED_CLEAN
        ):
            raise ValueError("held-out retrieval requests require curated-clean base status")
        return self


class BenchmarkCheckerIdentity(_BenchmarkModel):
    """Intended frozen checker identity, without behavior or output."""

    defect_class: DefectClass
    checker_id: str
    checker_version: str

    @field_validator("checker_id", "checker_version")
    @classmethod
    def identity_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="checker identity")


class UnsupportedClaimMethodSpec(_BenchmarkModel):
    """Execution provenance for the already-frozen unsupported-claim method."""

    checker_id: str
    semantic_model: str
    semantic_revision: str
    composition: str

    @field_validator("checker_id", "semantic_model", "semantic_revision", "composition")
    @classmethod
    def provenance_strings_must_not_be_blank(cls, value: str) -> str:
        return _nonblank(value, field_name="unsupported-claim method provenance")


class BenchmarkSpec(_BenchmarkModel):
    """Complete immutable v0.1 challenge specification before detector execution."""

    schema_version: str
    benchmark_id: str
    fixtures: tuple[BenchmarkFixture, ...]
    static_mutation_cases: tuple[StaticMutationBenchmarkCase, ...]
    clean_control_cases: tuple[CleanControlBenchmarkCase, ...]
    retrieval_conditions: tuple[RetrievalCondition, ...]
    retrieval_cases: tuple[RetrievalBenchmarkCase, ...]
    development_artifacts: tuple[str, ...]
    checker_identities: tuple[BenchmarkCheckerIdentity, ...]
    unsupported_claim_method: UnsupportedClaimMethodSpec

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_frozen(cls, value: str) -> str:
        if value != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"unsupported benchmark schema_version: {value!r}")
        return value

    @field_validator("benchmark_id")
    @classmethod
    def benchmark_id_is_frozen(cls, value: str) -> str:
        if value != BENCHMARK_ID:
            raise ValueError(f"unsupported benchmark_id: {value!r}")
        return value

    @field_validator("fixtures")
    @classmethod
    def fixtures_are_unique_and_canonical(
        cls,
        value: tuple[BenchmarkFixture, ...],
    ) -> tuple[BenchmarkFixture, ...]:
        ids = [fixture.fixture_id for fixture in value]
        if len(set(ids)) != len(ids):
            raise ValueError("benchmark fixture IDs must be unique")
        return tuple(sorted(value, key=lambda fixture: fixture.fixture_id))

    @field_validator("static_mutation_cases")
    @classmethod
    def static_cases_are_canonical(
        cls,
        value: tuple[StaticMutationBenchmarkCase, ...],
    ) -> tuple[StaticMutationBenchmarkCase, ...]:
        return tuple(sorted(value, key=lambda case: case.case_id))

    @field_validator("clean_control_cases")
    @classmethod
    def clean_cases_are_canonical(
        cls,
        value: tuple[CleanControlBenchmarkCase, ...],
    ) -> tuple[CleanControlBenchmarkCase, ...]:
        return tuple(sorted(value, key=lambda case: case.case_id))

    @field_validator("retrieval_cases")
    @classmethod
    def retrieval_cases_are_canonical(
        cls,
        value: tuple[RetrievalBenchmarkCase, ...],
    ) -> tuple[RetrievalBenchmarkCase, ...]:
        return tuple(sorted(value, key=lambda case: case.case_id))

    @field_validator("retrieval_conditions")
    @classmethod
    def conditions_are_unique_and_canonical(
        cls,
        value: tuple[RetrievalCondition, ...],
    ) -> tuple[RetrievalCondition, ...]:
        ids = [condition.condition_id for condition in value]
        if len(set(ids)) != len(ids):
            raise ValueError("retrieval condition IDs must be unique")
        return tuple(sorted(value, key=lambda condition: condition.condition_id))

    @field_validator("development_artifacts")
    @classmethod
    def development_artifacts_are_frozen(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(set(value)) != len(value):
            raise ValueError("development artifacts must be nonblank and unique")
        canonical = tuple(sorted(value))
        if canonical != tuple(sorted(REQUIRED_DEVELOPMENT_ARTIFACTS)):
            raise ValueError("development artifact registry must match benchmark v0.1")
        return canonical

    @field_validator("checker_identities")
    @classmethod
    def checker_identities_are_canonical(
        cls,
        value: tuple[BenchmarkCheckerIdentity, ...],
    ) -> tuple[BenchmarkCheckerIdentity, ...]:
        defects = [identity.defect_class for identity in value]
        if len(set(defects)) != len(defects):
            raise ValueError("checker identities must have unique defect classes")
        return tuple(sorted(value, key=lambda identity: identity.defect_class.value))

    @model_validator(mode="after")
    def benchmark_v01_matrix_is_frozen(self) -> BenchmarkSpec:
        fixtures = {fixture.fixture_id: fixture for fixture in self.fixtures}
        if len(fixtures) != 3:
            raise ValueError("benchmark v0.1 requires exactly three held-out fixtures")
        if any(
            fixture.base_store_status is not BaseStoreStatus.CURATED_CLEAN
            for fixture in self.fixtures
        ):
            raise ValueError("every held-out fixture must be explicitly curated clean")

        all_cases: tuple[
            StaticMutationBenchmarkCase
            | CleanControlBenchmarkCase
            | RetrievalBenchmarkCase,
            ...,
        ] = (
            *self.static_mutation_cases,
            *self.clean_control_cases,
            *self.retrieval_cases,
        )
        case_ids = [case.case_id for case in all_cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("benchmark case IDs must be globally unique")
        if any(case.split is not BenchmarkSplit.HELD_OUT for case in all_cases):
            raise ValueError("benchmark v0.1 case matrices contain held-out cases only")

        for case in all_cases:
            if case.base_fixture_id not in fixtures:
                raise ValueError(f"unknown base fixture ID: {case.base_fixture_id}")
            fixture = fixtures[case.base_fixture_id]
            transcript_fixture_id = getattr(case, "transcript_fixture_id", None)
            if transcript_fixture_id is not None:
                if transcript_fixture_id != case.base_fixture_id:
                    raise ValueError("transcript fixture must match the base fixture bundle")
                if fixture.transcripts_path is None:
                    raise ValueError("referenced fixture has no transcript path")
            scope_fixture_id = getattr(case, "scope_policy_fixture_id", None)
            if scope_fixture_id is not None:
                if scope_fixture_id != case.base_fixture_id:
                    raise ValueError("scope-policy fixture must match the base fixture bundle")
                if fixture.scope_policy_path is None:
                    raise ValueError("referenced fixture has no scope-policy path")

        static_counts = Counter(
            case.defect_class for case in self.static_mutation_cases
        )
        if static_counts != Counter(STATIC_HELD_OUT_COUNTS):
            raise ValueError("held-out static mutation counts do not match benchmark v0.1")
        if len(self.static_mutation_cases) != 39:
            raise ValueError("benchmark v0.1 requires exactly 39 static mutation cases")

        orphan_subtypes = Counter(
            case.subtype
            for case in self.static_mutation_cases
            if case.defect_class is DefectClass.ORPHANED_PROVENANCE
        )
        if orphan_subtypes != Counter(
            {"missing_transcript": 3, "missing_turn": 3, "invalid_span": 3}
        ):
            raise ValueError("orphaned-provenance subtype matrix must be balanced 3/3/3")

        redundancy_targets: dict[str, set[str | None]] = defaultdict(set)
        for case in self.static_mutation_cases:
            if case.defect_class is DefectClass.REDUNDANCY_BLOAT:
                redundancy_targets[case.base_fixture_id].add(
                    case.mutation_request.target_memory_id
                )
        if set(redundancy_targets) != set(fixtures) or any(
            len(targets) < 2 for targets in redundancy_targets.values()
        ):
            raise ValueError("redundancy matrix requires two targets from every fixture")

        stale_domains = {
            case.semantic_domain
            for case in self.static_mutation_cases
            if case.defect_class is DefectClass.STALE_ACTIVE
        }
        required_stale_domains = {
            "preference",
            "employment_project",
            "location_device_subscription",
        }
        if not required_stale_domains <= stale_domains:
            raise ValueError("stale-active matrix does not cover the frozen domain groups")

        privacy_subtypes = Counter(
            case.subtype
            for case in self.static_mutation_cases
            if case.defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION
        )
        if privacy_subtypes != Counter({"cross_user_copy": 3, "cross_agent_copy": 3}):
            raise ValueError("privacy matrix must be split 3 cross-user / 3 cross-agent")

        unsupported_domains = {
            case.semantic_domain
            for case in self.static_mutation_cases
            if case.defect_class is DefectClass.UNSUPPORTED_CLAIM
        }
        if None in unsupported_domains or len(unsupported_domains) < 4:
            raise ValueError("unsupported-claim matrix requires at least four semantic domains")

        clean_counts = Counter(case.defect_class for case in self.clean_control_cases)
        if clean_counts != Counter({defect: 3 for defect in STATIC_BENCHMARK_DEFECTS}):
            raise ValueError("clean-control matrix requires three cases per static defect class")
        if len(self.clean_control_cases) != 15:
            raise ValueError("benchmark v0.1 requires exactly 15 clean controls")
        for defect in STATIC_BENCHMARK_DEFECTS:
            defect_fixtures = {
                case.base_fixture_id
                for case in self.clean_control_cases
                if case.defect_class is defect
            }
            if defect_fixtures != set(fixtures):
                raise ValueError("each clean-control defect must cover all three fixtures")

        expected_identity = {
            defect: (defect.value, "1.0") for defect in STATIC_BENCHMARK_DEFECTS
        }
        actual_identity = {
            item.defect_class: (item.checker_id, item.checker_version)
            for item in self.checker_identities
        }
        if actual_identity != expected_identity:
            raise ValueError("checker identity registry does not match frozen implementations")
        method = self.unsupported_claim_method
        if (
            method.checker_id != "unsupported_claim"
            or method.semantic_model != "cross-encoder/nli-MiniLM2-L6-H768"
            or method.semantic_revision
            != "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
            or method.composition != "plain"
        ):
            raise ValueError("unsupported-claim method provenance does not match the freeze")

        if len(self.retrieval_conditions) != 1:
            raise ValueError("benchmark v0.1 requires exactly one retrieval condition")
        condition = self.retrieval_conditions[0]
        if (
            condition.condition_id != "lexical-baseline-k3"
            or condition.policy is not RetrievalSufficiencyPolicy.ALL_EXPECTED
            or condition.top_k != 3
            or condition.retriever_kind != "experimental_lexical"
            or condition.retriever_config_version != "0.1"
        ):
            raise ValueError("retrieval condition does not match lexical-baseline-k3")
        if len(self.retrieval_cases) != 12:
            raise ValueError("ready benchmark v0.1 requires exactly 12 retrieval cases")
        retrieval_counts = Counter(case.base_fixture_id for case in self.retrieval_cases)
        if retrieval_counts != Counter({fixture_id: 4 for fixture_id in fixtures}):
            raise ValueError("retrieval matrix requires four cases per held-out fixture")
        retrieval_targets = [
            case.mutation_request.target_memory_id for case in self.retrieval_cases
        ]
        retrieval_queries = [case.mutation_request.query for case in self.retrieval_cases]
        if len(set(retrieval_targets)) != 12 or len(set(retrieval_queries)) != 12:
            raise ValueError("retrieval cases require distinct targets and queries")
        for case in self.retrieval_cases:
            if (
                case.retrieval_condition_id != condition.condition_id
                or case.policy is not condition.policy
                or case.top_k != condition.top_k
            ):
                raise ValueError("retrieval case does not match its frozen condition")
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the pre-execution specification deterministically."""

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


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    """Load and validate one benchmark specification without executing a detector."""

    benchmark_path = Path(path)
    try:
        text = benchmark_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationInputError(f"could not read benchmark specification: {error}") from error
    try:
        return BenchmarkSpec.model_validate_json(text)
    except ValidationError as error:
        raise EvaluationInputError(f"invalid benchmark specification: {error}") from error


def _resolved_source_segments(
    memory: NormalizedMemory,
    transcripts: TranscriptSet,
) -> tuple[str, ...]:
    segments: list[str] = []
    for source_ref in memory.source_refs:
        transcript = transcripts.get(source_ref.transcript_id)
        if transcript is None:
            raise EvaluationInputError(
                f"fixture memory {memory.id} references a missing transcript"
            )
        if source_ref.turn_idx is None:
            if source_ref.span is not None:  # pragma: no cover - normalized model excludes this
                raise EvaluationInputError("source span requires a turn index")
            segments.extend(turn.content for turn in transcript.turns)
            continue
        turn = next(
            (item for item in transcript.turns if item.index == source_ref.turn_idx),
            None,
        )
        if turn is None:
            raise EvaluationInputError(
                f"fixture memory {memory.id} references a missing transcript turn"
            )
        if source_ref.span is None:
            segments.append(turn.content)
        else:
            start, end = source_ref.span
            if end > len(turn.content):
                raise EvaluationInputError(
                    f"fixture memory {memory.id} has an invalid source span"
                )
            segments.append(turn.content[start:end])
    return tuple(segments)


def validate_benchmark_fixture_eligibility(
    spec: BenchmarkSpec,
    *,
    repository_root: str | Path,
) -> None:
    """Validate fixture references and challenge preconditions without mutation or detection."""

    root = Path(repository_root)
    stores: dict[str, NormalizedStore] = {}
    transcripts_by_fixture: dict[str, TranscriptSet] = {}
    for fixture in spec.fixtures:
        store = load_store(root / fixture.store_path)
        if fixture.transcripts_path is None:
            raise EvaluationInputError("held-out fixtures require transcript paths")
        transcripts = load_transcripts(root / fixture.transcripts_path)
        stores[fixture.fixture_id] = store
        transcripts_by_fixture[fixture.fixture_id] = transcripts
        if len(store.memories) < 5:
            raise EvaluationInputError("held-out fixture must contain at least five memories")
        if any(
            memory.scope.user_id is None
            or memory.scope.agent_id is None
            or memory.scope.session_id is None
            for memory in store.memories
        ):
            raise EvaluationInputError("held-out fixture memories require complete known scope")
        declared = [memory for memory in store.memories if memory.source_refs]
        if len(declared) < 3:
            raise EvaluationInputError("held-out fixture requires three declared memories")
        if len({memory.content for memory in store.memories}) != len(store.memories):
            raise EvaluationInputError("held-out fixture contains exact duplicate content")
        if any(memory.supersedes for memory in store.memories):
            raise EvaluationInputError("held-out fixture contains an explicit supersession")
        for memory in declared:
            _resolved_source_segments(memory, transcripts)

    for case in spec.static_mutation_cases:
        store = stores[case.base_fixture_id]
        target_id = case.mutation_request.target_memory_id
        target = None if target_id is None else store.get(target_id)
        if target is None:
            raise EvaluationInputError(f"static case target does not exist: {case.case_id}")
        if case.defect_class is DefectClass.REDUNDANCY_BLOAT and all(
            value is None
            for value in (
                target.scope.user_id,
                target.scope.agent_id,
                target.scope.session_id,
            )
        ):
            raise EvaluationInputError(
                f"redundancy case target has no observable scope: {case.case_id}"
            )
        if case.defect_class is DefectClass.UNSUPPORTED_CLAIM:
            transcripts = transcripts_by_fixture[case.base_fixture_id]
            segments = _resolved_source_segments(target, transcripts)
            replace_from = case.mutation_request.replace_from
            replace_to = case.mutation_request.replace_to
            if replace_from is None or replace_to is None:
                raise EvaluationInputError("unsupported case requires explicit substitution")
            if target.content.count(replace_from) != 1:
                raise EvaluationInputError(
                    f"unsupported case replace_from is not unique: {case.case_id}"
                )
            if not any(replace_from in segment for segment in segments):
                raise EvaluationInputError(
                    f"unsupported source does not contain replace_from: {case.case_id}"
                )
            if any(replace_to in segment for segment in segments):
                raise EvaluationInputError(
                    f"unsupported replacement appears in declared evidence: {case.case_id}"
                )

    for retrieval_case in spec.retrieval_cases:
        store = stores[retrieval_case.base_fixture_id]
        target_id = retrieval_case.mutation_request.target_memory_id
        if target_id is None or store.get(target_id) is None:
            raise EvaluationInputError(
                f"retrieval target does not exist: {retrieval_case.case_id}"
            )
