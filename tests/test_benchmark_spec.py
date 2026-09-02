from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import palintrace.cli as cli
from palintrace.checkers import (
    OrphanedProvenanceChecker,
    PrivacyScopeViolationChecker,
    RedundancyBloatChecker,
    StaleActiveChecker,
    load_scope_policy,
)
from palintrace.evaluation import (
    BENCHMARK_ID,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_SPEC_SHA256,
    BenchmarkCaseKind,
    BenchmarkSpec,
    BenchmarkSplit,
    EvaluationInputError,
    load_benchmark_spec,
    validate_benchmark_fixture_eligibility,
)
from palintrace.models import NormalizedMemory, NormalizedStore, TranscriptSet
from palintrace.mutations import BaseStoreStatus, GoldLabelUnit, MutationRequest, mutate
from palintrace.mutations.base import semantic_store_digest
from palintrace.retrieval import RetrievalSufficiencyPolicy
from palintrace.serialization import load_store, load_transcripts
from palintrace.taxonomy import DefectClass

REPOSITORY_ROOT = Path(".")
FIXTURE_ROOT = Path("tests/fixtures/benchmark_v0.1")
BENCHMARK_PATH = FIXTURE_ROOT / "benchmark.json"
HASH_MANIFEST_PATH = Path("tests/fixtures/benchmark_v0.1.sha256.json")
BENCHMARK_MODULE_PATH = Path("src/palintrace/evaluation/benchmark.py")

EXPECTED_SOURCE_VALUES = {
    "h1-journal-app": "Obsidian",
    "h1-writing-tool": "FocusWriter",
    "h1-recipe-editor": "MarkText",
    "h1-research-notes": "Logseq",
    "h1-breakfast-time": "7:15 a.m.",
    "h1-yoga-day": "Tuesdays",
    "h1-tea-count": "two tins of oolong",
    "h1-library-branch": "Riverside branch",
    "h2-code-editor": "Clojure services in Kate",
    "h2-android-ide": "Android prototypes in Android Studio",
    "h2-terminal-editor": "server configuration files with micro",
    "h2-diff-tool": "local diffs in Meld",
    "h2-employer": "Cedar Harbor Analytics",
    "h2-home-city": "Porto's Cedofeita district",
    "h2-work-laptop": "ThinkPad T14",
    "h2-office-day": "Wednesdays",
    "h3-documentation-editor": "project documentation in Typora",
    "h3-project-ide": "Kestrel mobile app in Eclipse",
    "h3-config-editor": "deployment manifests in Kakoune",
    "h3-release-editor": "release notes in HedgeDoc",
    "h3-project-name": "Kestrel migration project",
    "h3-subscription-cycle": "renews monthly",
    "h3-status-channel": "status updates by email",
    "h3-support-count": "three support queues",
}


@pytest.fixture(scope="module")
def benchmark_spec() -> BenchmarkSpec:
    return load_benchmark_spec(BENCHMARK_PATH)


def _fixture_inputs(
    benchmark_spec: BenchmarkSpec,
) -> dict[str, tuple[NormalizedStore, TranscriptSet]]:
    inputs: dict[str, tuple[NormalizedStore, TranscriptSet]] = {}
    for fixture in benchmark_spec.fixtures:
        assert fixture.transcripts_path is not None
        inputs[fixture.fixture_id] = (
            load_store(fixture.store_path),
            load_transcripts(fixture.transcripts_path),
        )
    return inputs


def _source_text(memory: NormalizedMemory, transcripts: TranscriptSet) -> str:
    assert len(memory.source_refs) == 1
    source_ref = memory.source_refs[0]
    transcript = transcripts.get(source_ref.transcript_id)
    assert transcript is not None
    assert source_ref.turn_idx is not None
    turn = next(item for item in transcript.turns if item.index == source_ref.turn_idx)
    if source_ref.span is None:
        return turn.content
    return turn.content[source_ref.span[0] : source_ref.span[1]]


def _all_json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested_key for item in value.values() for nested_key in _all_json_keys(item)
        }
    if isinstance(value, list):
        return {nested_key for item in value for nested_key in _all_json_keys(item)}
    return set()


def _all_json_strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {item for nested in value.values() for item in _all_json_strings(nested)}
    if isinstance(value, list):
        return {item for nested in value for item in _all_json_strings(nested)}
    return set()


def test_benchmark_version_split_and_case_kind_are_exact() -> None:
    assert BENCHMARK_SCHEMA_VERSION == "0.1"
    assert BENCHMARK_ID == "palintrace-controlled-v0.1"
    assert tuple(BenchmarkSplit) == (
        BenchmarkSplit.DEVELOPMENT,
        BenchmarkSplit.HELD_OUT,
    )
    assert [split.value for split in BenchmarkSplit] == ["development", "held_out"]
    assert tuple(BenchmarkCaseKind) == (
        BenchmarkCaseKind.STATIC_MUTATION,
        BenchmarkCaseKind.CLEAN_CONTROL,
        BenchmarkCaseKind.RETRIEVAL_CHALLENGE,
    )
    assert [kind.value for kind in BenchmarkCaseKind] == [
        "static_mutation",
        "clean_control",
        "retrieval_challenge",
    ]


def test_benchmark_spec_loads_and_serializes_deterministically(
    benchmark_spec: BenchmarkSpec,
) -> None:
    first = benchmark_spec.to_json()
    second = load_benchmark_spec(BENCHMARK_PATH).to_json()
    assert first == second
    assert BenchmarkSpec.model_validate_json(first) == benchmark_spec
    assert benchmark_spec.schema_version == BENCHMARK_SCHEMA_VERSION
    assert benchmark_spec.benchmark_id == BENCHMARK_ID


def test_benchmark_canonical_sha_is_frozen(benchmark_spec: BenchmarkSpec) -> None:
    digest = hashlib.sha256(benchmark_spec.to_json(indent=None).encode("utf-8")).hexdigest()
    assert digest == BENCHMARK_SPEC_SHA256
    assert digest == "90793984a6e191cad4cd276c185dfe8a1eac814035493a101768a96051a240f1"


def test_every_benchmark_fixture_file_hash_is_frozen() -> None:
    expected = json.loads(HASH_MANIFEST_PATH.read_text(encoding="utf-8"))
    fixture_paths = {
        path.as_posix() for path in FIXTURE_ROOT.iterdir() if path.is_file()
    }
    assert set(expected) == fixture_paths
    actual = {
        relative_path: hashlib.sha256(Path(relative_path).read_bytes()).hexdigest()
        for relative_path in sorted(expected)
    }
    assert actual == expected


def test_exact_static_held_out_matrix_is_frozen(benchmark_spec: BenchmarkSpec) -> None:
    counts = Counter(case.defect_class for case in benchmark_spec.static_mutation_cases)
    assert counts == {
        DefectClass.ORPHANED_PROVENANCE: 9,
        DefectClass.REDUNDANCY_BLOAT: 6,
        DefectClass.STALE_ACTIVE: 6,
        DefectClass.PRIVACY_SCOPE_VIOLATION: 6,
        DefectClass.UNSUPPORTED_CLAIM: 12,
    }
    assert len(benchmark_spec.static_mutation_cases) == 39
    assert all(
        case.split is BenchmarkSplit.HELD_OUT
        and case.kind is BenchmarkCaseKind.STATIC_MUTATION
        and isinstance(case.mutation_request, MutationRequest)
        for case in benchmark_spec.static_mutation_cases
    )
    assert {
        case.defect_class for case in benchmark_spec.static_mutation_cases
    }.isdisjoint(
        {DefectClass.INTERNAL_CONTRADICTION, DefectClass.INJECTED_INSTRUCTION}
    )


def test_static_subtype_and_domain_matrices_are_frozen(
    benchmark_spec: BenchmarkSpec,
) -> None:
    orphaned = Counter(
        case.subtype
        for case in benchmark_spec.static_mutation_cases
        if case.defect_class is DefectClass.ORPHANED_PROVENANCE
    )
    privacy = Counter(
        case.subtype
        for case in benchmark_spec.static_mutation_cases
        if case.defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION
    )
    unsupported_domains = {
        case.semantic_domain
        for case in benchmark_spec.static_mutation_cases
        if case.defect_class is DefectClass.UNSUPPORTED_CLAIM
    }
    assert orphaned == {"missing_transcript": 3, "missing_turn": 3, "invalid_span": 3}
    assert privacy == {"cross_user_copy": 3, "cross_agent_copy": 3}
    assert len(unsupported_domains) >= 4
    assert None not in unsupported_domains


def test_exact_clean_control_matrix_is_curated_and_policy_explicit(
    benchmark_spec: BenchmarkSpec,
) -> None:
    counts = Counter(case.defect_class for case in benchmark_spec.clean_control_cases)
    assert counts == {
        DefectClass.ORPHANED_PROVENANCE: 3,
        DefectClass.REDUNDANCY_BLOAT: 3,
        DefectClass.STALE_ACTIVE: 3,
        DefectClass.PRIVACY_SCOPE_VIOLATION: 3,
        DefectClass.UNSUPPORTED_CLAIM: 3,
    }
    assert len(benchmark_spec.clean_control_cases) == 15
    fixtures = {fixture.fixture_id: fixture for fixture in benchmark_spec.fixtures}
    assert all(
        fixtures[case.base_fixture_id].base_store_status
        is BaseStoreStatus.CURATED_CLEAN
        for case in benchmark_spec.clean_control_cases
    )
    privacy_controls = [
        case
        for case in benchmark_spec.clean_control_cases
        if case.defect_class is DefectClass.PRIVACY_SCOPE_VIOLATION
    ]
    assert all(case.scope_policy_fixture_id is not None for case in privacy_controls)


def test_three_held_out_fixtures_satisfy_declared_input_principles(
    benchmark_spec: BenchmarkSpec,
) -> None:
    assert tuple(fixture.fixture_id for fixture in benchmark_spec.fixtures) == ("H1", "H2", "H3")
    assert all(
        fixture.base_store_status is BaseStoreStatus.CURATED_CLEAN
        for fixture in benchmark_spec.fixtures
    )
    inputs = _fixture_inputs(benchmark_spec)
    for store, transcripts in inputs.values():
        assert len(store.memories) == 8
        assert sum(bool(memory.source_refs) for memory in store.memories) == 8
        assert all(
            memory.scope.user_id is not None
            and memory.scope.agent_id is not None
            and memory.scope.session_id is not None
            for memory in store.memories
        )
        assert len({memory.id for memory in store.memories}) == len(store.memories)
        assert len({memory.content for memory in store.memories}) == len(store.memories)
        assert all(memory.supersedes == () for memory in store.memories)
        assert len(transcripts.transcripts) == 8


def test_fixture_eligibility_validation_uses_no_detector(
    benchmark_spec: BenchmarkSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_checker_call(*args: object, **kwargs: object) -> None:
        pytest.fail(f"fixture eligibility must not run a checker: {args}, {kwargs}")

    for checker_class in (
        OrphanedProvenanceChecker,
        RedundancyBloatChecker,
        StaleActiveChecker,
        PrivacyScopeViolationChecker,
    ):
        monkeypatch.setattr(checker_class, "check", forbidden_checker_call)

    validate_benchmark_fixture_eligibility(
        benchmark_spec,
        repository_root=REPOSITORY_ROOT,
    )


def test_unmutated_structural_fixture_controls_are_clean(
    benchmark_spec: BenchmarkSpec,
) -> None:
    policy = load_scope_policy(FIXTURE_ROOT / "scope_policy.json")
    for store, transcripts in _fixture_inputs(benchmark_spec).values():
        assert OrphanedProvenanceChecker().check(
            store,
            transcripts=transcripts,
        ).findings == ()
        assert RedundancyBloatChecker().check(store).findings == ()
        assert StaleActiveChecker().check(store).findings == ()
        assert PrivacyScopeViolationChecker(policy).check(store).findings == ()


def test_semantic_fixture_cleanliness_uses_explicit_source_values_not_minilm(
    benchmark_spec: BenchmarkSpec,
) -> None:
    inputs = _fixture_inputs(benchmark_spec)
    memories = {
        memory.id: (memory, transcripts)
        for store, transcripts in inputs.values()
        for memory in store.memories
    }
    assert set(memories) == set(EXPECTED_SOURCE_VALUES)
    for memory_id, expected_value in EXPECTED_SOURCE_VALUES.items():
        memory, transcripts = memories[memory_id]
        assert expected_value in memory.content
        assert expected_value in _source_text(memory, transcripts)


def test_all_static_mutation_specs_are_constructible_without_checker_execution(
    benchmark_spec: BenchmarkSpec,
) -> None:
    inputs = _fixture_inputs(benchmark_spec)
    for case in benchmark_spec.static_mutation_cases:
        store, transcripts = inputs[case.base_fixture_id]
        mutation = mutate(store, case.mutation_request, transcripts)
        manifest = mutation.manifest
        assert manifest.defect_class is case.defect_class
        assert manifest.subtype == case.subtype
        assert manifest.base_store_status is BaseStoreStatus.CURATED_CLEAN
        assert manifest.base_store_digest == semantic_store_digest(store)
        assert manifest.gold_label.unit in {GoldLabelUnit.MEMORY, GoldLabelUnit.MEMORY_PAIR}
        assert manifest.gold_label.memory_ids
        assert all(
            mutation.mutated_store.get(memory_id) is not None
            for memory_id in manifest.gold_label.memory_ids
        )


def test_unsupported_replacements_are_absent_from_all_declared_source_segments(
    benchmark_spec: BenchmarkSpec,
) -> None:
    inputs = _fixture_inputs(benchmark_spec)
    banned_values = {
        "Python",
        "Rust",
        "Berlin",
        "Munich",
        "Emacs",
        "Vim",
        "Neovim",
        "VS Code",
        "Atlas Corp",
        "Orion Labs",
    }
    unsupported_cases = [
        case
        for case in benchmark_spec.static_mutation_cases
        if case.defect_class is DefectClass.UNSUPPORTED_CLAIM
    ]
    assert len(unsupported_cases) == 12
    for case in unsupported_cases:
        store, transcripts = inputs[case.base_fixture_id]
        target_id = case.mutation_request.target_memory_id
        assert target_id is not None
        memory = store.get(target_id)
        assert memory is not None
        source_segments = tuple(
            _source_text(memory, transcripts) for _ in memory.source_refs
        )
        replacement = case.mutation_request.replace_to
        assert replacement is not None
        assert all(replacement not in segment for segment in source_segments)
        assert replacement not in banned_values
        assert case.mutation_request.replace_from not in banned_values


def test_retrieval_matrix_and_condition_are_frozen_and_constructible(
    benchmark_spec: BenchmarkSpec,
) -> None:
    assert len(benchmark_spec.retrieval_conditions) == 1
    condition = benchmark_spec.retrieval_conditions[0]
    assert condition.condition_id == "lexical-baseline-k3"
    assert condition.policy is RetrievalSufficiencyPolicy.ALL_EXPECTED
    assert condition.top_k == 3
    assert condition.retriever_kind == "experimental_lexical"
    assert condition.retriever_config_version == "0.1"
    assert len(benchmark_spec.retrieval_cases) == 12
    assert Counter(case.base_fixture_id for case in benchmark_spec.retrieval_cases) == {
        "H1": 4,
        "H2": 4,
        "H3": 4,
    }

    inputs = _fixture_inputs(benchmark_spec)
    target_contents: set[str] = set()
    for case in benchmark_spec.retrieval_cases:
        store, transcripts = inputs[case.base_fixture_id]
        target_id = case.mutation_request.target_memory_id
        assert target_id is not None
        target = store.get(target_id)
        assert target is not None
        target_contents.add(target.content)
        mutation = mutate(store, case.mutation_request, transcripts)
        probe = mutation.manifest.retrieval_probe
        assert probe is not None
        assert probe.expected_memory_ids == (target_id,)
        assert mutation.mutated_store.get(target_id) is not None
        assert case.policy is condition.policy
        assert case.top_k == condition.top_k
    assert len(target_contents) == 12


def test_development_editor_case_is_not_reused_by_retrieval_matrix(
    benchmark_spec: BenchmarkSpec,
) -> None:
    inputs = _fixture_inputs(benchmark_spec)
    development_pair = (
        "Which editor does the user prefer?",
        "User's favorite editor is Neovim.",
    )
    held_out_pairs = {
        (
            case.mutation_request.query,
            inputs[case.base_fixture_id][0].get(
                case.mutation_request.target_memory_id or ""
            ).content,
        )
        for case in benchmark_spec.retrieval_cases
    }
    assert development_pair not in held_out_pairs
    assert all("Neovim" not in content for _, content in held_out_pairs)


def test_development_holdout_collision_sanity_checks(
    benchmark_spec: BenchmarkSpec,
) -> None:
    development_store = load_store("examples/mutation-store.json")
    development_transcripts = load_transcripts("examples/mutation-transcripts.json")
    inputs = _fixture_inputs(benchmark_spec)
    held_out_memory_ids = {
        memory.id for store, _ in inputs.values() for memory in store.memories
    }
    held_out_contents = {
        memory.content for store, _ in inputs.values() for memory in store.memories
    }
    held_out_transcript_ids = {
        transcript.id
        for _, transcripts in inputs.values()
        for transcript in transcripts.transcripts
    }
    assert held_out_memory_ids.isdisjoint(
        memory.id for memory in development_store.memories
    )
    assert held_out_contents.isdisjoint(
        memory.content for memory in development_store.memories
    )
    assert held_out_transcript_ids.isdisjoint(
        transcript.id for transcript in development_transcripts.transcripts
    )

    prior_probe_paths = tuple(Path("tests/fixtures").glob("*probe_v0.1.json"))
    prior_probe_values = set()
    for path in prior_probe_paths:
        prior_probe_values |= _all_json_strings(
            json.loads(path.read_text(encoding="utf-8"))
        )
    held_out_case_ids = {
        case.case_id
        for case in (
            *benchmark_spec.static_mutation_cases,
            *benchmark_spec.clean_control_cases,
            *benchmark_spec.retrieval_cases,
        )
    }
    assert held_out_case_ids.isdisjoint(prior_probe_values)


def test_unsupported_pairs_do_not_exactly_reuse_part_four_semantic_probe(
    benchmark_spec: BenchmarkSpec,
) -> None:
    probe = json.loads(
        Path("tests/fixtures/semantic_probe_v0.1.json").read_text(encoding="utf-8")
    )
    development_pairs = {(case["premise"], case["hypothesis"]) for case in probe}
    inputs = _fixture_inputs(benchmark_spec)
    held_out_pairs = set()
    for case in benchmark_spec.static_mutation_cases:
        if case.defect_class is not DefectClass.UNSUPPORTED_CLAIM:
            continue
        store, transcripts = inputs[case.base_fixture_id]
        target = store.get(case.mutation_request.target_memory_id or "")
        assert target is not None
        held_out_pairs.add((_source_text(target, transcripts), target.content))
    assert held_out_pairs.isdisjoint(development_pairs)


def test_development_artifacts_are_registered_but_not_held_out_fixtures(
    benchmark_spec: BenchmarkSpec,
) -> None:
    assert "examples/mutation-store.json" in benchmark_spec.development_artifacts
    assert "examples/mutation-transcripts.json" in benchmark_spec.development_artifacts
    assert {
        "tests/fixtures/semantic_probe_v0.1.json",
        "tests/fixtures/evidence_composition_probe_v0.1.json",
        "tests/fixtures/contradiction_pair_probe_v0.1.json",
        "tests/fixtures/instruction_contradiction_probe_v0.1.json",
        "tests/fixtures/injected_instruction_probe_v0.1.json",
    } <= set(benchmark_spec.development_artifacts)
    held_out_paths = {
        path
        for fixture in benchmark_spec.fixtures
        for path in (
            fixture.store_path,
            fixture.transcripts_path,
            fixture.scope_policy_path,
        )
        if path is not None
    }
    assert held_out_paths.isdisjoint(benchmark_spec.development_artifacts)


def test_benchmark_spec_and_json_have_no_output_or_metric_fields(
    benchmark_spec: BenchmarkSpec,
) -> None:
    forbidden = {
        "detected",
        "prediction",
        "result",
        "finding",
        "finding_id",
        "score",
        "confidence",
        "latency",
        "precision",
        "recall",
        "f1",
        "accuracy",
    }
    payload = benchmark_spec.model_dump(mode="json")
    assert _all_json_keys(payload).isdisjoint(forbidden)
    assert "execution_status" not in BenchmarkSpec.model_fields
    assert not any(
        path.name
        in {
            "results.csv",
            "benchmark-results.json",
            "performance.md",
            "leaderboard.md",
        }
        for path in FIXTURE_ROOT.iterdir()
    )


def test_benchmark_module_cannot_execute_mutations_checkers_or_models() -> None:
    tree = ast.parse(
        BENCHMARK_MODULE_PATH.read_text(encoding="utf-8"),
        filename=str(BENCHMARK_MODULE_PATH),
    )
    forbidden_names = {
        "mutate",
        "OrphanedProvenanceChecker",
        "RedundancyBloatChecker",
        "StaleActiveChecker",
        "PrivacyScopeViolationChecker",
        "UnsupportedClaimChecker",
        "LocalNLISemanticJudge",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            violations.append(f"{node.lineno}:{node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            violations.append(f"{node.lineno}:{node.attr}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = (
                {alias.name for alias in node.names}
                if isinstance(node, ast.Import)
                else {alias.name for alias in node.names}
            )
            if imported & forbidden_names:
                violations.append(f"{node.lineno}:forbidden import")
    assert violations == []


def test_spec_self_validation_rejects_frozen_boundary_changes(
    benchmark_spec: BenchmarkSpec,
) -> None:
    invalid_payloads: list[dict[str, Any]] = []

    wrong_version = benchmark_spec.model_dump()
    wrong_version["schema_version"] = "0.2"
    invalid_payloads.append(wrong_version)

    wrong_id = benchmark_spec.model_dump()
    wrong_id["benchmark_id"] = "mutable-benchmark"
    invalid_payloads.append(wrong_id)

    duplicate_case = benchmark_spec.model_dump()
    duplicate_case["clean_control_cases"][0]["case_id"] = duplicate_case[
        "static_mutation_cases"
    ][0]["case_id"]
    invalid_payloads.append(duplicate_case)

    missing_case = benchmark_spec.model_dump()
    missing_case["static_mutation_cases"] = missing_case["static_mutation_cases"][:-1]
    invalid_payloads.append(missing_case)

    wrong_fixture_status = benchmark_spec.model_dump()
    wrong_fixture_status["fixtures"][0]["base_store_status"] = "unknown"
    invalid_payloads.append(wrong_fixture_status)

    mixed_request = benchmark_spec.model_dump()
    mixed_request["static_mutation_cases"][0]["mutation_request"][
        "defect_class"
    ] = "stale_active"
    invalid_payloads.append(mixed_request)

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            BenchmarkSpec.model_validate(payload)


def test_loading_path_has_no_evaluation_cli_or_manifest_audit_input() -> None:
    parser = cli.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert set(command_action.choices) == {
        "dump",
        "mutate",
        "audit",
        "retrieval-audit",
    }
    audit_parser = command_action.choices["audit"]
    assert "manifest" not in {action.dest for action in audit_parser._actions}


def test_loader_reports_missing_and_invalid_spec_files(tmp_path: Path) -> None:
    with pytest.raises(EvaluationInputError, match="could not read"):
        load_benchmark_spec(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="invalid benchmark specification"):
        load_benchmark_spec(invalid)
