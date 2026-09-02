from pathlib import Path

import pytest

import palintrace.cli as cli
from palintrace.checkers import CheckerResult
from palintrace.cli import main
from palintrace.models import (
    NormalizedMemory,
    NormalizedStore,
    ProvenanceStatus,
    SourceRef,
    Transcript,
    TranscriptSet,
    TranscriptTurn,
)
from palintrace.mutations import MutationRequest, mutate
from palintrace.semantics import (
    SemanticDependencyError,
    SemanticJudgeError,
    SemanticJudgment,
    SemanticModelConfigError,
    SemanticRelation,
    SemanticUsage,
)
from palintrace.serialization import dumps_transcripts, load_store, load_transcripts
from palintrace.taxonomy import DefectClass


class _FakeLocalJudge:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        relation: SemanticRelation,
    ) -> None:
        self.judge_id = f"hf-nli:{model_id}"
        self.judge_version = revision
        self.relation = relation
        self.calls: list[tuple[str, str]] = []

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        self.calls.append((premise, hypothesis))
        return SemanticJudgment(
            relation=self.relation,
            score=0.8,
            usage=SemanticUsage(model_calls=1, input_tokens=7, output_tokens=0),
        )


def _patch_local_judge(
    monkeypatch: pytest.MonkeyPatch,
    relation: SemanticRelation,
) -> tuple[list[tuple[str, str, str]], list[_FakeLocalJudge]]:
    constructor_calls: list[tuple[str, str, str]] = []
    judges: list[_FakeLocalJudge] = []

    def construct(*, model_id: str, revision: str, device: str) -> _FakeLocalJudge:
        constructor_calls.append((model_id, revision, device))
        judge = _FakeLocalJudge(model_id=model_id, revision=revision, relation=relation)
        judges.append(judge)
        return judge

    monkeypatch.setattr(cli, "LocalNLISemanticJudge", construct)
    return constructor_calls, judges


def _unexpected_local_judge(**kwargs: object) -> None:
    pytest.fail(f"semantic model should not be constructed: {kwargs}")


def _write_semantic_audit_inputs(tmp_path: Path) -> tuple[Path, Path]:
    store_path = tmp_path / "semantic-store.json"
    transcripts_path = tmp_path / "semantic-transcripts.json"
    NormalizedStore(
        adapter="test",
        memories=(
            NormalizedMemory(
                id="m1",
                content="User prefers Rust.",
                source_refs=(SourceRef(transcript_id="t1", turn_idx=0),),
                provenance_status=ProvenanceStatus.DECLARED,
            ),
        ),
    ).to_json(store_path)
    transcripts = TranscriptSet(
        transcripts=(
            Transcript(
                id="t1",
                turns=(
                    TranscriptTurn(index=0, role="user", content="I prefer Python."),
                ),
            ),
        )
    )
    transcripts_path.write_text(dumps_transcripts(transcripts), encoding="utf-8")
    return store_path, transcripts_path


def _semantic_audit_arguments(
    store_path: Path,
    transcripts_path: Path,
    *,
    output: Path | None = None,
) -> list[str]:
    arguments = [
        "audit",
        "--store",
        str(store_path),
        "--transcripts",
        str(transcripts_path),
        "--checker",
        "unsupported_claim",
        "--semantic-model-id",
        "test/model",
        "--semantic-model-revision",
        "revision-test",
    ]
    if output is not None:
        arguments.extend(("--output", str(output)))
    return arguments


def test_audit_cli_writes_deterministic_checker_result(tmp_path: Path) -> None:
    base_store = load_store("examples/mutation-store.json")
    transcripts = load_transcripts("examples/mutation-transcripts.json")
    mutation_result = mutate(
        base_store,
        MutationRequest(
            defect_class=DefectClass.ORPHANED_PROVENANCE,
            subtype="missing_turn",
            target_memory_id="preference-python",
        ),
        transcripts,
    )
    store_path = tmp_path / "store.json"
    transcripts_path = tmp_path / "transcripts.json"
    output = tmp_path / "findings.json"
    mutation_result.mutated_store.to_json(store_path)
    transcripts_path.write_text(dumps_transcripts(transcripts), encoding="utf-8")
    before_store = store_path.read_bytes()
    before_transcripts = transcripts_path.read_bytes()

    result = main(
        [
            "audit",
            "--store",
            str(store_path),
            "--transcripts",
            str(transcripts_path),
            "--checker",
            "orphaned_provenance",
            "--output",
            str(output),
        ]
    )
    checker_result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))

    assert result == 0
    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].memory_ids == mutation_result.manifest.gold_label.memory_ids
    assert store_path.read_bytes() == before_store
    assert transcripts_path.read_bytes() == before_transcripts


def test_audit_cli_emits_clean_result_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "audit",
            "--store",
            "examples/mutation-store.json",
            "--transcripts",
            "examples/mutation-transcripts.json",
            "--checker",
            "orphaned_provenance",
        ]
    )
    checker_result = CheckerResult.model_validate_json(capsys.readouterr().out)

    assert result == 0
    assert checker_result.findings == ()


def test_audit_cli_surfaces_missing_required_transcript_set(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--checker",
                "orphaned_provenance",
            ]
        )

    assert "orphaned_provenance checker requires a TranscriptSet" in capsys.readouterr().err


@pytest.mark.parametrize("input_name", ["store", "transcripts"])
def test_audit_cli_rejects_overwriting_inputs(input_name: str, tmp_path: Path) -> None:
    store_path = Path("examples/mutation-store.json")
    transcripts_path = Path("examples/mutation-transcripts.json")
    output = store_path if input_name == "store" else transcripts_path

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                str(store_path),
                "--transcripts",
                str(transcripts_path),
                "--checker",
                "orphaned_provenance",
                "--output",
                str(output),
            ]
        )


def test_audit_cli_has_no_manifest_argument(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--transcripts",
                "examples/mutation-transcripts.json",
                "--checker",
                "orphaned_provenance",
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )


def test_unsupported_audit_cli_constructs_explicit_cpu_judge_and_emits_finding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constructor_calls, judges = _patch_local_judge(monkeypatch, SemanticRelation.NEUTRAL)
    store_path, transcripts_path = _write_semantic_audit_inputs(tmp_path)
    output = tmp_path / "unsupported.json"

    result = main(_semantic_audit_arguments(store_path, transcripts_path, output=output))
    checker_result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))

    assert result == 0
    assert constructor_calls == [("test/model", "revision-test", "cpu")]
    assert judges[0].calls == [("I prefer Python.", "User prefers Rust.")]
    assert checker_result.checker_id == "unsupported_claim"
    assert len(checker_result.findings) == 1
    evidence = checker_result.findings[0].evidence[0]
    assert evidence.data["semantic_relation"] == "neutral"
    assert evidence.data["composition_style"] == "plain"


def test_unsupported_audit_cli_preserves_entailment_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_local_judge(monkeypatch, SemanticRelation.ENTAILMENT)
    store_path, transcripts_path = _write_semantic_audit_inputs(tmp_path)
    output = tmp_path / "supported.json"

    result = main(_semantic_audit_arguments(store_path, transcripts_path, output=output))
    checker_result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))

    assert result == 0
    assert checker_result.findings == ()
    assert checker_result.stats.details["assessed_memories"] == 1
    assert checker_result.stats.details["entailment_judgments"] == 1


def test_unsupported_audit_cli_preserves_contradiction_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_local_judge(monkeypatch, SemanticRelation.CONTRADICTION)
    store_path, transcripts_path = _write_semantic_audit_inputs(tmp_path)
    output = tmp_path / "contradiction.json"

    result = main(_semantic_audit_arguments(store_path, transcripts_path, output=output))
    checker_result = CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))

    assert result == 0
    assert len(checker_result.findings) == 1
    assert checker_result.findings[0].evidence[0].data["semantic_relation"] == "contradiction"


def test_unsupported_audit_rejects_missing_transcripts_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _unexpected_local_judge)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--checker",
                "unsupported_claim",
                "--semantic-model-id",
                "test/model",
                "--semantic-model-revision",
                "revision-test",
            ]
        )

    assert "unsupported_claim checker requires --transcripts" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("semantic_arguments", "expected_error"),
    [
        (["--semantic-model-revision", "revision-test"], "--semantic-model-id"),
        (["--semantic-model-id", "test/model"], "--semantic-model-revision"),
        ([], "--semantic-model-id"),
        (
            [
                "--semantic-model-id",
                " ",
                "--semantic-model-revision",
                "revision-test",
            ],
            "--semantic-model-id",
        ),
        (
            [
                "--semantic-model-id",
                "test/model",
                "--semantic-model-revision",
                " ",
            ],
            "--semantic-model-revision",
        ),
    ],
)
def test_unsupported_audit_rejects_incomplete_or_blank_model_configuration(
    semantic_arguments: list[str],
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _unexpected_local_judge)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--transcripts",
                "examples/mutation-transcripts.json",
                "--checker",
                "unsupported_claim",
                *semantic_arguments,
            ]
        )

    assert expected_error in capsys.readouterr().err


@pytest.mark.parametrize(
    ("construction_error", "expected_error"),
    [
        (
            SemanticDependencyError(
                "local NLI requires the 'semantic-local' optional dependencies; "
                "install palintrace[semantic-local]"
            ),
            "install palintrace[semantic-local]",
        ),
        (SemanticModelConfigError("unsafe semantic model configuration"), "unsafe semantic model"),
        (SemanticJudgeError("failed to load local semantic model"), "failed to load"),
    ],
)
def test_unsupported_audit_surfaces_safe_model_setup_errors_without_traceback(
    construction_error: SemanticJudgeError,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store_path, transcripts_path = _write_semantic_audit_inputs(tmp_path)

    def fail_construction(**kwargs: object) -> None:
        raise construction_error

    monkeypatch.setattr(cli, "LocalNLISemanticJudge", fail_construction)

    with pytest.raises(SystemExit, match="2"):
        main(_semantic_audit_arguments(store_path, transcripts_path))

    error_output = capsys.readouterr().err
    assert expected_error in error_output
    assert "Traceback" not in error_output


@pytest.mark.parametrize(
    ("checker_name", "checker_arguments"),
    [
        (
            "orphaned_provenance",
            ["--transcripts", "examples/mutation-transcripts.json"],
        ),
        ("redundancy_bloat", []),
        ("stale_active", []),
        (
            "privacy_scope_violation",
            ["--scope-policy", "examples/scope-policy.json"],
        ),
    ],
)
def test_structural_audits_do_not_construct_semantic_models(
    checker_name: str,
    checker_arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _unexpected_local_judge)
    output = tmp_path / f"{checker_name}.json"

    result = main(
        [
            "audit",
            "--store",
            "examples/mutation-store.json",
            "--checker",
            checker_name,
            *checker_arguments,
            "--output",
            str(output),
        ]
    )

    assert result == 0
    CheckerResult.model_validate_json(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "checker_name",
    [
        "orphaned_provenance",
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
    ],
)
@pytest.mark.parametrize(
    "semantic_argument",
    ["--semantic-model-id", "--semantic-model-revision"],
)
def test_structural_audits_reject_semantic_configuration(
    checker_name: str,
    semantic_argument: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _unexpected_local_judge)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--checker",
                checker_name,
                semantic_argument,
                "irrelevant",
            ]
        )

    assert "semantic-model-revision are only valid for unsupported_claim" in (
        capsys.readouterr().err
    )


def test_unsupported_audit_rejects_scope_policy_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _unexpected_local_judge)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit",
                "--store",
                "examples/mutation-store.json",
                "--transcripts",
                "examples/mutation-transcripts.json",
                "--checker",
                "unsupported_claim",
                "--semantic-model-id",
                "test/model",
                "--semantic-model-revision",
                "revision-test",
                "--scope-policy",
                "examples/scope-policy.json",
            ]
        )

    assert "--scope-policy is only valid for privacy_scope_violation" in capsys.readouterr().err


def test_unsupported_audit_cannot_overwrite_transcripts_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path, transcripts_path = _write_semantic_audit_inputs(tmp_path)
    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _unexpected_local_judge)

    with pytest.raises(SystemExit, match="2"):
        main(
            _semantic_audit_arguments(
                store_path,
                transcripts_path,
                output=transcripts_path,
            )
        )


def test_audit_checker_choices_are_exactly_the_implemented_checkers() -> None:
    assert cli.CHECKER_NAMES == (
        "orphaned_provenance",
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
        "unsupported_claim",
    )
