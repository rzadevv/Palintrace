"""Behavioural CLI contract that replaces the former byte freeze on cli.py."""

from __future__ import annotations

import argparse

import pytest

from palintrace import cli, command
from palintrace.retrieval import RetrievalSufficiencyPolicy

PUBLIC_COMMANDS = (
    "audit",
    "capabilities",
    "dump",
    "mutate",
    "preflight",
    "retrieval-audit",
)


def _subparsers(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction[argparse.ArgumentParser]:
    return next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == dest)


def test_public_commands_are_exactly_the_documented_set() -> None:
    commands = _subparsers(command.build_parser())

    assert tuple(sorted(commands.choices)) == PUBLIC_COMMANDS


def test_audit_checker_choices_are_the_five_public_ids_plus_all() -> None:
    commands = _subparsers(command.build_parser())
    checker_action = _action(commands.choices["audit"], "checker")

    assert cli.CHECKER_NAMES == (
        "orphaned_provenance",
        "redundancy_bloat",
        "stale_active",
        "privacy_scope_violation",
        "unsupported_claim",
    )
    assert tuple(checker_action.choices) == (*cli.CHECKER_NAMES, "all")
    assert checker_action.required is True


def test_semantic_model_options_have_no_defaults_on_any_command() -> None:
    commands = _subparsers(command.build_parser())

    for name in ("audit", "preflight"):
        for dest in ("semantic_model_id", "semantic_model_revision"):
            action = _action(commands.choices[name], dest)
            assert action.default is None
            assert action.required is False
            assert action.choices is None


def test_unsupported_claim_never_builds_a_judge_without_explicit_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("semantic judge must not be constructed implicitly")

    monkeypatch.setattr(cli, "LocalNLISemanticJudge", _forbidden)
    monkeypatch.setattr(command, "LocalNLISemanticJudge", _forbidden)
    base = [
        "audit",
        "--store",
        "examples/mutation-store.json",
        "--transcripts",
        "examples/mutation-transcripts.json",
        "--checker",
        "unsupported_claim",
    ]

    for incomplete in (
        [],
        ["--semantic-model-id", "test/model"],
        ["--semantic-model-revision", "revision-test"],
        ["--semantic-model-id", " ", "--semantic-model-revision", "revision-test"],
        ["--semantic-model-id", "test/model", "--semantic-model-revision", " "],
    ):
        with pytest.raises(SystemExit, match="2"):
            command.main([*base, *incomplete])


def test_retrieval_audit_keeps_its_recorded_observation_and_policy_contract() -> None:
    commands = _subparsers(command.build_parser())
    retrieval_audit = commands.choices["retrieval-audit"]

    observation = _action(retrieval_audit, "observation")
    policy = _action(retrieval_audit, "policy")

    assert observation.required is True
    assert policy.required is True
    assert tuple(policy.choices) == tuple(RetrievalSufficiencyPolicy)


@pytest.mark.parametrize("name", ["audit", "retrieval-audit"])
def test_gating_options_keep_their_choices_and_absent_default(name: str) -> None:
    commands = _subparsers(command.build_parser())
    fail_on = _action(commands.choices[name], "fail_on")

    assert tuple(fail_on.choices) == ("info", "warning", "error")
    assert fail_on.default is None
    assert fail_on.required is False


def test_dump_keeps_the_adapter_and_scope_flags_probe_inputs_were_built_with() -> None:
    commands = _subparsers(command.build_parser())
    dump = commands.choices["dump"]

    assert tuple(_action(dump, "adapter").choices) == ("file", "mem0", "graphiti", "letta")
    assert _action(dump, "adapter").required is True
    for dest in ("source", "output", "user_id", "agent_id", "session_id", "group_ids"):
        assert _action(dump, dest).default in (None, [])
    assert _action(dump, "include_raw").default is False


def test_cli_does_not_import_evaluation_or_probe_modules() -> None:
    source = (cli.__file__ or "")
    text = open(source, encoding="utf-8").read()

    assert "palintrace.evaluation" not in text
    assert "palintrace.mutations.shadowing" not in text
