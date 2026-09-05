"""Public command entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from palintrace import cli
from palintrace.adapters import AdapterError, adapter_capabilities
from palintrace.audit import AuditReport, run_aggregate_audit
from palintrace.checkers import CheckerResult, load_scope_policy
from palintrace.sarif import render_audit_report_sarif, render_sarif
from palintrace.semantics import (
    LocalNLISemanticJudge,
    SemanticJudge,
    SemanticJudgeError,
    SemanticJudgment,
)
from palintrace.serialization import load_store, load_transcripts

_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    commands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    audit = commands.choices["audit"]
    audit_help = "run one checker or all checkers on normalized data"
    audit.description = audit_help
    checker_action = next(action for action in audit._actions if action.dest == "checker")
    checker_action.choices = (*cli.CHECKER_NAMES, "all")
    next(action for action in commands._choices_actions if action.dest == "audit").help = (
        audit_help
    )
    capabilities = commands.add_parser(
        "capabilities", help="show the normalized field support for one adapter"
    )
    capabilities.add_argument(
        "--adapter", choices=("file", "mem0", "graphiti", "letta"), required=True
    )
    capabilities.add_argument(
        "--output", type=Path, help="write capability JSON to this path instead of stdout"
    )
    for name in ("audit", "retrieval-audit"):
        commands.choices[name].add_argument(
            "--fail-on",
            choices=tuple(_SEVERITY_RANK),
            help="return exit 1 when findings meet or exceed this severity",
        )
        commands.choices[name].add_argument(
            "--sarif-output",
            type=Path,
            help="write a SARIF 2.1.0 projection to this path",
        )
    return parser


def _gate_triggered(result: CheckerResult, fail_on: str | None) -> bool:
    return (
        fail_on is not None
        and bool(result.findings)
        and _SEVERITY_RANK[result.severity] >= _SEVERITY_RANK[fail_on]
    )


def _aggregate_gate_triggered(report: AuditReport, fail_on: str | None) -> bool:
    return any(_gate_triggered(result, fail_on) for result in report.results)


class _ConfiguredSemanticJudge:
    """Represent complete semantic configuration without loading its model."""

    def __init__(self, *, model_id: str, revision: str) -> None:
        self.judge_id = f"hf-nli:{model_id}"
        self.judge_version = revision

    def judge(self, *, premise: str, hypothesis: str) -> SemanticJudgment:
        raise AssertionError("configured semantic judge marker must not be invoked")


def _validate_aggregate_semantic_options(
    args: argparse.Namespace,
) -> tuple[str, str] | None:
    model_id = args.semantic_model_id
    revision = args.semantic_model_revision
    if model_id is None and revision is None:
        return None
    if model_id is None or not model_id.strip():
        raise ValueError("--checker all requires a nonblank --semantic-model-id when configured")
    if revision is None or not revision.strip():
        raise ValueError(
            "--checker all requires a nonblank --semantic-model-revision when configured"
        )
    return model_id, revision


def _validate_aggregate_output(args: argparse.Namespace) -> None:
    if args.output is None:
        return
    input_paths = {args.store.resolve()}
    if args.transcripts is not None:
        input_paths.add(args.transcripts.resolve())
    if args.scope_policy is not None:
        input_paths.add(args.scope_policy.resolve())
    if args.output.resolve() in input_paths:
        raise ValueError("audit output must not overwrite input files")


def _run_aggregate_audit(args: argparse.Namespace) -> AuditReport:
    _validate_sarif_output(args)
    semantic_configuration = _validate_aggregate_semantic_options(args)
    _validate_aggregate_output(args)

    store = load_store(args.store)
    transcripts = load_transcripts(args.transcripts) if args.transcripts is not None else None
    scope_policy = (
        load_scope_policy(args.scope_policy) if args.scope_policy is not None else None
    )

    semantic_judge: SemanticJudge | None = None
    if semantic_configuration is not None:
        model_id, revision = semantic_configuration
        if transcripts is None:
            semantic_judge = _ConfiguredSemanticJudge(model_id=model_id, revision=revision)
        else:
            try:
                semantic_judge = LocalNLISemanticJudge(
                    model_id=model_id,
                    revision=revision,
                    device="cpu",
                )
            except SemanticJudgeError as error:
                raise ValueError(str(error)) from None

    return run_aggregate_audit(
        store,
        transcripts=transcripts,
        scope_policy=scope_policy,
        semantic_judge=semantic_judge,
    )


def _validate_sarif_output(args: argparse.Namespace) -> None:
    if args.sarif_output is None:
        return
    compared_paths = [args.output]
    if args.command == "audit":
        compared_paths.extend((args.store, args.transcripts, args.scope_policy))
    else:
        compared_paths.append(args.observation)
    sarif_output = args.sarif_output.resolve()
    if any(path is not None and path.resolve() == sarif_output for path in compared_paths):
        raise ValueError("--sarif-output must not overwrite an input or canonical output")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capabilities":
        try:
            text = adapter_capabilities(args.adapter).to_json(args.output)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.output is None:
            sys.stdout.write(text)
        return 0
    if args.command == "audit" and args.checker == "all":
        try:
            report = _run_aggregate_audit(args)
            text = report.to_json(args.output)
            if args.sarif_output is not None:
                render_audit_report_sarif(report, args.sarif_output)
        except (AdapterError, OSError, ValueError) as error:
            parser.error(str(error))
        if args.output is None:
            sys.stdout.write(text)
        return 1 if _aggregate_gate_triggered(report, args.fail_on) else 0
    if args.command not in {"audit", "retrieval-audit"}:
        return cli.main(argv)

    try:
        _validate_sarif_output(args)
        if args.command == "audit":
            text = cli._run_audit(args)
        else:
            text = cli._run_recorded_retrieval_audit(args)
    except (AdapterError, OSError, ValueError) as error:
        parser.error(str(error))

    result = CheckerResult.model_validate_json(text)
    try:
        if args.sarif_output is not None:
            render_sarif(result, args.sarif_output)
    except OSError as error:
        parser.error(str(error))
    if args.output is None:
        sys.stdout.write(text)
    return 1 if _gate_triggered(result, args.fail_on) else 0
