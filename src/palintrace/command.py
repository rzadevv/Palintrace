"""Public command entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from palintrace import cli
from palintrace.adapters import AdapterError, adapter_capabilities
from palintrace.checkers import CheckerResult
from palintrace.sarif import render_sarif

_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    commands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
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
