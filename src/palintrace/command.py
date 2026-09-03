"""Public command entry point with optional audit exit policy."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from palintrace import cli
from palintrace.adapters import AdapterError
from palintrace.checkers import CheckerResult

_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    commands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for name in ("audit", "retrieval-audit"):
        commands.choices[name].add_argument(
            "--fail-on",
            choices=tuple(_SEVERITY_RANK),
            help="return exit 1 when findings meet or exceed this severity",
        )
    return parser


def _gate_triggered(result: CheckerResult, fail_on: str | None) -> bool:
    return (
        fail_on is not None
        and bool(result.findings)
        and _SEVERITY_RANK[result.severity] >= _SEVERITY_RANK[fail_on]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in {"audit", "retrieval-audit"}:
        return cli.main(argv)

    try:
        if args.command == "audit":
            text = cli._run_audit(args)
        else:
            text = cli._run_recorded_retrieval_audit(args)
    except (AdapterError, OSError, ValueError) as error:
        parser.error(str(error))

    result = CheckerResult.model_validate_json(text)
    if args.output is None:
        sys.stdout.write(text)
    return 1 if _gate_triggered(result, args.fail_on) else 0
