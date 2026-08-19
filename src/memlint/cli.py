"""Command-line entry points for normalized exports, mutations, and audits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from memlint.adapters import AdapterDataError, AdapterError, FileAdapter
from memlint.adapters.graphiti import GraphitiAdapter
from memlint.adapters.letta import LettaAdapter
from memlint.adapters.mem0 import Mem0Adapter
from memlint.checkers import Checker, OrphanedProvenanceChecker, RedundancyBloatChecker
from memlint.models import MemoryScope, NormalizedStore
from memlint.mutations import (
    BaseStoreStatus,
    ConflictRelation,
    DistractorFamily,
    MutationRequest,
    mutate,
)
from memlint.serialization import load_store, load_transcripts
from memlint.taxonomy import DefectClass

CHECKER_FACTORIES: dict[str, type[Checker]] = {
    "orphaned_provenance": OrphanedProvenanceChecker,
    "redundancy_bloat": RedundancyBloatChecker,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memlint", description="Normalize agent memory stores")
    commands = parser.add_subparsers(dest="command", required=True)
    dump = commands.add_parser("dump", help="export one backend as a NormalizedStore")
    dump.add_argument("--adapter", choices=("file", "mem0", "graphiti", "letta"), required=True)
    dump.add_argument(
        "--source",
        type=Path,
        help="input file; required for file, optional fixture/export input for external adapters",
    )
    dump.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")

    dump.add_argument("--user-id", help="Mem0 filter or explicit normalized scope")
    dump.add_argument("--agent-id", help="Mem0 filter or Letta agent ID")
    dump.add_argument("--session-id", help="Mem0 run_id filter / normalized session scope")
    dump.add_argument("--page-size", type=int, default=100)

    dump.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    dump.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER"))
    dump.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    dump.add_argument("--group-id", action="append", dest="group_ids")
    dump.add_argument("--include-embeddings", action="store_true")

    dump.add_argument("--letta-base-url", default=os.getenv("LETTA_BASE_URL"))

    mutation = commands.add_parser("mutate", help="inject one deterministic defect")
    mutation.add_argument("--store", type=Path, required=True, help="base NormalizedStore JSON")
    mutation.add_argument("--transcripts", type=Path, help="TranscriptSet JSON when required")
    mutation.add_argument("--defect", choices=tuple(DefectClass), required=True)
    mutation.add_argument("--subtype")
    mutation.add_argument("--seed", type=int, default=0)
    mutation.add_argument("--target-id")
    mutation.add_argument("--replace-from")
    mutation.add_argument("--replace-to")
    mutation.add_argument("--conflict-relation", choices=tuple(ConflictRelation))
    mutation.add_argument("--destination-user-id")
    mutation.add_argument("--destination-agent-id")
    mutation.add_argument("--query")
    mutation.add_argument("--distractor-family", choices=tuple(DistractorFamily))
    mutation.add_argument("--distractor-count", type=int, default=3)
    mutation.add_argument(
        "--base-store-status",
        choices=tuple(BaseStoreStatus),
        default=BaseStoreStatus.UNKNOWN,
    )
    mutation.add_argument("--output", type=Path, required=True, help="mutated store JSON")
    mutation.add_argument("--manifest", type=Path, required=True, help="separate gold JSON")

    audit = commands.add_parser("audit", help="run one checker on normalized data")
    audit.add_argument("--store", type=Path, required=True, help="NormalizedStore JSON")
    audit.add_argument("--transcripts", type=Path, help="TranscriptSet JSON when required")
    audit.add_argument("--checker", choices=tuple(CHECKER_FACTORIES), required=True)
    audit.add_argument("--output", type=Path, help="write checker result JSON instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "mutate":
            _run_mutation(args)
            return 0
        if args.command == "audit":
            text = _run_audit(args)
            if args.output is None:
                sys.stdout.write(text)
            return 0
        store = _build_store(args)
        text = store.to_json(args.output)
    except (AdapterError, OSError, ValueError) as error:
        parser.error(str(error))
    if args.output is None:
        sys.stdout.write(text)
    return 0


def _run_audit(args: argparse.Namespace) -> str:
    input_paths = {args.store.resolve()}
    if args.transcripts is not None:
        input_paths.add(args.transcripts.resolve())
    if args.output is not None and args.output.resolve() in input_paths:
        raise ValueError("audit output must not overwrite input files")

    store = load_store(args.store)
    transcripts = load_transcripts(args.transcripts) if args.transcripts is not None else None
    checker = CHECKER_FACTORIES[args.checker]()
    result = checker.check(store, transcripts=transcripts)
    return result.to_json(args.output)


def _run_mutation(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    manifest_output = args.manifest.resolve()
    input_paths = {args.store.resolve()}
    if args.transcripts is not None:
        input_paths.add(args.transcripts.resolve())
    if output == manifest_output:
        raise ValueError("--output and --manifest must be different paths")
    if output in input_paths or manifest_output in input_paths:
        raise ValueError("mutation outputs must not overwrite input files")

    store = load_store(args.store)
    transcripts = load_transcripts(args.transcripts) if args.transcripts is not None else None
    request = MutationRequest(
        defect_class=args.defect,
        subtype=args.subtype,
        seed=args.seed,
        target_memory_id=args.target_id,
        replace_from=args.replace_from,
        replace_to=args.replace_to,
        conflict_relation=args.conflict_relation,
        destination_user_id=args.destination_user_id,
        destination_agent_id=args.destination_agent_id,
        query=args.query,
        distractor_family=args.distractor_family,
        distractor_count=args.distractor_count,
        base_store_status=args.base_store_status,
    )
    result = mutate(store, request, transcripts)
    result.mutated_store.to_json(args.output)
    result.manifest.to_json(args.manifest)


def _build_store(args: argparse.Namespace) -> NormalizedStore:
    if args.adapter == "file":
        if args.source is None:
            raise AdapterDataError("--source is required for the file adapter")
        return FileAdapter(args.source).dump()

    records = _load_external_export(args.source, args.adapter) if args.source else None
    if args.adapter == "mem0":
        filters = {
            key: value
            for key, value in {
                "user_id": args.user_id,
                "agent_id": args.agent_id,
                "run_id": args.session_id,
            }.items()
            if value is not None
        }
        return Mem0Adapter(records=records, filters=filters, page_size=args.page_size).dump()

    if args.adapter == "graphiti":
        scope = MemoryScope(
            user_id=args.user_id,
            agent_id=args.agent_id,
            session_id=args.session_id,
        )
        return GraphitiAdapter(
            records=records,
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            group_ids=args.group_ids,
            scope=scope,
            include_embeddings=args.include_embeddings,
            page_size=args.page_size,
        ).dump()

    return LettaAdapter(
        records=records,
        agent_id=args.agent_id,
        user_id=args.user_id,
        base_url=args.letta_base_url,
    ).dump()


def _load_external_export(path: Path, adapter: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            value = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        else:
            raise AdapterDataError("external fixture/export sources must be JSON or YAML")
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise AdapterDataError(f"could not parse external export {path}: {error}") from error

    if adapter == "letta" and isinstance(value, Mapping):
        records: list[dict[str, Any]] = []
        for key, memory_type in (("blocks", "core"), ("passages", "archival")):
            items = value.get(key, [])
            if not isinstance(items, list):
                raise AdapterDataError(f"Letta export field {key!r} must be a list")
            for item in items:
                if not isinstance(item, Mapping):
                    raise AdapterDataError(f"Letta export field {key!r} contains a non-object")
                record = dict(item)
                record["memory_type"] = memory_type
                records.append(record)
        if records or "blocks" in value or "passages" in value:
            return records

    envelope_keys = {
        "mem0": ("results", "memories"),
        "graphiti": ("edges", "results"),
        "letta": ("records",),
    }
    if isinstance(value, Mapping):
        for key in envelope_keys[adapter]:
            if key in value:
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise AdapterDataError(f"{adapter} export must contain a list of record objects")
    return [dict(item) for item in value]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
