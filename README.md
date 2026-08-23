# MemLint

MemLint is a system-agnostic foundation for auditing agent memory. Its research goal is to study
which memory defects can eventually be detected from an existing store and its source transcripts,
without external annotations or hidden canonical user state.

This repository currently implements the frozen Foundation, the Part 2 taxonomy and mutation
harness, four structural checkers, and the first semantic checker prototype.

Implemented:

- a versioned normalized memory schema;
- Mem0, Graphiti, Letta, and explicit file adapters;
- a minimal transcript representation;
- deterministic JSON serialization;
- `memlint dump`;
- taxonomy version 1.0 with eight frozen defect classes;
- deterministic controlled mutations and separate gold manifests;
- a typed, deterministic checker result API;
- a deterministic semantic evidence resolver;
- deterministic semantic evidence composition;
- a provider-independent `SemanticJudge` contract;
- an optional local CPU NLI `SemanticJudge` implementation;
- the `orphaned_provenance` checker;
- the exact-duplicate `redundancy_bloat` checker;
- the explicit-supersession `stale_active` checker;
- the policy-directed exact-replica `privacy_scope_violation` checker;
- the dependency-injected `unsupported_claim` checker with optional local CPU audit integration;
- retrieval runtime, sufficiency, and result-projection contracts implemented; runtime CLI
  orchestration remains pending;
- `memlint dump`, `memlint mutate`, and `memlint audit`.

Not implemented yet:

- the `internal_contradiction` checker or semantic duplicate detection;
- the instruction-related checker;
- the retrieval or retrieval-shadowing checker;
- any API-hosted or generative LLM judge;
- paper benchmark evaluation, including HaluMem, LongMemEval, or LoCoMo;
- embedding generation, automatic repair, or benchmark scoring.

## Architecture

```text
Mem0 ──────┐
Graphiti ──┤
Letta ─────┼──► Adapter ─► NormalizedStore
Files ─────┘
```

Part 2 branches a clean/base normalized store into two separate artifacts:

```text
NormalizedStore ─► controlled mutation ─┬─► mutated NormalizedStore
                                       └─► gold mutation manifest
```

Checker code may receive the mutated store but must not receive the gold manifest. Implemented
checkers read only normalized data; transcript input is checker-specific.

All future generic code receives only `NormalizedStore`. Backend SDK imports stay inside adapter
modules. Source-native fields are retained under `raw` for reproduction, but a test rejects `.raw`
attribute reads from generic/core modules. `NormalizedMemory.semantic_dict()` makes comparisons that
deliberately omit `raw` straightforward.

## Install

The file adapter needs only the core dependencies:

```bash
python -m pip install -e .
```

Backend SDKs are optional:

```bash
python -m pip install -e '.[mem0]'
python -m pip install -e '.[graphiti]'
python -m pip install -e '.[letta]'
```

The local CPU NLI judge is also optional and does not affect a core install:

```bash
python -m pip install -e '.[semantic-local]'
```

Development checks:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
mypy src/memlint
```

## Normalized schema

Every dump has this envelope:

```json
{
  "schema_version": "0.1",
  "adapter": "file",
  "exported_at": null,
  "memories": []
}
```

`exported_at` is opt-in data, not an automatic clock read. It remains `null` in adapter dumps so the
same source produces byte-for-byte stable output.

Each memory has these fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique within the normalized store. Source IDs are preserved. |
| `content` | string | Source text, without summarization or semantic rewriting. |
| `created_at` | aware ISO-8601 or `null` | Source creation/ingestion time when exposed. |
| `updated_at` | aware ISO-8601 or `null` | Source update time when exposed. |
| `source_refs` | list | Transcript ID plus optional turn index and character span. |
| `provenance_status` | enum | `declared`, `known_absent`, or `unavailable`. |
| `scope` | object | Separate nullable `user_id`, `agent_id`, and `session_id`. |
| `active` | boolean or `null` | Adapter-specific current/retrievable mapping; `null` means unknown. |
| `supersedes` | list[string] | Explicit source-provided replacement IDs only; never inferred. |
| `embedding` | list[number] or `null` | Existing source vector only. MemLint never generates one. |
| `raw` | object | JSON-safe backend-native data for debugging/reproduction. |

### Foundation decisions

**Missing timestamps.** Both timestamp fields are nullable. MemLint never fills them with the current
time. Provided timestamps must include a UTC offset (`Z` is accepted); ambiguous naive datetimes are
rejected.

**Missing provenance.** An empty `source_refs` does not carry enough meaning by itself:

- `declared`: the source, backend, or explicit adapter mapping supplied at least one reference;
- `known_absent`: the source explicitly supplied an empty provenance collection;
- `unavailable`: the adapter/backend did not expose transcript provenance.

`declared` does not mean that an adapter checked a reference against a transcript. Mutation
preconditions may resolve a reference when a controlled mutation specifically requires it. Graphiti
episode IDs remain in `raw` and are not treated as transcript IDs unless the caller provides an
explicit episode-to-transcript mapping.

**Meaning of `active`.** This field is deliberately nullable and its mapping is documented per adapter
below. It means "current/retrievable according to this source path," not a universal truth predicate.

**Stable IDs.** A source ID is used unchanged. If absent, MemLint hashes canonical JSON containing the
adapter name, exact content, canonical UTC creation timestamp, scope, and canonical source-reference
set, producing `<adapter>:<24 hex characters>`. Source-reference ordering does not change the ID.
Identical anonymous records intentionally collide; the store rejects the duplicate and asks the
producer to supply real identities instead of inventing unstable random IDs.

**Embeddings.** Existing vectors are copied when exposed. Missing vectors stay `null`; no paid or local
embedding service is called.

## Adapter mappings

| Backend | ID/content | Timestamps | Scope | Provenance | `active` | Backend-only data |
|---|---|---|---|---|---|---|
| File | `id`, `content` | Explicit fields | `scope` or top-level scope keys | Explicit `source_refs` | Explicit value, otherwise `null` | Unknown metadata and supplied `raw` |
| Mem0 | `id`, `memory` | `created_at`, `updated_at` | Record `user_id`, `agent_id`, `run_id/session_id`; missing values fall back to query scope | Current documented memory response has no first-class transcript refs, so `unavailable` | `null` unless an export explicitly contains `active` | Full response including hash/metadata |
| Graphiti | EntityEdge `uuid`, `fact` | `created_at` is ingestion time; validity timestamps stay in `raw` | Explicit adapter scope; `group_id` is not guessed to be a user/session | Episode IDs stay in `raw`; transcript refs require an explicit mapping | true iff exposed `invalid_at` and `expired_at` are both null; `null` when those fields are absent | Relation, nodes, group, episodes, bi-temporal fields, attributes |
| Letta | Block `id`/`value`; Passage `id`/`text` | Passage times copied; current Block schema exposes none | Adapter agent/user context | `unavailable` unless an export explicitly supplies refs | attached core blocks are active; non-deleted archival passages are active; deleted passages are inactive; explicit value wins | `memory_type`, labels, archive/file metadata, mutability metadata |

Mem0 transport follows the documented paginated `MemoryClient.get_all(...)` response. Graphiti reads
`EntityEdge.get_by_group_ids(...)` through a Neo4j driver and does not instantiate its LLM ingestion
client. Letta uses the documented agent block and archival-passage list endpoints. Links to the source
contracts: [Mem0 export](https://docs.mem0.ai/cookbooks/essentials/exporting-memories),
[Graphiti EntityEdge source](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py), and
[Letta Python agent resources](https://docs.letta.com/api/python/resources/agents).

## File formats

JSON and YAML accept one record, a top-level list, or `{"memories": [...]}`. JSONL accepts exactly one
memory object per nonblank line.

Markdown intentionally supports exactly one memory per file and requires YAML front matter:

```markdown
---
id: m1
created_at: "2026-08-10T14:20:00+02:00"
user_id: user-1
active: true
---
User prefers Python.
```

The body is `content`; putting `content` in front matter is rejected. Arbitrary prose Markdown is not
guessed to be a structured memory store.

## Usage

Credential-free reference example:

```bash
memlint dump --adapter file --source examples/store.yaml
memlint dump --adapter file --source examples/store.yaml --output normalized.json
```

Controlled mutation example:

```bash
memlint mutate \
  --store examples/mutation-store.json \
  --transcripts examples/mutation-transcripts.json \
  --defect unsupported_claim \
  --seed 42 \
  --target-id preference-python \
  --replace-from Python \
  --replace-to Rust \
  --output mutated.json \
  --manifest mutation.json
```

The output paths must differ. `mutated.json` contains only normalized store data; `mutation.json`
contains the injected gold positive, deterministic IDs/digests, and reproduction parameters. A
manifest defaults to `base_store_status: unknown`: records outside the injected positive are not
automatically verified negatives. Use `curated_clean` only for an independently curated clean input.

Retrieval-shadowing mutations create distractor-crowding challenges and a retrieval probe. They set
`requires_runtime_validation: true`; adding distractors alone is not evidence that shadowing occurred.
See [the frozen taxonomy](docs/taxonomy.md) and [mutation harness](docs/mutations.md) for precise
boundaries and supported subtypes.

Run the structural provenance audit separately from mutation gold data:

```bash
memlint audit \
  --store mutated.json \
  --transcripts examples/mutation-transcripts.json \
  --checker orphaned_provenance \
  --output findings.json
```

The audit command has no manifest input. See [checker results](docs/checkers.md) for the result and
evidence contracts.

Scope audits require an explicit authoritative-principal policy:

```bash
memlint audit \
  --store scoped.json \
  --checker privacy_scope_violation \
  --scope-policy examples/scope-policy.json \
  --output findings.json
```

This checker identifies policy-prohibited exact portable replicas. It does not reconstruct
historical copy direction or provide general privacy-compliance analysis.

Unsupported-claim semantic auditing is optional. Install `memlint[semantic-local]`, then provide an
explicit local model and revision:

```bash
memlint audit \
  --store normalized.json \
  --transcripts transcripts.json \
  --checker unsupported_claim \
  --semantic-model-id cross-encoder/nli-MiniLM2-L6-H768 \
  --semantic-model-revision b95119ce93d3e065de6214e38cd4a97b0f2f2c6d \
  --output findings.json
```

The [semantic groundwork](docs/semantics.md) resolves and composes declared transcript coordinates
and defines a provider-independent judgment contract. The dependency-injected `unsupported_claim`
prototype uses that contract, while an optional pinned local CPU NLI implementation is available for
explicit development use. The current MiniLM plus PLAIN configuration is not a final research winner,
and semantic performance has not been established. There is no score threshold, and zero findings
does not imply that every memory was assessed; inspect the checker skip counters. Core installs and
structural audits do not include, construct, or download the local semantic model.

A development-only bidirectional pair-policy probe exists for a future `internal_contradiction`
checker; it has not frozen a primary policy, and no such checker is implemented.

Committed source-shaped exports can exercise every adapter offline:

```bash
memlint dump --adapter mem0 --source tests/fixtures/mem0.json
memlint dump --adapter graphiti --source tests/fixtures/graphiti.json --user-id user-123
memlint dump --adapter letta --source tests/fixtures/letta.json --agent-id agent-1
```

Documented live command shapes are:

```bash
MEM0_API_KEY=... memlint dump --adapter mem0 --user-id user-123

export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...

memlint dump --adapter graphiti \
  --group-id user-123-group

LETTA_API_KEY=... memlint dump --adapter letta --agent-id agent-...
# Or for self-hosted Letta:
memlint dump --adapter letta --letta-base-url http://localhost:8283 --agent-id agent-...
```

Secrets can be supplied by the named environment variables; avoid storing credentials in shell
history. These live paths require the corresponding optional extra.

## Transcript representation

`TranscriptSet` contains uniquely identified `Transcript` objects. Each transcript has ordered,
uniquely indexed `TranscriptTurn` values with role, exact content, optional aware timestamp, and JSON
metadata. [examples/transcripts.json](examples/transcripts.json) demonstrates how
`examples/store.yaml` points to turn `0` and character span `[0, 20]`.

No matching, entailment, inference, or source reconstruction is performed by the Foundation or
mutation harness.

## Known limitations

- Live external accounts/services were not used in the Foundation test suite.
- Live Mem0, Graphiti, and Letta integrations require user-managed services, credentials, and optional
  SDKs; local tests use source-shaped fixtures and selected fake transports, not real accounts.
- Graphiti `group_id` is kept in `raw` because its meaning is deployment-specific. Pass explicit scope
  values when known.
- Graphiti episode IDs are source episode identifiers, not MemLint transcript IDs. Programmatic users
  may pass `episode_transcript_map` only when they have an explicit mapping.
- Current Letta core block responses do not expose creation/update timestamps.
- Letta `active` describes current attachment/listing state along that adapter source path; it is not a
  backend-independent truth state.
- Backend metadata cannot be promoted into portable fields unless the mapping is explicit and stable.
- Normalized schema version `0.1` contains no findings, gold labels, checker results, or repair data.
- Mutation challenges create controlled research inputs but do not report detector or retrieval
  performance.
