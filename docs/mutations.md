# Mutation harness

MemLint mutations create controlled defects; they do not detect them. The public entry point is:

```python
result = mutate(store, request, transcripts=None)
```

Mutation manifest schema 1.1 adds explicit gold-label units and structural invariants. It remains
separate from the frozen taxonomy version 1.0.

`result.mutated_store` is the detector-visible normalized store. `result.manifest` is separate gold
data for later evaluation and must never be passed to a detector. Mutation labels, mutation IDs, and
manifest data are never inserted into normalized memories, `raw`, transcript metadata, or store
fields.

## Reproducibility

A mutation is determined by the normalized base store, optional transcript set, complete request,
seed, and taxonomy version. Selection uses a local seeded random generator. IDs use canonical JSON
and SHA-256; the harness does not use the current time, random UUIDs, global random state, or Python's
process-dependent `hash()`. Generated memory IDs use the opaque form `mem-<24 hex characters>`; role
and mutation data participate only in the hash input and are not visible in the ID.

Store digests cover the portable normalized semantics in memory-ID order. They omit backend-specific
`raw` and the export envelope's `exported_at`; mutations preserve both without inspecting native
metadata. Transcript inputs receive their own canonical digest. Manifest JSON is key-sorted and has no
execution timestamp.

Mutation functions return newly validated Pydantic objects. The input store is frozen and unchanged.

## Gold scope

`base_store_status` defaults to `unknown`. An injected defect is a known positive, but records outside
the mutation are not automatically verified negatives. Use `curated_clean` only when the caller has
independently established that the fixture belongs to a curated clean dataset.

The manifest includes target, created, modified, and removed memory IDs; controlled parameters;
portable store digests; and the base-store status. Target roles describe context without assigning
independent record-level positives.

Each manifest has one explicit gold label unit:

- `memory` for unsupported, stale-active, orphaned-provenance, injected-instruction, and scope cases;
- `memory_pair` for contradiction and redundancy, with both IDs kept together as one relation;
- `retrieval_case` for shadowing challenges.

A retrieval challenge has `observed_positive: false` until a real retrieval run establishes the
behavioral defect. All other controlled mutations represent an injected positive directly.

## Supported mutations

| Defect | Subtype | Transcript input | Runtime validation |
|---|---|---:|---:|
| `unsupported_claim` | `factual_substitution` | required | no |
| `internal_contradiction` | `controlled_conflict` plus `exclusive_value` contract | no | no |
| `stale_active` | `explicit_supersession` | no | no |
| `orphaned_provenance` | `missing_transcript`, `missing_turn`, `invalid_span` | required | no |
| `retrieval_shadowing` | `distractor_crowding`, `editor` family | no | yes |
| `injected_instruction` | `fixed_response`, `format_override` | no | no |
| `privacy_scope_violation` | `cross_user_copy`, `cross_agent_copy` | no | no |
| `redundancy_bloat` | `exact_duplicate` | no | no |

Retrieval challenge manifests contain a `RetrievalProbe` with a query, expected target IDs, and
distractor IDs. They deliberately contain no retrieved IDs, rank, pass/fail result, or claim that
shadowing occurred. A later configured retrieval experiment must supply that observation. The caller
must select the fixed `editor` distractor family explicitly; the harness does not infer a target's
topic or present these templates as generic distractors.

## Controlled contradiction contract

`controlled_conflict` produces valid contradiction gold only when the request explicitly declares
`conflict_relation: exclusive_value`. The fixture is asserting that the replaced slot accepts one
current value. For example, one current preferred editor cannot simultaneously be both Neovim and VS
Code under that fixture contract.

MemLint does not claim that arbitrary substitutions conflict. "User knows Python" and "User knows
Rust" may both be true, so a programming-language substitution without an exclusive-value contract
is rejected.

## Mutation artifact control

Synthetic benchmarks must not reward detectors for recognizing the harness instead of the defect:

- generated memory and broken-reference IDs are opaque and contain no mutation role or class names;
- content-changing substitutions require targets without stored embeddings, because Part 2 does not
  regenerate embeddings; derived changed-content records also carry no copied embedding;
- mutation metadata and gold semantics never enter detector-visible records or `raw`;
- relational gold is scored as one memory-pair relation, not two independent memory labels;
- fixed templates define controlled challenges and must not be treated as evidence that a method
  generalizes beyond those templates.

## CLI

The mutated store and gold manifest require different output paths. Outputs also cannot overwrite the
input store or transcripts.

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

Class-specific preconditions fail explicitly. The harness does not guess replacements, destination
principals, semantic conflict relations, distractor families, queries, or missing source evidence.

For contradiction, add the explicit fixture contract:

```bash
--conflict-relation exclusive_value
```

For the current retrieval challenge family, add:

```bash
--distractor-family editor
```
