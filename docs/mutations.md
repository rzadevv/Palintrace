# Mutation harness

MemLint mutations create controlled defects; they do not detect them. The public entry point is:

```python
result = mutate(store, request, transcripts=None)
```

`result.mutated_store` is the detector-visible normalized store. `result.manifest` is separate gold
data for later evaluation and must never be passed to a detector. Mutation labels, mutation IDs, and
manifest data are never inserted into normalized memories, `raw`, transcript metadata, or store
fields.

## Reproducibility

A mutation is determined by the normalized base store, optional transcript set, complete request,
seed, and taxonomy version. Selection uses a local seeded random generator. IDs use canonical JSON
and SHA-256; the harness does not use the current time, random UUIDs, global random state, or Python's
process-dependent `hash()`.

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
portable store digests; and the base-store status. Target details distinguish records receiving the
gold label from contextual source or superseding records.

## Supported mutations

| Defect | Subtype | Transcript input | Runtime validation |
|---|---|---:|---:|
| `unsupported_claim` | `factual_substitution` | required | no |
| `internal_contradiction` | `controlled_conflict` | no | no |
| `stale_active` | `explicit_supersession` | no | no |
| `orphaned_provenance` | `missing_transcript`, `missing_turn`, `invalid_span` | required | no |
| `retrieval_shadowing` | `distractor_crowding` | no | yes |
| `injected_instruction` | `fixed_response`, `format_override` | no | no |
| `privacy_scope_violation` | `cross_user_copy`, `cross_agent_copy` | no | no |
| `redundancy_bloat` | `exact_duplicate` | no | no |

Retrieval challenge manifests contain a `RetrievalProbe` with a query, expected target IDs, and
distractor IDs. They deliberately contain no retrieved IDs, rank, pass/fail result, or claim that
shadowing occurred. A later configured retrieval experiment must supply that observation.

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
principals, queries, or missing source evidence.
