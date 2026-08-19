# Checkers

A checker receives portable `NormalizedStore` data and, when required, a `TranscriptSet`. It does not
receive mutation requests, manifests, or labels, and it does not inspect backend-native `raw` fields.
Mutation manifests remain evaluation-only artifacts outside the checker and audit interfaces.

Checker output uses schema version `0.2` and contains:

- `Finding`: an immutable defect class, one or more memory IDs, confidence, and supporting evidence;
- `EvidenceItem`: a stable machine-readable kind, concise message, and minimal JSON data;
- `CheckerCost`: nonnegative model-call and token counters;
- `CheckerStats`: scanned-memory and finding counts plus checker-specific nonnegative details;
- `CheckerResult`: the checker identity, defect class, sorted findings, cost, and statistics.

Finding IDs are derived from canonical semantic inputs with SHA-256. Memory IDs and evidence
identities are sorted before hashing, and an evidence identity contains only its kind and structured
data. Human-readable evidence wording does not affect identity. Identical semantic inputs therefore
produce identical IDs and byte-stable sorted JSON. Results contain no execution timestamp or runtime
duration.

Validated findings canonicalize their visible memory-ID and evidence ordering. Evidence JSON and
checker-specific statistics are recursively immutable in memory while retaining ordinary JSON object
and array representations when serialized.

Generic checker statistics contain `memories_scanned`, `findings_emitted`, and a `details` object.
The orphaned-provenance checker reports its structural work as
`details.source_refs_scanned`.

For a structural finding, confidence `1.0` means that the declared rule was deterministically
satisfied. It is not a statistically calibrated probability. Cost values report calls and tokens
used; the structural checkers report zero for all three and make no pricing assumptions.

## Orphaned provenance

`orphaned_provenance` requires a `TranscriptSet` and resolves source references only for memories
whose provenance status is `declared`. It emits one Finding per defective memory, combining multiple
broken references into separately ordered evidence items.

The supported evidence kinds are:

- `missing_transcript`: the referenced transcript ID is absent;
- `missing_turn`: the transcript exists but the referenced turn index is absent;
- `invalid_span`: the transcript and turn exist but the span end exceeds the turn length.

A declared reference to a whole transcript, a valid whole turn, or a valid character span is not a
finding. `unavailable` provenance means the backend did not expose usable provenance and is not
orphaned. `known_absent` provenance is also not orphaned. The checker performs no semantic comparison
between memory text and transcript text.

If no transcript set is supplied, the checker raises an input error instead of treating every source
reference as broken. Evidence identifies reference coordinates and lengths but does not copy full
transcript content.

Run it with:

```bash
memlint audit \
  --store mutated.json \
  --transcripts examples/mutation-transcripts.json \
  --checker orphaned_provenance \
  --output findings.json
```

Omit `--output` to write deterministic JSON to standard output. The output path must differ from both
input paths. The audit command does not accept a mutation manifest.

## Redundancy / bloat

`redundancy_bloat` implements only the structural `exact_duplicate` case. Two memories form a
duplicate pair when their stored content strings are exactly equal and their normalized scopes are
equal with at least one known scope dimension. A pair whose user, agent, and session dimensions are
all unknown is skipped because missing scope does not establish a shared principal.

The checker does not strip whitespace, change case, normalize Unicode or punctuation, or evaluate
semantic similarity. Similar or paraphrased claims are therefore outside this check. Timestamps,
provenance, active state, supersession, and embeddings do not alter exact claim identity.

Each duplicate memory pair receives a separate Finding with `exact_duplicate` evidence containing an
exact-content SHA-256 digest, content length, and normalized scope. The evidence does not repeat the
memory content. Three matching memories produce three pair findings. The checker requires no
transcripts and uses no model calls or tokens.

```bash
memlint audit \
  --store duplicated.json \
  --checker redundancy_bloat \
  --output findings.json
```
