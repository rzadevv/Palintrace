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
used; the structural checkers report zero model and token use and make no pricing assumptions.

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

## Unsupported claim

`unsupported_claim` is the first semantic checker prototype. It requires a `TranscriptSet` and an
injected `SemanticJudge`; the checker itself does not instantiate or configure a model. The audit CLI
constructs the existing local CPU judge only when this checker is selected. It assesses only memories
whose provenance status is `declared`. The premise is the composed declared transcript evidence, and
the hypothesis is the exact `memory.content`. PLAIN is the current primary development composition
style.

Assessability is deliberately narrower than non-entailment. If any declared source reference has a
`missing_transcript`, `missing_turn`, or `invalid_span` issue, the checker abstains for that memory,
even when another reference resolves. It also abstains when an existing declared source produces no
segments, when the composed premise is whitespace-only, or when the complete premise/hypothesis pair
exceeds the judge input limit. Evidence is never truncated, chunked, or summarized. Memories with
`unavailable` or `known_absent` provenance are skipped because missing provenance does not establish
an unsupported claim. Judge failures other than the typed over-limit condition fail the checker
instead of becoming clean or defective results.

For an assessed memory, the frozen three-class relation maps as follows:

- `entailment`: emit no Finding;
- `neutral`: emit one unsupported-claim Finding;
- `contradiction`: emit one unsupported-claim Finding.

There is no score threshold. `Finding.confidence` is the selected NLI class's softmax score without
transformation; it is judge-specific and is not a calibrated probability that the memory is
defective. Each unsupported memory receives one memory-level Finding. Evidence records the selected
relation, captured judge identity and version, composition style, unique-segment count, SHA-256
digests, and canonical unique source coordinates. Repeated declarations removed by composition do
not change semantic Finding identity. Evidence does not copy premise, hypothesis, transcript, or
backend-native text. The floating score is excluded from evidence identity, so score-only jitter
does not change a Finding ID when the selected relation is unchanged.

Semantic usage returned by every successful judgment is summed into `CheckerCost`; skipped memories
contribute no model calls or tokens. Checker details report declared and assessed memories, separate
skip counts for non-declared provenance, structural issues, absent evidence, and oversized input, and
counts for each semantic relation. These deterministic counters make abstention visible. In
particular, zero unsupported-claim findings does not mean that every memory was assessed or supported.

Install the optional local semantic dependencies before using the CLI integration:

```bash
python -m pip install -e '.[semantic-local]'
```

Then provide both the model ID and an explicit, nonblank revision:

```bash
memlint audit \
  --store normalized.json \
  --transcripts transcripts.json \
  --checker unsupported_claim \
  --semantic-model-id cross-encoder/nli-MiniLM2-L6-H768 \
  --semantic-model-revision b95119ce93d3e065de6214e38cd4a97b0f2f2c6d \
  --output findings.json
```

The CLI uses the local classifier on CPU and has no hidden model revision, score threshold, or
composition override. The shown MiniLM revision with PLAIN composition is a development
configuration, not a final research winner, and semantic defect-detection performance has not been
established. Zero findings does not imply full assessment: inspect the skip counters in
`CheckerStats.details`. Structural audits neither construct nor download a semantic model.

## Identity-grounded unsupported-claim candidate

`IdentityGroundedUnsupportedClaimChecker` version `0.1` is a DEVELOPMENT candidate that is separate
from the frozen `UnsupportedClaimChecker` version `1.0`. It is not a replacement, is not exported
from the public checker package, and is not available through the CLI, benchmark v0.1 dispatch, or
default checker lists. Part 6G-C supports its frozen representation on an independently constructed
synthetic held-out set, but this is not real-world deployment validation and does not establish that
production integrations can supply trustworthy speaker bindings.

The constructor requires explicit version `0.1` `SpeakerIdentityBindings` in addition to an injected
`SemanticJudge`. For each declared memory, the candidate first resolves transcript evidence and
requires complete, nonempty evidence. It composes that evidence with fixed PLAIN composition, then
resolves the memory's exact `(transcript_id, turn_idx)` source coordinates against the explicit
bindings. Only `RESOLVED` identity is assessed. The premise is built by the frozen helper in exactly
this form:

```text
The speaker is {speaker_label}.
{plain_composed_evidence}
```

`UNAVAILABLE` identity and `CONFLICT` identity both abstain without a semantic call or Finding.
There is no PLAIN fallback. This freezes a reduced-coverage policy because ungrounded PLAIN evidence
is the representation implicated by the v0.1 clean-selectivity failure. Identity abstentions remain
visible as `skipped_identity_unavailable` and `skipped_identity_conflict` in
`CheckerStats.details`; they are capability limitations, not unsupported-claim defects.

For assessed memories, the relation policy is unchanged: entailment emits no Finding, while neutral
and contradiction each emit one unsupported-claim Finding. Candidate evidence records the fixed
PLAIN and explicit-binding method, canonical source coordinates and counts, and premise/hypothesis
SHA-256 digests. It does not serialize the speaker label, premise, hypothesis, transcript text,
memory text, or raw bindings.

Speaker-grounded checking trades semantic selectivity for an explicit capability requirement.
Memories without trustworthy bindings for every relevant source turn are not assessed. Evaluations
must report both semantic outcomes on assessed memories and identity-grounding coverage and
abstention counts. Abstentions must not be hidden or interpreted as clean memories. H3
confidence/selectivity policy and H4 retrieval design remain untested.

Part 6G-D finds no current adapter that automatically supplies both exact turn attribution and a
human-readable label. Optional use therefore still requires explicit caller/operator assertions
admitted through the trust contract described in
[`speaker_identity_integrations.md`](speaker_identity_integrations.md). Mem0 scope IDs, Letta
agent/user context, Graphiti groups/episodes, file metadata, and transcript roles do not become
speaker labels automatically.

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

## Stale active

`stale_active` implements only the controlled `explicit_supersession` case. It emits a Finding for
an old memory when another memory directly names its ID in `supersedes` and the old memory has
`active=true`. An inactive old memory is already resolved, while `active=null` does not provide
enough evidence that it remains current.

Missing supersession targets and self-links do not create findings. One old memory receives one
Finding even when several memories supersede it; each direct superseder contributes a separate
`active_superseded` evidence item. Direct links in a chain are evaluated independently, without
computing transitive supersession.

The checker does not infer replacement from timestamps, similar content, changed values, scope, or
other metadata. It requires no transcripts and uses no model calls or tokens. Its details report
supersession links scanned, resolved links, missing targets skipped, and self-links skipped.

```bash
memlint audit \
  --store stale.json \
  --checker stale_active \
  --output findings.json
```

## Privacy / scope violation

`privacy_scope_violation` implements policy-prohibited exact portable replicas relative to an
explicitly declared authoritative principal. A version `0.1` `ScopeIsolationPolicy` is required and
supports only `user_id` and `agent_id`. Each rule names an `authoritative_source_principal` and one
or more prohibited destination principals; there is no default assumption that all principals are
mutually isolated. Policies contain principal-boundary configuration only, with no memory, mutation,
or gold identifiers.

For a `user_id` rule, replica identity excludes only the memory ID, `raw`, and `user_id`. For an
`agent_id` rule, it excludes only the memory ID, `raw`, and `agent_id`. Every other portable field
must match exactly. The checker does not normalize text, compare meaning, or inspect backend-native
data. Unknown principal values cannot satisfy a rule.

Historical copy direction is not recoverable from two otherwise identical records without
independent lineage evidence. The policy therefore supplies authoritative ownership for this audit;
the checker does not reconstruct a copy event. For the same A/B store, an A-authoritative policy
targets the prohibited B record, while a B-authoritative policy targets the prohibited A record.
The destination memory alone receives the Finding, and authoritative records appear only in
`prohibited_exact_replica` evidence.

Confidence `1.0` means that the explicit policy and exact structural replica rule were satisfied. It
is not a probability that a historical copy occurred. Independently created records could in
principle have identical portable fields, so this is deterministic evidence of a policy-prohibited
exact replica, not general privacy compliance or forensic proof of copying. The checker requires no
transcripts and uses no model calls or tokens.

```bash
memlint audit \
  --store scoped.json \
  --checker privacy_scope_violation \
  --scope-policy examples/scope-policy.json \
  --output findings.json
```
