# Retrieval runtime contract

Retrieval shadowing is an operational defect: a valid, relevant memory exists in the store, but the
configured runtime retriever does not return it sufficiently for a relevant query. Static store
inspection cannot establish that behavior. A returned ID list alone is also insufficient because it
does not say which absent store records should have been retrieved.

Part 5A therefore freezes two logically separate inputs:

1. a `RetrievalAuditRequest` declaring what behavior is being audited; and
2. a `RetrievalResponse` reporting what the real retriever returned.

`run_retrieval_audit` joins those values into a deterministic `RetrievalObservation`. It validates
the evidence boundary but does not decide whether retrieval shadowing occurred.

## Declared audit target

An audit request contains a stable caller-supplied `request_id`, the exact `query`, a nonempty set of
`expected_memory_ids`, and a strict positive `top_k`. Expected IDs are unique and canonicalized into
sorted tuple order. Every expected ID must resolve in the supplied `NormalizedStore` before the
retriever runs.

`expected_memory_ids` is a declared audit target specification. It means that, for this request,
the caller identifies those visible store records as relevance targets. It is not hidden ground
truth and is not itself a defect label. Shadowing would still require a runtime retrieval
observation and a separately defined sufficiency policy. Retrieval auditing is consequently not
fully assumption-free: the relevance targets must come from an explicit audit request. Part 5A does
not infer them from memory content, lexical similarity, embeddings, NLI, or an LLM.

A missing expected ID is invalid audit configuration, not retrieval shadowing. The retriever is not
called for an invalid request.

## Target-blind retrieval

The provider-independent `Retriever` protocol declares a nonblank `retriever_id` and
`retriever_version`, which are captured as operational provenance. Its runtime method receives only:

```python
retrieve(*, query: str, top_k: int) -> RetrievalResponse
```

The retriever never receives the request ID, expected IDs, mutation metadata, gold labels, or
distractor IDs. This target blindness prevents evaluation targets from contaminating operational
retrieval.

Each `RetrievalHit` contains only `memory_id`, one-based positive `rank`, and an optional finite
`score`. Rank is the authoritative generic ordering. A score is retained only when the retriever
actually exposes one, and the generic contract does not assume that higher scores are better. Hits
are canonicalized by rank; ranks and memory IDs must each be unique.

A `RetrievalResponse` contains the hits plus `RetrievalUsage`. Usage reports strict nonnegative
`retrieval_calls` and `candidate_count` values supplied by the implementation. It contains no audit
targets, pass/fail result, finding, or shadowing flag. A retriever may return fewer than `top_k`
hits, including an empty result, but it may not return more. Every returned ID must resolve against
the same audited store snapshot. An unknown ID is an invalid runtime observation that cannot be
reconciled with the snapshot; it is not silently dropped or classified as shadowing.

## Observation and privacy

`RetrievalObservation` records the request ID, exact-query SHA-256, expected IDs, `top_k`, retriever
identity, actual hits, and usage. It does not retain the full query by default. The digest is computed
over the exact UTF-8 query bytes without normalization or rewriting. Observation JSON uses sorted
keys, rejects nonfinite JSON numbers, and contains no time, random UUID, or process-local hash, so
identical inputs and runtime responses serialize byte-for-byte identically.

The observation deliberately has no `passed`, `failed`, `shadowed`, defect class, or `Finding`.
Part 5A does not choose whether a target must rank first, appear anywhere in the top-k result, be
returned alongside all other targets, or satisfy a recall threshold. Those are later methodology
decisions. No `RetrievalShadowingChecker`, retrieval audit CLI checker, or concrete backend retriever
exists yet.

## Retrieval sufficiency policies

Part 5B evaluates an already-valid `RetrievalObservation` under one caller-selected
`RetrievalSufficiencyPolicy`. Exactly two policies are supported:

- `ALL_EXPECTED` (`all_expected`) is sufficient only when every expected memory ID appears
  somewhere in the observed hits; and
- `ANY_EXPECTED` (`any_expected`) is sufficient when at least one expected memory ID appears
  somewhere in the observed hits.

Neither policy is a universal or implicit default. Every assessment call must explicitly supply the
policy because a multi-target audit may mean that all targets are required or that any target is an
adequate answer. For a single expected target, the two policies necessarily produce the same
decision: present is sufficient and absent is insufficient.

`RetrievalSufficiencyAssessment` records the request ID, explicit policy, strict sufficiency boolean,
the canonical expected/retrieved/missing target partition, and `top_k`. An expected target counts if
it appears anywhere in the valid observed hit window; rank one is not required. Hit scores and
`RetrievalUsage` do not affect the target partition or decision and are not copied into the
assessment. Returned non-target memories are allowed and are not themselves classified as defects.
The assessment needs neither query text nor store or retriever access, and its deterministic JSON is
derived only from the recorded observation and explicit policy.

The sufficiency assessment itself is not a `Finding`. Part 5C's projection below maps an insufficient
case into the frozen result envelope without changing the Part 5B policies or generic checker
schemas.

## Retrieval-shadowing result projection

`project_retrieval_shadowing_result` consumes one recorded observation plus one explicit policy and
recomputes the sufficiency assessment internally. One runtime retrieval case remains the scientific
unit. A sufficient case emits no findings; an insufficient case emits exactly one
`retrieval_shadowing` `Finding`, regardless of how many expected targets are missing.

The frozen generic `Finding` requires nonempty `memory_ids`, so the one finding uses only
`missing_expected_memory_ids` as normalized-memory anchors. Successfully retrieved targets are not
anchors. This does not change the taxonomy gold unit from `RETRIEVAL_CASE` to `MEMORY` and does not
claim that each missing record independently has a retrieval defect. The defect is the case-level
combination of request identity, query digest, retriever identity and version, top-k window, explicit
policy, and target retrieval outcome. The complete expected/retrieved/missing partition remains in
the single evidence item.

Finding confidence is `1.0` because the recorded observation deterministically fails the
caller-declared policy. It does not mean that the caller's declared relevance targets are
objectively correct with 100 percent probability. Hit scores, ranks, retrieval usage, and ordinary
non-target result IDs are excluded from finding evidence and identity because they do not affect the
frozen presence-based policies. Changes to target membership or operational identity do affect the
finding ID.

Retrieval work is diagnostic accounting rather than model-token cost. `CheckerCost` remains all
zeros, while `retrieval_calls`, `candidate_count`, target counts, hit count, and the one assessed case
are recorded in `CheckerStats.details`. `memories_scanned` remains zero because projection reads no
memory contents.

This function is an operational result projection, not an implementation of the generic `Checker`
protocol. It accepts no store, transcripts, retriever, external assessment, or query text and never
executes retrieval. No `RetrievalShadowingChecker` class or retrieval CLI integration exists yet.
Persisted observations rely on Part 5A's execution-time reconciliation with the audited store
snapshot; they are not yet cryptographically bound to the complete store contents. That is a known
reproducibility limitation rather than a hidden redesign of the frozen observation schema.

## Recorded retrieval CLI

Project a previously recorded observation into the frozen checker result envelope with an explicit
policy:

```bash
memlint retrieval-audit \
  --observation retrieval-observation.json \
  --policy all_expected \
  --output findings.json
```

The input must be UTF-8 JSON conforming to the frozen `RetrievalObservation` schema. This command
does not execute retrieval, contact a backend, accept a store, or reconstruct the absent query text.
It accepts no query, expected-target, top-k, retriever-identity, usage, mutation, manifest, or model
overrides. The policy is required on every invocation and must be `all_expected` or `any_expected`;
there is no default.

Output is deterministic `CheckerResult` schema `0.2` JSON. A sufficient recorded case contains zero
findings, while an insufficient case contains the one case-level finding frozen in Part 5C. Omitting
`--output` writes only that JSON to stdout. An output path cannot overwrite the recorded observation.

Static `memlint audit --checker ...` still does not expose `retrieval_shadowing`, and no concrete
backend retriever or `RetrievalShadowingChecker` class exists. The CLI projects recorded evidence;
it does not resolve the known limitation that persisted observations are not cryptographically bound
to complete audited-store contents.

## Paired retrieval challenges

A mutated-store miss alone cannot support a causal claim that a distractor-crowding mutation induced
retrieval shadowing: the configured retriever may already have missed the target on the base store.
Controlled mutation-effect experiments therefore assess a recorded baseline observation together
with a recorded mutated observation under one explicit `ALL_EXPECTED` or `ANY_EXPECTED` policy. A
baseline-sufficient run is an eligibility condition for an induced-shadowing result.

`assess_paired_retrieval_challenge` requires the two observations to have exactly the same query
SHA-256, canonical expected target set, `top_k`, retriever ID, and retriever version. Their request
IDs may differ because they are separate runtime records; a nonblank caller-supplied `case_id` binds
them into one experimental challenge. The function reuses the frozen Part 5B sufficiency assessment
for both runs and never executes retrieval or reads a store.

Exactly three outcomes exist:

- `induced_shadowing`: the baseline is sufficient and the mutated observation is insufficient;
- `resilient`: both observations are sufficient; and
- `baseline_insufficient`: the baseline is insufficient, regardless of the mutated result.

Baseline-insufficient cases are excluded from mutation-effect success counts. A mutated success
after a baseline failure remains `baseline_insufficient`; Part 5E deliberately does not add an
"improved" outcome. Scores, retrieval usage, ordinary non-target hits, and a target's rank within
the valid top-k window do not affect these presence-based outcomes.

Part 2 distractor mutations define retrieval challenges, but their probe, manifest, mutation ID,
distractor IDs, and gold label remain evaluation-only metadata. Test or future evaluation harness
code may translate the public query and expected IDs into independent baseline and mutated audit
requests, but production paired assessment consumes only the two recorded observations. There is no
concrete retriever and no paired-challenge CLI command in this phase.

The compatibility checks do not independently prove that the observations came from the intended
base and corresponding mutated store snapshots. Part 5A observations are not yet cryptographically
bound to complete store contents; an outer controlled-evaluation harness may check mutation and store
digests later without changing this frozen paired contract.

## Part 2 isolation

Part 2 retrieval challenges are evaluation artifacts. Their mutation `RetrievalProbe` declares a
query, expected target IDs, and distractor IDs, but contains no actual retrieval observation. The
mutation alone is therefore not a positive observed retrieval-shadowing defect.

Production retrieval auditing never imports or accepts `RetrievalProbe`, `MutationRequest`,
`MutationManifest`, `GoldLabel`, or other mutation data. Test code may translate only the public
query and expected target IDs into an independent `RetrievalAuditRequest`, adding a test-selected
`top_k` and request ID, to demonstrate representational compatibility. Part 2 objects, distractor
IDs, manifests, mutation IDs, and gold labels are never passed to the production retriever or audit
execution function.
