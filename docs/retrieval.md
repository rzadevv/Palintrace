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

An insufficient assessment is not yet emitted as a `retrieval_shadowing` `Finding`. No
`RetrievalShadowingChecker` exists. Part 5C must first resolve two representation issues without
silently changing frozen schemas:

- retrieval-shadowing mutation gold uses the `RETRIEVAL_CASE` unit, while the generic `Finding`
  requires one or more `memory_ids`; Part 5B does not decide that mapping or modify `Finding`; and
- `CheckerCost` records model calls and token counts, while `RetrievalUsage` records retrieval calls
  and candidate counts; Part 5B does not modify `CheckerCost` or decide whether later retrieval usage
  belongs in `CheckerStats.details` or another explicitly versioned representation.

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
