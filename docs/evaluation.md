# Gold-safe evaluation accounting

Part 6A defines accounting infrastructure for controlled mutation experiments. It does not run a
benchmark, tune a checker, select a model, or report a performance result. Evaluation code consumes
gold mutation metadata only after a checker has produced its independent `CheckerResult`; manifests,
gold labels, base-store status, mutation IDs, and changed-memory lists are never detector inputs.

## Controlled static mutation accounting

`evaluate_mutation_trial` joins one base store, its mutated store, the separate mutation manifest,
the already-produced checker result, and the same optional transcript set used by mutation. Before
scoring, it verifies the frozen semantic digests for both stores and the transcript-set digest. It
also requires the result defect class to match the manifest, every finding memory ID to exist in the
mutated snapshot, and finding arity to match the manifest's `MEMORY` or `MEMORY_PAIR` gold unit.

Runtime retrieval challenges are rejected by this static pathway. A Part 2
`retrieval_shadowing` mutation contains a challenge rather than an observed positive, so its
evaluation uses the paired Part 5E methodology described below.

The controlled static evaluator preserves exactly three scientific label concepts:

- `INJECTED_POSITIVE` is the mutation's one exact controlled gold unit;
- `VERIFIED_CLEAN` describes a base-only, non-gold unit when the manifest explicitly declares a
  `CURATED_CLEAN` base fixture; and
- `UNKNOWN_NATURAL` describes a base-only, non-gold unit when base-store status is `UNKNOWN`.

`UNKNOWN_NATURAL` is not a negative, clean label, or false positive. An alert on an unlabeled
natural unit may be correct, incorrect, or ambiguous, so it is reported separately and never added
to a precision denominator.

A non-gold finding involving any manifest-declared created or modified memory ID is instead placed
in the separate `MUTATION_CONTEXT` accounting bucket. That bucket is unscored and is not a fourth
scientific label. Mutation-context classification takes precedence even for a curated-clean base,
because mutation-created or modified context was not independently established as clean.

## Gold matching and duplicates

A prediction matches the injected positive only through exact defect class and exact memory-ID unit
identity. `MEMORY_PAIR` IDs are compared order-independently. Content, target roles, evidence text,
finding ID, confidence, checker cost, and checker stats do not change the match.

Each eligible trial contributes exactly one injected-positive unit. One or more exact findings mean
that unit was detected once; zero exact findings mean it was missed. Additional distinct findings on
the same gold unit are recorded as duplicate-positive diagnostics. They neither increase recall nor
become false positives.

All other findings are accounted in this order:

1. any created or modified ID: unscored mutation context;
2. all IDs from a `CURATED_CLEAN` base: verified-clean alert; or
3. all IDs from an `UNKNOWN` base: unknown-natural alert.

Finding IDs are retained only as canonical diagnostic references. They do not define gold matching.

## Safe static summary

`summarize_mutation_trials` reports detected and missed injected positives, exact and duplicate gold
findings, verified-clean alerts, unknown-natural alerts, mutation-context alerts, and total findings.
It may compute `injected_positive_recall` because every eligible controlled static trial contributes
one known injected positive:

```text
detected injected-positive trials / eligible injected-positive trials
```

Part 6A does not compute precision, F1, accuracy, specificity, or false-positive rate. Ignoring
unknown-natural alerts and naming the remaining ratio precision would not make the unknown units
verified negatives. The global summary may mix defect classes, but that aggregate is not sufficient
for publication reporting; later experiment tooling must also preserve per-defect results.

## Paired retrieval challenge summary

`summarize_retrieval_challenges` consumes only frozen Part 5E
`PairedRetrievalChallengeAssessment` values. It accepts no mutation manifest, `RetrievalProbe`, or
gold label. One summary requires a common explicit sufficiency policy, retriever ID, retriever
version, and `top_k`; mixing any of those experimental conditions is rejected. Queries and target
sets may differ by case and query text is never included.

The summary reports induced-shadowing, resilient, and baseline-insufficient case counts. Its
eligible denominator is:

```text
induced-shadowing cases + resilient cases
```

Baseline-insufficient cases are excluded because a mutated-store miss cannot establish a mutation
effect when the target was already unretrievable on the base store. The induced-shadowing rate is:

```text
induced-shadowing cases / baseline-eligible cases
```

When no case is baseline-eligible, the rate is explicitly `null`, never NaN. This rate is the
fraction of eligible controlled challenges in which the mutation induced retrieval insufficiency;
it is not accuracy, recall, precision, or generic checker performance.

## Current evidence boundary

No final research benchmark has been run. No precision or F1 claim is supported. Unknown-natural
alerts remain unlabeled, mutation-context alerts remain unscored, injected-positive recall applies
only to controlled injected positives, and retrieval induced-shadowing rates apply only to
baseline-eligible paired challenges under one fixed retrieval condition.
