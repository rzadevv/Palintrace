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

## Benchmark v0.1 freeze

The controlled benchmark specification was frozen on 2026-08-23 before detector inference:

- schema: `0.1`;
- benchmark ID: `memlint-controlled-v0.1`;
- execution status: `NOT_RUN`; and
- canonical specification SHA-256:
  `fd11b0d547197495d51684f005ac17c861392891e464d818815e04eb6f37dad0`.

The static held-out scope contains only implemented detector classes: 9 orphaned-provenance, 6
redundancy/bloat, 6 stale-active, 6 privacy/scope, and 12 unsupported-claim mutation cases, for 39
controlled static trials. Fifteen separate held-out clean controls provide one unmutated H1, H2,
and H3 case for each of those five classes. `internal_contradiction` and
`injected_instruction` remain deferred negative method-development results and are not benchmarked
as implemented detectors. No overall eight-class accuracy is defined.

Three newly authored synthetic fixture bundles, H1 through H3, are explicitly `CURATED_CLEAN` for
these controlled conditions. Their label comes from deliberate construction and input-suitability
audits, not from a claim that natural deployed stores are globally clean. Structural checkers are
used only to validate the unmutated input fixtures. Unsupported-claim cleanliness is established by
the explicit memory/source construction; MiniLM is not used to certify the labels.

The operational matrix contains 12 retrieval challenges, four per fixture, under the sole intended
condition `lexical-baseline-k3`: `all_expected`, `top_k=3`, experimental retriever kind
`experimental_lexical`, configuration version `0.1`. Each challenge is constructible with the
unchanged Part 2 `distractor_crowding` mutation and its fixed editor-family distractors. No paired
observations or retrieval rates were produced during the specification freeze.

The old example store and transcripts and all Part 4 semantic, composition, contradiction,
instruction-compatibility, and injection probes are registered as `DEVELOPMENT`. They are excluded
from held-out evaluation. Deterministic collision checks cover exact IDs, contents, case IDs, and
Part 4 premise/hypothesis pairs; these checks are leakage sanity tests, not proof of semantic
independence.

The byte-level held-out file hashes are frozen in
`tests/fixtures/benchmark_v0.1.sha256.json`:

| File | SHA-256 |
|---|---|
| `README.md` | `5037ed3d59a3680ef51371dd5f982eb3abc887e1e9c6c3226ecf28f13222226c` |
| `benchmark.json` | `93013dccf171db8ca6cb8558bccbd10021a19d65673a31a490e99fc39baaa9fc` |
| `fixture_h1_store.json` | `2b688007a19a88835779b9ad44d02891f5c2f643930811a21d4ac477904a7474` |
| `fixture_h1_transcripts.json` | `0ac39a2003056e8d48a91bc597cf1c8e73477afe85a4daadd59e46169a66b9d9` |
| `fixture_h2_store.json` | `6eef745671708adb38844c358bb25a29692657478331df49f1bad9aab5df0728` |
| `fixture_h2_transcripts.json` | `23adf3c70943bb9f0a5c633bed9a59cafede2018398f7f47121643338ebfe9c0` |
| `fixture_h3_store.json` | `0cbd93b42229ca94f672ceee788f8133bf150a0a2499d561eb537f5821a51fed` |
| `fixture_h3_transcripts.json` | `1e159f90df2b29f447b94658b3b92450927b4c3f903b16e07cbfd513ca02323a` |
| `scope_policy.json` | `ed412b9dbb8b1e13bb8a42c66a03ab191a3d6bf45906cee126229974b510a999` |

The benchmark has not been executed. No detector output, performance metric, result file, or model
download was produced during the freeze. Editing held-out contents after observing outputs is not
permitted under v0.1; any such change requires a new explicitly versioned benchmark.

## Benchmark execution methodology

Part 6C freezes how benchmark v0.1 will be executed and summarized without executing any held-out
case. The Part 6B case specification, fixture bytes, canonical benchmark SHA, mutation requests,
queries, targets, checker identities, semantic method identity, and retrieval condition remain
unchanged. The research runner performs all benchmark and fixture hash checks before constructing
the pinned CPU MiniLM judge. It accepts no threshold, model, retriever, or condition override.

### Static and clean-control accounting

Static mutation trials retain the Part 6A injected-positive accounting. Per defect class, the
descriptive positive metric is:

```text
injected-positive trials detected / injected-positive trials
```

Each unmutated clean control is an explicitly `CURATED_CLEAN` whole-store audit case. Its case-level
alert rate is:

```text
clean-control cases containing one or more findings / clean-control cases
```

One control containing two findings therefore contributes one alerting case and two verified-clean
alerts. `clean_control_alert_rate` is not a generic false-positive rate: the controls are whole-store
audits, not an exhaustive enumeration of every negative memory or memory-pair unit. The methodology
does not compute precision, F1, accuracy, specificity, or generic false-positive rate. It continues
to report unknown-natural alerts, mutation-context alerts, and duplicate-positive findings as
separate diagnostics.

Benchmark v0.1 has 39 mutations but only three independently authored synthetic base fixtures.
Cases derived from the same fixture are correlated. Results must therefore retain exact counts and
per-defect reporting; this methodology defines no naive confidence intervals, p-values, or other
inferential statistics that treat all mutation trials as independently sampled deployed systems.
The holdout is a content-and-case holdout. It is not evidence of complete generalization across a
mutation family or representative real-world stores.

### Frozen experimental lexical baseline

The evaluation-only `experimental_lexical` retriever, version `0.1`, implements the one frozen
`lexical-baseline-k3` condition. It is deliberately not part of `memlint.retrieval` and is not a
production backend. It scores every normalized memory's `content` and no other field. Tokenization
uses exactly the ASCII-alphanumeric regular expression `[A-Za-z0-9]+`, with each match lowercased.
There is no stemming, lemmatization, synonym expansion, stop-word list, query rewriting, embedding,
or domain vocabulary.

For each distinct query token `q`, with `N` candidate memories:

```text
df(q)  = number of memory contents containing q
idf(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
```

The document contribution is standard deterministic BM25-style scoring with `k1 = 1.2` and
`b = 0.75`:

```text
idf(q) * tf(q,d) * (k1 + 1)
         ---------------------------------------------
         tf(q,d) + k1 * (1 - b + b * dl / avgdl)
```

Only candidates with total score greater than zero are returned. Results sort by descending score,
then ascending memory ID, and receive one-based ranks after the `top_k` cut. Empty stores and
token-free queries return no hits. Usage records one retrieval call and the complete store memory
count. The retriever receives only the store at construction and `query` plus `top_k` at execution;
it never receives expected targets, distractor IDs, manifests, mutation IDs, or gold labels.

Retrieval orchestration creates separate baseline and mutated retrievers, then reuses the frozen
Part 5 audit observation, explicit sufficiency, paired challenge, and baseline-eligible summary
contracts. The public execution artifact retains query hashes rather than query text. The sole
retrieval metric remains `induced_shadowing_rate` among baseline-eligible cases; it is not retrieval
accuracy.

### Artifacts and non-execution status

The future runner writes one canonical benchmark result at schema `0.1` and a separate environment
provenance artifact. Canonical results contain no timestamps, latency, transcripts, memory contents,
mutation substitution parameters, or runtime host identity. Safe provenance records version-only
Python/platform and local semantic dependency information, the pinned model identity/revision,
CPU device, and benchmark SHA; it does not affect scoring.

Benchmark v0.1 remains `NOT_RUN`. Part 6C produced no held-out static predictions, clean-control
predictions, paired retrieval outcomes, benchmark result files, semantic model inference, or
performance metrics.
