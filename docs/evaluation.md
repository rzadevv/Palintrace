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

Controlled benchmark v0.1 has now been executed once under the frozen Part 6D methodology. It
remains a small controlled synthetic benchmark rather than a final claim of real-world detector
performance. No precision or F1 claim is supported. Unknown-natural alerts remain unlabeled,
mutation-context alerts remain unscored, injected-positive recall applies only to controlled
injected positives, and retrieval induced-shadowing rates apply only to baseline-eligible paired
challenges under one fixed retrieval condition.

## Benchmark v0.1 freeze

The controlled benchmark specification was frozen on 2026-08-23 before detector inference:

- schema: `0.1`;
- benchmark ID: `memlint-controlled-v0.1`;
- execution status at specification freeze time: `NOT_RUN`; and
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

At specification freeze time, the benchmark had not been executed. No detector output, performance
metric, result file, or model download was produced during that freeze phase. Editing held-out
contents after observing outputs is not permitted under v0.1; any such change requires a new
explicitly versioned benchmark.

## Benchmark execution methodology

Part 6C freezes how benchmark v0.1 will be executed and summarized without executing any held-out
case. The Part 6B case specification, fixture bytes, canonical benchmark SHA, mutation requests,
queries, targets, checker identities, semantic method identity, and retrieval condition remain
unchanged. The research runner performs all benchmark and fixture hash checks before constructing
the pinned CPU MiniLM judge. Preflight anchors the canonical benchmark specification SHA, the
frozen fixture hash-manifest byte SHA, and every fixture byte SHA. The runner accepts no threshold,
model, retriever, or condition override.

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

### Artifacts and execution history

The Part 6C runner writes one canonical benchmark result at schema `0.1` and a separate environment
provenance artifact. Canonical results contain no timestamps, latency, transcripts, memory contents,
mutation substitution parameters, or runtime host identity. Safe provenance records version-only
Python/platform and local semantic dependency information, the pinned model identity/revision,
CPU device, and benchmark SHA; it does not affect scoring.

Part 6C froze the execution harness and itself produced no held-out predictions, retrieval outcomes,
benchmark result files, semantic inference, or performance metrics. The first Part 6D attempt then
failed during environment/model construction before any held-out case executed and produced no
result artifact; that failed startup is not a benchmark result. After isolating the Python
environment, Part 6D completed the first successful frozen benchmark execution without method
changes. That successful run is the canonical v0.1 execution. Part 6E subsequently performed
descriptive post-hoc analysis without rerunning the benchmark or changing methods.

The canonical outputs are external research artifacts and are not committed to this repository:

- `benchmark-result.json` SHA-256:
  `fe20c4e8c6512da9874318129464bd896871ec5257520870df746e630346d5af`;
- `execution-provenance.json` SHA-256:
  `56b7998089d2da271a1ac879dfb4f30f6ed453f6eff073fefaf35ce65304c8b2`;
- Part 6E `benchmark-analysis.json` SHA-256:
  `860b13dfd7c98c962353bbec54bf38663f78ba2312923b65bd04b4afe59ba1c2`;
- Part 6E `benchmark-analysis.md` SHA-256:
  `cb0cac7ab4ba242e593b1f9081ffd2d42845fe1165491182751c543e9176fe3b`.

## Post-v0.1 speaker-identity probe

Part 6F-A freezes a new DEVELOPMENT hypothesis experiment motivated by the frozen Part 6E
post-hoc analysis. Part 6E found that unsupported-claim clean alerts were associated with
first-person transcript claims normalized into named-person memories while PLAIN evidence
composition did not explicitly identify the speaker. The probe addresses H1 (explicit
speaker-identity grounding) and only the narrow representation-level part of H2. It does not test
H3 confidence/selectivity policies or H4 retrieval challenge design.

The fixture `tests/fixtures/unsupported_identity_probe_v0.1.json` contains 24 fresh scenarios that
do not reuse the v0.1 held-out people or sentences: 18 identity-sensitive cases and six
identity-free controls. Every scenario has a clean hypothesis and a one-value unsupported
hypothesis. Part 6F-B evaluated both hypotheses under both frozen premise conditions, for exactly
96 judgments:

- `plain` uses `source_text` unchanged;
- `speaker_grounded_v0.1` uses exactly
  `The speaker is {person_name}.\n{source_text}`.

The identity-free controls already name the person in `source_text` but still receive the same
prefix. They test whether the prefix itself perturbs ordinary explicit-name relations. The
execution model was frozen as `cross-encoder/nli-MiniLM2-L6-H768` at revision
`b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, on CPU, with no threshold, truncation, replacement
model, or role-labeled composition.

The outcomes and interpretation are preregistered before semantic execution. PLAIN reproduces the
failure pattern only when identity-sensitive clean entailments are at most 12/18. If reproduced,
the clean-rescue gate requires at least 16/18 grounded clean entailments and an increase of at least
four. The unsupported-safety gate requires at least 17/18 grounded unsupported detections and no
drop larger than one from PLAIN. The prefix-stability gate allows at most one exact clean-relation
change and at most one unsupported detected-to-missed transition among the six identity-free
controls. These gates map to exactly `SUPPORTS_H1`, `DOES_NOT_SUPPORT_H1`, or
`INCONCLUSIVE_FAILURE_NOT_REPRODUCED`.

`speaker_grounded_v0.1` assumes a trustworthy mapping from transcript speaker to `person_name`.
MemLint production does not currently expose a general, proven source for that human-readable
identity mapping. Passing this probe would therefore support a DEVELOPMENT representation
hypothesis, not automatically justify adding the prefix to `UnsupportedClaimChecker`. A later phase
must establish whether identity grounding is available and valid in a deployable memory-system
contract.

This probe is not a second held-out benchmark or independent validation of an improved method. If
it informs v0.2, that method requires a new future held-out set. At the Part 6F-A freeze, MiniLM had
not been instantiated or run, no semantic outputs had been observed, and H3 and H4 had not been
tested. The real frozen probe was reserved for one Part 6F-B execution after
fixture/hash/schema/model preflight succeeded.

### Frozen Part 6F-B development result

Part 6F-B executed the preregistered `unsupported-identity-counterfactual-v0.1` development probe.
The canonical external `identity-probe-result.json` has SHA-256
`a205fd355291d42aab0dc267241d5b5ea03613f1d324e64ca6cf4ec8a9320219` and is not committed to this
repository. Its frozen interpretation is `SUPPORTS_H1`.

For the 18 identity-sensitive cases, PLAIN produced 4/18 clean entailments and detected 18/18
unsupported hypotheses. Speaker-grounded composition produced 18/18 clean entailments and detected
18/18 unsupported hypotheses. For the six identity-free controls, both conditions produced 6/6
clean entailments and 6/6 unsupported detections, with no relation changes. The frozen
failure-pattern, clean-rescue, unsupported-safety, and prefix-stability gates all passed.

This is a DEVELOPMENT experiment, not independent held-out validation of a v0.2 checker. It does
not make identity grounding production-ready, show real-world generalization, or prove that speaker
identity is the sole semantic failure mechanism. No production method was changed after observing
the result. H3 confidence/selectivity policy and H4 retrieval challenge design remain untested.

### Part 6G-B candidate integration freeze

Part 6G-B freezes a separate identity-grounded unsupported-claim candidate implementation before
designing or running a new held-out evaluation. The method requires explicit turn-level speaker
bindings, uses fixed PLAIN evidence plus the exact Part 6F speaker prefix, and abstains without a
PLAIN fallback when identity is unavailable or conflicting. It is not a CLI or default method.

No semantic model or held-out benchmark is run in Part 6G-B, so there are no candidate performance
results to report. A future fresh evaluation must report both semantic outcomes among assessed
memories and identity-grounding coverage and abstention counts. H3 and H4 remain untested.

## Part 6G-C identity-grounded held-out preregistration

Part 6G-C is an independently constructed synthetic held-out evaluation of the frozen
`IdentityGroundedUnsupportedClaimChecker` version `0.1`. It is isolated from benchmark v0.1 and
does not promote the candidate into the public API, CLI, defaults, or benchmark dispatch. The
fixture is `tests/fixtures/unsupported_identity_grounded_heldout_v0.1.json`, schema `0.1`, with
SHA-256 `a0384e2d4e5d7764c45c87e1c729762cbd2714ced2faa3cb7e36a2b50283169b`.

The semantic matrix has 40 fresh fictional scenarios, each evaluated with a clean and an exact
one-value unsupported hypothesis under both frozen checkers: 160 paired semantic judgments. Thirty
scenarios are identity-sensitive and ten are identity-free controls. Ten factual domains each
contribute four scenarios: software/tool, device/hardware, workplace/project, location, schedule,
subscription/preference, education/course, travel, ordinary possession, and biography. The
identity-sensitive transformations comprise 20 first-person-subject and ten
first-person-possessive cases; the controls already name their subjects. Identities, case IDs, and
sentences do not overlap the Part 6F development probe, and a preregistered lexical-similarity guard
rejects close copies of its source sentences.

The baseline remains `UnsupportedClaimChecker` version `1.0` with PLAIN evidence. The candidate
remains `IdentityGroundedUnsupportedClaimChecker` version `0.1`, also with PLAIN evidence followed
by the exact frozen prefix `The speaker is {speaker_label}.\n{evidence_text}`. Both conditions use
`cross-encoder/nli-MiniLM2-L6-H768` at revision
`b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` on CPU. There is no score threshold, truncation,
confidence abstention, alternate composition, or model override.

Six additional capability cases separately cover one resolved turn, multiple same-speaker turns,
a missing binding, a transcript-level reference, incomplete multi-turn bindings, and conflicting
speakers. Their preregistered statuses are two `RESOLVED`, three `UNAVAILABLE`, and one `CONFLICT`.
The last four must abstain with zero semantic calls and zero findings. Including the 80 resolved
candidate semantic memories, the complete candidate population is 86 memories: 82 resolved and
four capability abstentions if no resolved input exceeds the model limit. Coverage is reported
over all 86; it is never folded into conditional semantic accuracy.

### Frozen metrics and execution order

Raw counts precede rates. For identity-sensitive clean cases the result reports entailments, false
alerts, paired non-entailment-to-entailment rescues, and paired entailment-to-non-entailment
regressions. For identity-sensitive unsupported cases it reports detections, misses, and both paired
detection transitions. Controls report exact relation changes, clean regressions, and
detected-to-missed regressions. Candidate clean-entailment and unsupported-detection rates are
reported both conditionally among assessed cases and effectively over the entire eligible stratum,
so semantic abstentions reduce the effective rate.

Coverage reports total and declared candidate memories, evidence-resolvable memories, identity
`RESOLVED`/`UNAVAILABLE`/`CONFLICT`, assessed and abstained memories, semantic calls, assessment
coverage, unavailable rate, and conflict rate. Abstention is not a correct prediction.

Execution order is frozen as semantic case ID; within each case, clean then unsupported hypothesis;
within each hypothesis, baseline then candidate. The six coverage cases follow in case-ID order.
The successful full execution therefore makes 160 paired semantic calls plus two calls for the
resolved capability cases, while the four unavailable/conflicting capability cases make no call.

### Frozen gates and interpretation

The baseline failure pattern is reproduced only with at least 8/30 identity-sensitive clean false
alerts. This is a deliberately nontrivial more-than-one-quarter failure floor, not a threshold
derived to match the 6F score. If that floor is not reached, the only interpretation is
`INCONCLUSIVE_BASELINE_FAILURE_NOT_REPRODUCED`.

If the baseline pattern is reproduced, all of these preregistered gates must pass to produce
`SUPPORTS_CANDIDATE`:

- clean selectivity: at least 24/30 candidate clean entailments, at least eight paired rescues, at
  most two clean regressions, and at least eight fewer candidate false alerts than baseline;
- unsupported safety: at least 27/30 candidate detections, at most two detected-to-missed
  transitions, and a detection-count drop from baseline of at most two;
- identity-free stability: at most two exact relation changes across the 20 clean/unsupported
  control trials, at most one clean regression, and at most one detected-to-missed regression;
- abstention contract: every resolved capability case is assessed once, while every unavailable or
  conflicting case has zero calls and findings;
- regression/privacy: all protected predecessor hashes match, the candidate remains nonpublic and
  non-CLI, and no fixture speaker label, transcript, memory, hypothesis, or grounded premise appears
  in result serialization.

The 24/30 clean floor requires 80% clean entailment; the 27/30 safety floor requires 90%
unsupported detection. The paired-regression allowances prevent aggregate improvement from hiding
more than two adverse sensitive transitions, while the control bounds permit only small prefix
instability. If the baseline failure reproduces but any gate fails, the interpretation is
`DOES_NOT_SUPPORT_CANDIDATE`. These are the only three interpretation labels.

The result schema is immutable, rejects extra fields and nonfinite JSON, canonicalizes execution
rows, and recomputes all metrics, gates, and interpretation from row-level outcomes. It serializes
case IDs, relations, scores, judge identity, usage, hashes, source coordinates, counts, and identity
status—but no source text, memory content, premise, or speaker label. Safe environment provenance
is stored separately. At preregistration freeze time no semantic model has been instantiated and no
held-out relation has been observed. H3 confidence/selectivity research and H4 retrieval research
remain outside this phase.

### Frozen Part 6G-C held-out result

After the complete preregistration was committed, the pinned CPU MiniLM evaluation was executed
exactly once in forced-offline mode. The canonical external `heldout-result.json` has SHA-256
`3fea4d3d27a6082e259794210ae20f8aa444895810b3e13f040f12bbfcfa8380`; its separate external
`execution-provenance.json` has SHA-256
`51ce4c9afbca10de511c6ba96730d286624a7b4dc6f66ec90782ba8d46b800a5`. Neither artifact is
committed to this repository.

On the 30 resolved identity-sensitive clean cases, the baseline produced 0/30 entailments and
30/30 false alerts. The candidate produced 30/30 entailments and 0/30 false alerts: 30 paired clean
rescues and no clean regressions. Both methods detected all 30/30 identity-sensitive unsupported
hypotheses, with no detected-to-missed transition. On the ten identity-free controls, both methods
produced 10/10 clean entailments and 10/10 unsupported detections; exact relation changes, clean
regressions, and unsupported regressions were all zero.

Across all 86 candidate memories, 82 were identity-resolved and assessed, three were unavailable,
and one was conflicting. The four unavailable/conflicting cases abstained with zero semantic calls
and findings. Assessment coverage was 82/86 (`0.9534883720930233`), the unavailable rate was 3/86,
and the conflict rate was 1/86. Conditional and effective semantic rates were both 30/30 for
sensitive clean entailment and 30/30 for sensitive unsupported detection because every semantic
comparison case had an explicit resolved binding; the lower 82/86 deployment-contract coverage is
reported separately and must not be hidden behind those perfect conditional results.

The baseline failure reproduced, and the clean-selectivity, unsupported-safety,
identity-free-stability, abstention-contract, and regression/privacy gates all passed. The frozen
interpretation is `SUPPORTS_CANDIDATE`.

This supports the exact frozen representation on this independently constructed synthetic held-out
set. It is not real-world deployment validation or proof that production integrations can supply
trustworthy speaker bindings. The experiment supplied every semantic-case binding explicitly.
Candidate promotion, CLI/default integration, and production binding availability require later
decisions and evidence. Benchmark v0.1 remains frozen; H3 and H4 remain untested.

### Part 6G-D speaker-binding feasibility

Part 6G-D separately audited whether the current File, Mem0, Letta, and Graphiti paths can supply
the exact turn-level human labels assumed by 6G-C. No semantic model or additional semantic example
was used. None of the current adapters automatically supplies both trustworthy turn attribution
and a human-readable label. File input can be configured explicitly; a future Letta message/Identity
join is supported in principle when documented sender and label fields are complete and unique;
Mem0 and Graphiti require their surrounding integrations to preserve explicit assertions at
ingestion time. Scope IDs, roles, metadata, episode prose, groups, and raw fields are insufficient.

A separate source-admission envelope records trust class, source provenance, optional principal ID,
and speaker label before compiling to the unchanged `SpeakerIdentityBindings`. It rejects
unavailable, ambiguous, or conflicting input. Under the preregistered promotion ladder, controlled
contract and regression tests establish `OPTIONAL_EXPLICIT_API_READY`, not `DEFAULT_READY`. The
candidate remains nonpublic, non-CLI, and nondefault; real production binding prevalence remains
unknown.

## Strong retrieval-shadowing development probe

The controlled benchmark v0.1 retrieval challenge was too weak to test H4 meaningfully. All 12
paired cases were baseline-eligible, but none produced induced shadowing: every target remained at
rank one before and after the three fixed editor distractors were added. That negative challenge
result does not show that retrieval shadowing is absent; it shows that the frozen v0.1 mutation did
not stress the experimental lexical retriever.

Part 6H-A therefore preregisters a separate development probe with 30 entirely new synthetic cases:
24 strong challenges and six low-overlap resilience controls. The strong cases are divided equally
among query-term crowding, explicitly negated competing-value decoys, and non-answer contextual
mentions. Every case has one target plus three baseline non-target memories, then exactly eight
frozen distractors. The six domains are balanced with four strong cases and one control each.

The retriever remains the byte-frozen evaluation-only `experimental_lexical` version `0.1`, using
its existing tokenizer, BM25 formula, score handling, and tie order. Every case fixes `top_k=3`, one
expected target, and `ALL_EXPECTED`; no retrieval or BM25 setting is exposed for tuning. The fixture
is frozen by byte SHA before any retrieval observation. Part 6H-A uses only structural validation,
manual semantic review, independent regex-token overlap descriptions, and fake hit observations for
gate arithmetic. It observes no BM25 scores, target ranks, or real retrieval outputs.

Part 6H-B may execute the frozen runner once. The preregistered strong baseline-eligibility gate is
at least 20/24. Among eligible strong cases, the shadowing gate requires at least eight induced
cases and at least a 40% induced rate; induced cases must span at least two of the three strong
families. Controls require at least 5/6 eligible and at most one induced case. The only final labels
are `SUPPORTS_H4`, `DOES_NOT_SUPPORT_H4`, and `INCONCLUSIVE_BASELINE_CONSTRUCTION`.

This probe is development-only, not a held-out final benchmark or a production prevalence study.
Even `SUPPORTS_H4` would establish only susceptibility of this frozen lexical retriever under the
preregistered synthetic challenge. It would not establish that BM25 is generally poor, that other
memory retrievers share the behavior, or that the induced rate estimates real-world prevalence.

## Fresh negation-retrieval confirmatory probe

Broad H4 remains `DOES_NOT_SUPPORT_H4`. The frozen Part 6H-B development run induced shadowing in
8/24 strong cases (33.33%), below its preregistered 40% requirement. Part 6H-C then observed post
hoc that seven of eight negated competing-value cases were induced, compared with one of eight
query-crowding cases, zero of eight contextual-mention cases, and zero of six low-overlap controls.
That post-hoc pattern generated a new hypothesis, H4-N; it was not known before the 6H-B result and
remains `NOT TESTED`.

Part 6I-A preregisters a fresh synthetic confirmatory probe for H4-N. Its 18 newly authored base
scenarios are balanced across six organizational domains. Each scenario stores one target and three
baseline non-targets, then reuses that exact query, target, baseline, expected ID, `top_k=3`, and
`ALL_EXPECTED` policy across three matched mutations: eight explicitly negated competing-value
decoys, eight contextual non-answer controls using the same competing values one-for-one, and eight
low-overlap controls. The scenario—not its three condition variants—is the primary matched unit.

Before freezing, a separate lowercase `[A-Za-z0-9]+` tokenizer verifies all 144
negated/contextual pairs use the same competing value, have the same query-token intersection
count, and differ in length by at most three tokens. It also verifies each scenario's median
low-overlap Jaccard is below both matched conditions. These are fixture-construction invariants,
not retrieval outcomes, and the production tokenizer and BM25 implementation are not used.

The baseline gate requires at least 17/18 eligible scenarios. Confirmation then requires at least
12 negation-induced scenarios and at least two thirds of eligible scenarios; at least eight
negation-specific discordances and at most two reverse discordances; no more than five contextual
inductions; no more than one low-overlap induction; and negation-induced cases in at least five of
six domains. The only interpretations are `SUPPORTS_H4_N`, `DOES_NOT_SUPPORT_H4_N`, and
`INCONCLUSIVE_BASELINE_CONSTRUCTION`.

The fixture was frozen by byte hash before retrieval execution. Part 6I-A observed no retrieval
outputs, BM25 scores, or target ranks, and reserved the frozen runner for exactly one Part 6I-B
invocation. Its result contract stores one shared baseline per scenario, recomputes every condition's
paired assessment from stored observations, and then recomputes the global summary from those
validated scenario records.
This is fresh confirmatory synthetic evidence following a post-hoc hypothesis, not a real-world
benchmark, independent production validation, or a prevalence estimate. Broad H4 remains negative,
and H3 confidence/selectivity research remains untested.

Part 6I-B subsequently executed that frozen confirmatory probe once. All 18 baselines were eligible.
The negated and matched contextual conditions each induced shadowing in the same two scenarios;
the low-overlap condition induced none, and there were zero negation-specific discordances. The
negation-replication, matched-specificity, and domain-breadth gates failed, so the frozen
interpretation is `DOES_NOT_SUPPORT_H4_N`. Broad H4 separately remains `DOES_NOT_SUPPORT_H4`.
These synthetic negative results close the current retrieval branch without supporting a production
prevalence claim.

## Semantic selectivity / abstention probe

Part 6J-A preregisters H3, which was previously untested. The evaluation asks whether a confidence-
based decision layer can reduce clean unsupported-claim alerts while preserving unsupported
detection and bounded abstention. It does not modify the frozen MiniLM NLI model, its relation
labels, scores, tokenization, or the production `UnsupportedClaimChecker`. `ENTAILMENT` remains
`NO_ALERT`; a non-entailment is `ALERT` when its frozen maximum-softmax score meets the selected
threshold and otherwise becomes `ABSTAIN`. The score is not described as a calibrated probability.

The synthetic fixture contains 48 fresh scenarios: 24 calibration and 24 confirmatory, balanced at
four scenarios per split in each of six domains. Every premise and both hypotheses explicitly name
the same third-person subject, avoiding the speaker-grounding confound. Each clean hypothesis is
constructed as clearly supported, while its paired unsupported hypothesis changes exactly one
exclusive factual value. Both splits and all text are authored and byte-frozen before any NLI run.

Calibration considers only the frozen grid `0.50`, `0.60`, `0.70`, `0.80`, `0.90`, and `0.95`.
A threshold is eligible only when at least 22/24 calibration unsupported cases remain alerts. The
deterministic selection rule then minimizes clean alerts, maximizes unsupported alerts, minimizes
total abstentions, and finally chooses the lower threshold. No confirmatory result can influence
selection. A future Part 6J-B runner will load MiniLM once, calibrate, select mechanically, and—if
selection succeeds—immediately execute confirmation in the same one-shot invocation.

The confirmatory baseline must first produce at least four clean alerts. Unsupported safety then
requires at least 22/24 selective alerts and a drop of at most one from baseline. Clean-alert
reduction requires at least three fewer alerts and no more than half the baseline count. Total
selective abstentions across the 48 confirmatory judgments may not exceed eight, preserving at least
40 assessed decisions. The only interpretations are `SUPPORTS_H3`, `DOES_NOT_SUPPORT_H3`, and
`INCONCLUSIVE_BASELINE_TOO_EASY`.

Part 6J-A observes no NLI relation, confidence score, selected threshold, or confirmatory result.
The probe is synthetic development evidence, not production calibration, universal confidence
calibration, or a real-world prevalence study.
