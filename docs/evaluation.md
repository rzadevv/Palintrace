# Evaluation

Palintrace evaluates controlled mutations without exposing gold information to the checker. Detector
code receives only normalized memory data and checker-specific evidence. Mutation manifests,
expected targets, base-store labels, and changed-memory lists are joined only after a
`CheckerResult` has been produced.

Final experimental outcomes and claim boundaries are reported in [Evaluation results](results.md).

## Controlled mutation accounting

`evaluate_mutation_trial` combines:

- the original normalized store;
- the mutated normalized store;
- the separate `MutationManifest`;
- an already-produced `CheckerResult`; and
- the transcript set when the mutation used one.

Before scoring, it verifies the stored semantic digests for both stores and the transcript-set
digest. It also checks that the result defect class matches the mutation and that every finding
refers to memories in the mutated snapshot.

Retrieval mutations are not scored through this static path. They define a challenge that requires
paired runtime observations, described below.

## Evaluation labels

The static evaluator uses three scientific label concepts:

| Label | Meaning |
|---|---|
| `INJECTED_POSITIVE` | The exact memory or memory pair changed by the controlled mutation |
| `VERIFIED_CLEAN` | A non-gold unit from an explicitly curated-clean base store |
| `UNKNOWN_NATURAL` | A non-gold unit from a base store whose status is unknown |

`UNKNOWN_NATURAL` is not a negative label. An alert on an unknown natural unit might be correct,
incorrect, or ambiguous, so it is reported separately and never included in a precision
denominator.

A finding involving a memory created or modified by the mutation, but not matching the injected
gold unit, is recorded as `MUTATION_CONTEXT`. This is an unscored diagnostic bucket rather than a
fourth label.

## Gold matching

A finding matches the injected positive through:

- the same defect class; and
- the exact gold memory ID or unordered memory-ID pair.

Finding text, confidence, evidence, cost, and finding ID do not affect the match. Multiple findings
for the same gold unit count as one detected positive plus duplicate diagnostics; they do not
increase recall.

Other findings are classified in this order:

1. findings involving created or modified records become mutation-context alerts;
2. findings over a curated-clean base become verified-clean alerts; and
3. findings over an unknown base become unknown-natural alerts.

## Curated-clean controls

A `CURATED_CLEAN` label applies only to deliberately constructed fixtures that were reviewed for a
specific checker and condition. It does not mean a natural memory store is globally defect-free.
The evaluator reports both alerting control cases and the number of verified-clean alert
occurrences.

Whole-store clean-control alert rates are not generic false-positive rates. A control run may
contain many possible memory or pair units, and the benchmark does not enumerate every negative
unit.

## Static summaries

`summarize_mutation_trials` reports raw counts for:

- detected and missed injected-positive trials;
- exact and duplicate gold findings;
- verified-clean alerts;
- unknown-natural alerts;
- mutation-context alerts; and
- total findings.

Injected-positive recall is valid for the controlled trials because each eligible mutation defines
one known positive. The evaluator does not compute precision, F1, accuracy, specificity, or a
generic false-positive rate.

## Paired retrieval challenges

Retrieval shadowing is assessed by comparing two recorded observations for the same request:

1. retrieve from the baseline store;
2. retrieve from the corresponding mutated store; and
3. apply the same explicit sufficiency policy to both observations.

The pair must agree on query hash, expected targets, `top_k`, and retriever identity. The outcomes
are:

| Outcome | Meaning |
|---|---|
| `induced_shadowing` | Baseline sufficient, mutated insufficient |
| `resilient` | Baseline sufficient, mutated sufficient |
| `baseline_insufficient` | Baseline already insufficient |

Only baseline-eligible cases contribute to the induced-shadowing rate. A static distractor mutation
does not establish shadowing until retrieval has actually been observed.

## Benchmark design

The controlled benchmark combines 39 static mutations, 15 independently authored clean controls,
and 12 paired retrieval challenges. It covers the five implemented checker classes; deferred
contradiction and injected-instruction methods are not presented as production detectors.

The benchmark uses three synthetic base fixtures, so cases derived from one fixture are correlated.
Counts are reported by defect class without confidence intervals or significance tests that would
treat them as independent deployed systems.

The evaluation-only lexical retriever uses a fixed ASCII-alphanumeric tokenizer and deterministic
BM25-style scoring. It is not a production backend. Its configuration is kept fixed so retrieval
experiments stress the same method rather than tuning it after results are observed.

## Reproducibility and privacy

The benchmark runner verifies the benchmark specification and fixture manifest before constructing
the local semantic judge. Result artifacts contain identifiers, relations, scores, usage, counts,
and hashes, but omit transcript text, memory content, mutation substitutions, and host identity.
Environment provenance is stored separately from scored results.

The repository keeps the benchmark and final probe fixtures, evaluation modules, runners, and
validation tests. Result JSON files remain external; their identifiers are listed in
[Evaluation results](results.md).
