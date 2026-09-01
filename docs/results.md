# Evaluation results

MemLint has been evaluated with controlled synthetic fixtures. These experiments test specific
methods under known conditions; they do not estimate real-world accuracy, false-positive rates,
identity availability, retrieval failures, or defect prevalence.

## Summary

| Question or component | Result | Status |
|---|---|---|
| Structural checkers | 27/27 injected positives detected; 0/12 clean controls alerted | **SUPPORTED** |
| Plain unsupported-claim checker | 12/12 substitutions detected, but all three semantic clean controls alerted | **CAPABILITY-LIMITED** |
| Identity-grounded candidate | Corrected the tested identity mismatch without losing unsupported detection | **SUPPORTED, OPTIONAL** |
| Trusted speaker bindings | Explicit caller assertions work; current adapters do not provide them automatically | **CAPABILITY-LIMITED** |
| Broad retrieval-shadowing hypothesis | Strong probe missed its preregistered rate threshold | **NOT SUPPORTED** |
| Negation-specific retrieval hypothesis | Matched confirmatory probe did not replicate a specific effect | **NOT SUPPORTED** |
| Confidence abstention | Confirmatory baseline produced no clean alerts to reduce | **INCONCLUSIVE** |
| Internal contradiction | CPU probes did not establish a robust production detector | **DEFERRED** |
| Injected instruction | CPU probes did not establish a robust production detector | **DEFERRED** |

## Structural checks

The controlled benchmark detected all 27 structural injected positives. None of the 12 corresponding
curated-clean controls alerted. This supports the tested mutation families for orphaned provenance,
exact redundancy, explicit stale-active records, and policy-directed scope violations. It is not a
claim of perfect performance on natural memory stores.

The same benchmark included 12 retrieval challenges. All baselines were eligible, no case induced
shadowing, all 12 were resilient, and every target remained rank 1. That challenge was too weak to
test retrieval shadowing meaningfully.

## Unsupported claims and speaker identity

The plain unsupported-claim checker detected all 12 controlled substitutions. It also alerted on
all three checker-level semantic clean controls: 86 alert occurrences across 19 unique curated-clean
memories. The main failure was representational. First-person transcript evidence such as “I …” was
compared with a named-person memory without an explicit speaker-identity bridge.

An identity development probe tested the exact premise prefix
`The speaker is {speaker_label}.\n{evidence}`:

- identity-sensitive clean entailments improved from 4/18 to 18/18;
- unsupported detection stayed 18/18 under both representations; and
- six identity-free controls had no relation changes.

The result was `SUPPORTS_H1`, limited to development evidence.

A fresh synthetic held-out probe then compared the plain checker with the separate identity-grounded
candidate:

| Outcome | Plain baseline | Identity-grounded candidate |
|---|---:|---:|
| Identity-sensitive clean entailments | 0/30 | 30/30 |
| Identity-sensitive clean alerts | 30/30 | 0/30 |
| Identity-sensitive unsupported detections | 30/30 | 30/30 |
| Identity-free clean entailments | 10/10 | 10/10 |
| Identity-free unsupported detections | 10/10 | 10/10 |

There were 0/20 exact relation changes across the identity-free controls. In the complete synthetic
capability population, 82 memories were resolved, three were unavailable, and one conflicted, so
82/86 (95.35%) were assessed. All evaluation gates passed and the result was
`SUPPORTS_CANDIDATE`.

That coverage belongs only to the constructed fixture. Current adapters do not automatically supply
both trustworthy turn attribution and a human-readable speaker label. The candidate is therefore
optional and nondefault (`OPTIONAL_EXPLICIT_API_READY`, not `DEFAULT_READY`).

## Retrieval experiments

The strong development probe kept the evaluation-only lexical BM25 implementation, tokenizer,
`top_k=3`, and `ALL_EXPECTED` policy unchanged. All 24 strong baselines were eligible and 8/24
(33.33%) induced shadowing. The criterion required at least eight induced cases and a rate of at
least 40%, so the result was `DOES_NOT_SUPPORT_H4`.

| Strong family or control | Induced shadowing |
|---|---:|
| Query-term crowding | 1/8 |
| Negated-value decoys | 7/8 |
| Contextual mentions | 0/8 |
| Low-overlap controls | 0/6 |

The 7/8 negation concentration was noticed after that run. A fresh matched confirmatory probe tested
whether negation itself explained the effect. All 18 baselines were eligible; negated distractors
induced 2/18, matched contextual distractors induced the same 2/18, and low-overlap distractors
induced 0/18. Both negation-specific and reverse-specific discordance counts were zero. The result
was `DOES_NOT_SUPPORT_H4_N`; the apparent negation-specific effect did not replicate.

These results do not imply that retrieval shadowing is absent. They only close the tested synthetic
branch without supporting either preregistered hypothesis.

## Confidence abstention

The confidence-selectivity probe calibrated over 48 judgments. All six thresholds were eligible,
and the deterministic selection rule chose 0.50. The confirmatory results were:

| Outcome | Plain baseline | Selective policy |
|---|---:|---:|
| Clean alerts | 0/24 | 0/24 |
| Unsupported alerts | 24/24 | 24/24 |
| Abstentions | — | 0/48 |

The baseline challenge and clean-alert-reduction gates failed; unsupported safety and coverage
passed. The exact result was `INCONCLUSIVE_BASELINE_TOO_EASY`. The identity-free synthetic fixture
did not reproduce enough clean false alerts to test whether abstention would reduce them. This is
neither support for nor evidence against H3.

## Current limitations

MemLint has five public static checker classes: `orphaned_provenance`, `redundancy_bloat`,
`stale_active`, `privacy_scope_violation`, and `unsupported_claim`. Their claims are checker-specific.
The identity-grounded unsupported-claim implementation remains a separate non-CLI candidate.

Retrieval observation, sufficiency, paired assessment, and finding projection are implemented, but
MemLint does not include a live production retriever. `internal_contradiction` and
`injected_instruction` remain deferred because the available zero-cost CPU probes did not identify
methods robust enough for production checkers.

Reliable semantic auditing depends on whether the available evidence contains the context needed to
evaluate a claim. Speaker identity is the clearest example. Missing context should reduce reported
coverage or produce abstention rather than be inferred from roles, scope IDs, metadata, or claim text.

## Reproducibility

The result artifacts are external to the repository. Their SHA-256 identifiers are:

| Result artifact | SHA-256 |
|---|---|
| Controlled benchmark | `fe20c4e8c6512da9874318129464bd896871ec5257520870df746e630346d5af` |
| Identity development probe | `a205fd355291d42aab0dc267241d5b5ea03613f1d324e64ca6cf4ec8a9320219` |
| Identity held-out probe | `3fea4d3d27a6082e259794210ae20f8aa444895810b3e13f040f12bbfcfa8380` |
| Strong retrieval probe | `a0d71428c050908c8a43e288dae41266857b8bc577c28ba40a6fdf94c6f5d874` |
| Negation confirmatory probe | `fdec1cd253bdfb078f60b1b9a40d35b1058362d4b591933f37adf3db8515eba2` |
| Confidence-selectivity probe | `a16bf5d4bc98973b3f43061c794959adeddf3faf81f99a8b907fa94f171ec9dd` |

The repository retains the corresponding fixtures, evaluation modules, runners, and validation
tests. These hashes identify external results; the result JSON files are not committed.
