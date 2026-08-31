# Evaluation results

Part 6 evaluated MemLint's implemented checks and several development hypotheses under controlled
synthetic conditions. These results concern the frozen methods and fixtures only. They are not
estimates of real-world defect prevalence, deployment accuracy, or coverage.

## Status at a glance

| Question / component | Result | Evidence | Status |
|---|---|---|---|
| Structural controlled benchmark | All 27 injected positives detected; 0/12 structural clean controls alerted | Controlled benchmark v0.1 | **SUPPORTED** |
| Frozen `UnsupportedClaimChecker` 1.0 | 12/12 controlled substitutions detected, but all 3 semantic clean-control runs alerted | Controlled benchmark v0.1 | **CAPABILITY-LIMITED** |
| Identity-grounded unsupported-claim candidate | Removed the tested synthetic identity mismatch without losing controlled unsupported detection | Development counterfactual and fresh synthetic held-out evaluation | **SUPPORTED, OPTIONAL** |
| Trusted speaker-binding feasibility | Explicit trusted/configured assertions are sufficient; current adapters do not supply them automatically | Integration capability audit | **OPTIONAL / CAPABILITY-LIMITED** |
| Broad retrieval H4 | Stronger challenge induced 8/24 cases, below the frozen 40% requirement | Synthetic development probe | **NOT SUPPORTED** |
| Negation retrieval H4-N | Negated and matched contextual conditions each induced 2/18, with no specific discordance | Fresh matched confirmatory probe | **NOT SUPPORTED** |
| Semantic selectivity H3 | Fresh confirmatory data produced 0/24 baseline clean alerts, so the reduction question could not be tested | Calibration and confirmatory synthetic probe | **INCONCLUSIVE** |
| Internal contradiction | CPU semantic probes did not establish a robust production method | Negative method-development evidence | **DEFERRED** |
| Injected instruction | CPU semantic probes did not establish a robust production method | Negative method-development evidence | **DEFERRED** |

## Controlled benchmark v0.1

The structural methods detected 27/27 injected positives, and none of the 12 corresponding
structural curated-clean controls alerted. This is clean controlled synthetic evidence for the
implemented structural mutation families, not a real-world perfection claim.

The frozen PLAIN unsupported-claim method detected 12/12 controlled substitutions. Its clean
specificity was poor on the benchmark: all 3/3 checker-level semantic clean controls alerted,
comprising 86 alert occurrences across 19 unique curated-clean memories. The main diagnosis was a
representation mismatch: first-person transcript evidence was compared with named-person memory
claims without an explicit speaker-identity bridge. The benchmark therefore established high
controlled unsupported-claim recall alongside an identity-sensitive clean-specificity limitation;
it did not make the structural results a failure.

The original retrieval component had 12/12 eligible challenges, 0 induced-shadowing outcomes, and
12 resilient outcomes. Every target remained rank 1, showing that the original challenge was too
weak rather than that retrieval shadowing cannot occur.

## Unsupported-claim identity grounding

The Part 6F-B development counterfactual changed identity-sensitive clean entailment from 4/18
under PLAIN evidence to 18/18 under the exact speaker-grounded representation. Unsupported
detection remained 18/18 under both representations, and the six identity-free controls had no
relation changes. The frozen result was `SUPPORTS_H1`, but it was development evidence rather than
held-out or deployment validation.

The fresh Part 6G-C synthetic held-out evaluation then produced:

- identity-sensitive clean: baseline 0/30 entailments and 30/30 false alerts; candidate 30/30
  entailments and 0/30 false alerts;
- identity-sensitive unsupported: 30/30 detected under both baseline and candidate;
- identity-free controls: 10/10 clean entailments and 10/10 unsupported detections under both,
  with 0/20 exact relation changes; and
- synthetic capability population: 82 resolved, 3 unavailable, and 1 conflict, giving 82/86
  assessed memories (95.35%).

All frozen gates passed, and the result was `SUPPORTS_CANDIDATE`. The 82/86 figure is coverage in a
deliberately constructed synthetic fixture. It is not an estimate of identity availability in
deployed memory systems.

## Speaker-identity capability boundary

The integration audit concluded `OPTIONAL_EXPLICIT_API_READY`, not `DEFAULT_READY`. No current File,
Mem0, Letta, Graphiti, or `TranscriptSet` path automatically supplies both trustworthy exact-turn
attribution and a human-readable semantic speaker label. Roles, scope IDs, arbitrary metadata, raw
backend fields, episode prose, and memory claim text are not admissible substitutes.

The frozen capability contract accepts only explicit trusted or operator-configured turn-level
assertions and fails closed on unavailable, ambiguous, or conflicting identity. MemLint does not
infer speaker identity. When required context is unavailable, an identity-grounded method must
expose reduced coverage or abstain.

## Retrieval-shadowing experiments

The stronger Part 6H development probe kept the evaluation-only lexical BM25 retriever, tokenizer,
`top_k=3`, and `ALL_EXPECTED` policy unchanged. All 24 strong cases were baseline-eligible; 8/24
induced shadowing (33.33%). The frozen H4 gate required both at least eight induced cases and at
least a 40% induced rate, so the exact result was `DOES_NOT_SUPPORT_H4`.

| Strong family / control | Induced |
|---|---:|
| Query-term crowding | 1/8 |
| Negated-value decoys | 7/8 |
| Contextual mentions | 0/8 |
| Low-overlap controls | 0/6 |

The 7/8 negation concentration was identified post hoc. A fresh matched probe therefore tested H4-N
without changing the retriever. All 18 baselines were eligible. Negated distractors induced 2/18,
matched contextual distractors induced the same 2/18, and low-overlap controls induced 0/18.
`NEGATION_SPECIFIC` and `REVERSE_SPECIFIC` were both zero: the two negated failures were the same two
scenarios that failed under contextual controls. The result was `DOES_NOT_SUPPORT_H4_N`; the earlier
apparent negation-specific effect did not replicate. The current retrieval experiment branch is
closed.

Part 5 nevertheless provides useful methodology: recorded retrieval observations can be audited,
paired baseline/mutated challenges can be assessed, and retrieval-shadowing findings can be
projected. MemLint does not ship a production backend retriever. The lexical BM25 implementation is
evaluation-only.

## Semantic selectivity / abstention

Part 6J tested whether confidence-based abstention could reduce clean unsupported-claim alerts while
preserving unsupported detection and coverage. Calibration made 48 judgments; all six frozen
thresholds were eligible, and the deterministic rule selected 0.50. Confirmation produced:

| Confirmatory outcome | Baseline | Selective |
|---|---:|---:|
| Clean alerts | 0/24 | 0/24 |
| Unsupported alerts | 24/24 | 24/24 |
| Total abstentions | none | 0/48 |

The baseline-selectivity challenge and clean-alert-reduction gates failed; unsupported safety and
coverage passed. The exact interpretation was `INCONCLUSIVE_BASELINE_TOO_EASY`. The fresh
identity-free synthetic fixture did not reproduce enough baseline clean false alerts to test
whether abstention reduces them. This result is neither `SUPPORTS_H3` nor `DOES_NOT_SUPPORT_H3`, and
it does not establish confidence calibration or a production abstention policy.

## What MemLint currently supports

MemLint has exactly five implemented static/production checker classes:

- `orphaned_provenance`;
- `redundancy_bloat`;
- `stale_active`;
- `privacy_scope_violation`; and
- `unsupported_claim`.

Their claims remain checker-specific. In particular, the frozen PLAIN unsupported-claim checker has
the identity-sensitive clean-specificity limitation described above. The separate identity-grounded
candidate is nondefault, non-CLI, and optional. Part 5 retrieval accounting and finding projection
are implemented, but no production retriever is included.

## What remains deferred

`internal_contradiction` and `injected_instruction` remain deferred. Under the zero-cost CPU
constraint, their semantic probes did not establish methods robust enough for production checkers.
They must not be presented as implemented detector classes.

## Reproducibility

The result artifacts remain external to the repository. Their canonical external artifact SHA-256
values are:

| Artifact | SHA-256 |
|---|---|
| Controlled benchmark v0.1 result | `fe20c4e8c6512da9874318129464bd896871ec5257520870df746e630346d5af` |
| Identity development probe | `a205fd355291d42aab0dc267241d5b5ea03613f1d324e64ca6cf4ec8a9320219` |
| Identity candidate held-out result | `3fea4d3d27a6082e259794210ae20f8aa444895810b3e13f040f12bbfcfa8380` |
| Strong retrieval H4 result | `a0d71428c050908c8a43e288dae41266857b8bc577c28ba40a6fdf94c6f5d874` |
| Negation confirmatory H4-N result | `fdec1cd253bdfb078f60b1b9a40d35b1058362d4b591933f37adf3db8515eba2` |
| Semantic selectivity H3 result | `a16bf5d4bc98973b3f43061c794959adeddf3faf81f99a8b907fa94f171ec9dd` |

These hashes identify canonical external results; they are not repository files.

## Claim boundaries

| Claim | Supported? |
|---|---|
| MemLint has five implemented static checkers | **YES** |
| Identity grounding fixes the tested synthetic identity mismatch | **YES** |
| Identity-grounded unsupported-claim checking is ready as the default | **NO** |
| Current adapters automatically supply trusted speaker labels | **NO** |
| Retrieval shadowing was broadly demonstrated | **NO** |
| The negation-specific retrieval effect replicated | **NO** |
| Confidence abstention reduces clean alerts | **INCONCLUSIVE** |
| Internal contradiction has a production detector | **NO** |
| Injected instruction has a production detector | **NO** |

The experiments do not support overall accuracy, precision, real-world false-positive rates,
real-world retrieval-shadowing rates, or defect-prevalence estimates. Reliable memory auditing
depends not only on classifier quality but also on whether the available evidence representation
contains the context required to evaluate a claim. Speaker identity is the clearest example. When
that capability is absent, exposing reduced coverage or abstaining is more defensible than inferring
missing context.

## Part 6 conclusion

- Structural evaluation: **SUPPORTED**.
- Frozen PLAIN unsupported-claim baseline: **HIGH CONTROLLED RECALL, WITH AN
  IDENTITY-SENSITIVE CLEAN-SPECIFICITY LIMITATION**.
- Explicit identity grounding: **SUPPORTED ON SYNTHETIC DEVELOPMENT AND HELD-OUT TESTS**.
- Default identity-grounded deployment: **NOT READY** because trusted bindings are not supplied
  automatically by current integrations.
- Broad retrieval H4: **NOT SUPPORTED**.
- Negation-specific H4-N: **NOT SUPPORTED**.
- Semantic selectivity H3: **INCONCLUSIVE UNDER THE CURRENT SYNTHETIC PROBE**.
- Internal contradiction: **DEFERRED**.
- Injected instruction: **DEFERRED**.

Part 6 is complete. No additional Part 6 experiment is currently justified before Part 7.
