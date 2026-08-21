# Semantic groundwork

The semantic layer separates transcript evidence resolution, composition, and semantic judgment. It
contains no checker module; the dependency-injected unsupported-claim checker consumes these frozen
contracts from the separate checker package. One optional implementation can run a pinned
three-class NLI model locally on CPU; there is no API-hosted or generative-model provider.

## Evidence resolution

`resolve_declared_evidence(memory, transcripts)` resolves the portable `SourceRef` values on a
`NormalizedMemory`. It performs only structural coordinate resolution:

- a whole-transcript reference produces one segment per turn, ordered by turn index;
- a whole-turn reference produces one segment containing the exact turn content;
- a character-span reference uses exact Python character indexing and preserves its span;
- missing transcripts, missing turns, and out-of-range spans produce structural issues rather than
  text segments.

Each `EvidenceSegment` retains the source-reference index, transcript and turn coordinates, role,
optional span, and exact source text. Repeated declarations are preserved even when they resolve to
the same turn or text. Memories whose provenance is `unavailable` or `known_absent` produce neither
segments nor issues; that result makes no claim about support, correctness, or cleanliness.

A whole-transcript reference to an existing empty transcript is structurally resolvable and
produces zero segments and zero issues. This means no resolved semantic text is available; it does
not mean entailed, neutral, unsupported, or clean. The unsupported-claim checker treats the absence
of text as an assessability condition and abstains.

The resolver does not decide which transcript roles are semantically authoritative. It preserves
each turn's exact role, text, and coordinates. The separate composition policy retains every role;
the unsupported-claim checker uses the frozen primary PLAIN representation by default. The resolver
itself still returns individual segments and does not concatenate them into a premise.

Structural issues use the same `missing_transcript`, `missing_turn`, and `invalid_span` distinctions
as the orphaned-provenance checker. A broken reference is therefore not converted into a semantic
`neutral` or `contradiction` judgment. Resolution does not compare memory content with transcript
content.

Resolved segments contain transcript text because a semantic judge needs its premise. They
are internal semantic inputs and are not automatically embedded into `CheckerResult`. Future
semantic checkers should expose minimal coordinates and scores through `EvidenceItem` unless source
text output is explicitly required.

## Evidence composition

`compose_evidence()` runs only after structural evidence resolution. It receives a nonempty tuple of
resolved `EvidenceSegment` values and never reaches back into `TranscriptSet` or bypasses the frozen
resolver. An empty tuple raises `SemanticCompositionError` because zero resolved evidence is
unassessable; the function does not invent an empty premise.

Composition is independent of caller tuple order. Segments are sorted by transcript ID, turn index,
span, source-reference index, role, and exact text. Declarations that are identical in transcript
ID, turn index, span, text, and role are then deduplicated. The original declaration count remains
in `segment_count`, while `unique_segment_count` records the post-deduplication count. Equal text at
different transcript coordinates remains separate evidence.

`EvidenceCompositionStyle` contains exactly two representations:

- `plain`: exact segment texts joined by one newline;
- `role_labeled`: each segment rendered as its exact normalized role, `": "`, and exact text, with
  rendered segments joined by one newline.

Neither style filters roles, expands character spans, strips text, normalizes Unicode or case,
rewrites pronouns, summarizes, or adds instructions. Composition does not truncate or chunk. If the
complete composed premise and hypothesis exceed the local judge limit,
`LocalNLISemanticJudge` remains responsible for raising `SemanticInputTooLongError`.

The fixed 18-case `evidence_composition_probe_v0.1.json` development probe compared both styles with
the pinned MiniLM Part 4B development judge. Both representations produced the same aggregate
development result: 15/18 overall, including 3/6 neutral, 6/6 entailment, and 6/6 contradiction.
Because all higher-priority semantic criteria tied, the frozen selection rule chose `plain` on fewer
input tokens (1,702 versus 2,022 across the identically structured evaluation calls).
`PRIMARY_EVIDENCE_COMPOSITION_STYLE` and the `compose_evidence()` default therefore use `plain` for
the unsupported-claim checker prototype.

This small probe is not a MemLint benchmark, publication evaluation, mutation gold, or audit-time
labels. It was not derived from Part 2 mutation manifests and does not show that either style is
universally better. The primary policy remains subject to later external evaluation. No
internal-contradiction checker exists yet.

## Semantic judgment contract

`SemanticJudge` is a provider-independent protocol with nonblank `judge_id` and `judge_version`
identities and one directional operation. `semantic_judge_identity()` provides reusable runtime
validation for those identity strings and returns the exact declared values without normalization:

```python
judgment = judge.judge(
    premise=composed.text,
    hypothesis=stored_claim,
)
```

The premise is evidence or context. The hypothesis is the claim being evaluated. The
unsupported-claim checker uses composed resolved transcript evidence as the premise and memory
content as the hypothesis. A future contradiction checker may evaluate claims in both directions;
that checker does not exist yet.

`SemanticRelation` has three generic NLI-style values:

- `entailment`: the premise supports the hypothesis under the judge's decision rule;
- `contradiction`: the premise and hypothesis are incompatible under that rule;
- `neutral`: neither entailment nor contradiction was established.

These are semantic-layer relations, not taxonomy labels such as supported or unsupported.
`SemanticJudgment.score` is a judge-specific confidence or decision score in `[0, 1]`; it is not
automatically a calibrated probability. `SemanticUsage` records nonnegative model-call and token
counts without prices, timestamps, or runtime duration.

Semantic numeric fields form a strict provider boundary. Usage counters and evidence coordinates
accept only actual nonnegative Python integers, not booleans, floats, or numeric strings. Scores
accept finite Python integers or floats in `[0, 1]`, but reject booleans and numeric strings.

## Local NLI judge

`LocalNLISemanticJudge` is an optional CPU implementation of the frozen `SemanticJudge` protocol.
Install it separately from the lightweight core:

```bash
python -m pip install -e '.[semantic-local]'
```

Construction requires an exact Hugging Face `model_id` and a nonblank `revision`. Recorded research
runs use a full immutable model-repository commit SHA rather than `main`. The declared identity is
preserved as `judge_id = "hf-nli:<model_id>"` and `judge_version = "<revision>"`; the implementation
does not silently normalize either value.

The loader sets `trust_remote_code=False`, requests safetensors weights, moves the classifier to CPU,
and calls `eval()`. Torch and Transformers imports are lazy, so importing `memlint` or
`memlint.semantics` does not require the optional extra. No model is downloaded by normal tests or
normal CI.

The judge reads `id2label` and `label2id` from the loaded model configuration and requires an
unambiguous mapping containing exactly `contradiction`, `entailment`, and `neutral`. Label-name case
is ignored during validation, but numerical order is never inferred from a checkpoint name.
Configurations containing generic names such as `LABEL_0` are rejected.

Each call receives exactly one premise/hypothesis pair. Both strings must be nonblank. The tokenizer
encodes the complete directional pair, including special tokens, with `truncation=False`. The
effective limit is the smallest declared tokenizer limit, declared model position limit, and the
explicit 4,096-token local safety cap. If the complete encoded pair is longer, the judge raises
`SemanticInputTooLongError` with only `observed_tokens` and `maximum_tokens`; it never includes the
input text or silently chooses an evidence-composition policy.

Inference is one deterministic classifier forward pass inside Torch inference mode. The selected
relation is the argmax of the three logits. `SemanticJudgment.score` is the softmax value for that
selected class. It is a judge-specific classifier score, not a calibrated factual probability and
not an unsupported-claim threshold. Usage reports one model call, the actual encoded pair length as
input tokens, and zero output tokens. No runtime duration is stored in semantic or checker models.

The local judge remains dependency-injected semantic infrastructure and is loaded by `memlint audit`
only when `unsupported_claim` is explicitly selected and configured. There is still no
internal-contradiction, instruction, or retrieval checker.

## Judge selection probe

`tests/fixtures/semantic_probe_v0.1.json` contains exactly 18 independent development sanity cases:
six entailment, six contradiction, and six neutral. It is not a MemLint benchmark, is not audit-time
gold, and is not derived from mutation manifests or Part 2 controlled mutation templates.

`tools/evaluate_semantic_judge.py` runs the fixture explicitly for one pinned checkpoint. It reports
one correctness pass, per-relation and confusion counts, incorrect case IDs, and a median over at
least three timed CPU passes after one warm-up call. Timing exists only in this selection tool. The
Part 4B comparison is restricted to these two checkpoints:

| Candidate | Model ID | Pinned model revision | License | Safetensors weights |
|---|---|---|---|---:|
| A | `cross-encoder/nli-MiniLM2-L6-H768` | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` | Apache-2.0 | 328,499,560 bytes |
| B | `cross-encoder/nli-deberta-v3-small` | `fa2804872c3b4bd748f38c0185cc85775361e735` | Apache-2.0 | 567,605,820 bytes |

The recorded comparison environment used Python 3.14.7, Transformers 5.15.1, Torch 2.13.0+cpu,
and an x86_64 CPU. The downloaded weight sizes above are for the pinned `model.safetensors` files,
not the repositories' optional ONNX or pickle artifacts.

For example:

```bash
python tools/evaluate_semantic_judge.py \
  --model-id cross-encoder/nli-MiniLM2-L6-H768 \
  --revision b95119ce93d3e065de6214e38cd4a97b0f2f2c6d \
  --cases tests/fixtures/semantic_probe_v0.1.json \
  --device cpu
```

Probe results select infrastructure for further validation only. They do not establish
unsupported-claim precision or recall, MemLint accuracy or F1, publication performance, or a final
research winner. Part 2 mutation gold is not used to tune the model, labels, thresholds, or any
future aggregation policy.

## Contradiction pair-policy probe

A future `internal_contradiction` finding covers an unordered memory pair, but an NLI judgment is
directional. The `contradiction_pair_probe_v0.1.json` development fixture therefore judges every
pair twice: memory A as premise with memory B as hypothesis, then memory B as premise with memory A
as hypothesis. It compares exactly two symmetric relation-only rules:

- `any_direction`: contradiction when either directional relation is `contradiction`;
- `both_directions`: contradiction only when both directional relations are `contradiction`.

Neither rule uses scores, thresholds, lexical exclusivity rules, transcripts, evidence composition,
or mutation metadata. The 18 fixed pairs comprise six clear contradictions, six simultaneously
compatible facts, and six temporal/change-compatible facts. This is a development probe, not
publication benchmark data, Part 2 gold, or audit-time ground truth.

The pinned CPU MiniLM run produced:

| Policy | Correct | Clear contradictions | Normal-compatible false positives | Temporal-compatible false positives |
|---|---:|---:|---|---|
| `any_direction` | 8/18 | 6/6 | N1, N2, N3, N4, N5, N6 | T1, T3, T4, T5 |
| `both_directions` | 11/18 | 6/6 | N1, N2, N3, N4 | T1, T4, T5 |

Both policies were pair-order invariant. N5, N6, and T3 were directionally asymmetric. The frozen
selection priority considers total compatible false positives first, then temporal false positives,
clear-contradiction detection, directional asymmetry, and finally a conservative
`both_directions` tie-break. That ordering makes `both_directions` the priority candidate here, but
seven false positives across twelve compatible cases, including three temporal cases, are still
substantial. No primary pair policy is frozen from this probe, no contradiction-performance claim is
made, and no internal-contradiction checker exists.

## Contradiction NLI robustness sweep

Part 4F showed that MiniLM pair aggregation failed the high-precision goal for an eventual internal
contradiction checker. Part 4F2 uses the same frozen 18-pair probe to distinguish a checkpoint-specific
failure from a broader limitation of ordinary pairwise three-class NLI. It compares exactly the two
existing MemLint development candidates, a stronger conventional SNLI/MNLI DeBERTa baseline, and a
DeBERTa checkpoint trained across broader adversarial, logical, fact-verification, and varied NLI
tasks. Each model is pinned to a full immutable repository commit and is evaluated by the unchanged
local CPU judge with `any_direction` and `both_directions` relation-only aggregation.

The readiness criteria were fixed before the new model outputs: at least five of six clear
contradictions detected, zero temporal-compatible false positives, and at most one false positive
across all twelve compatible pairs. These are development criteria, not a paper benchmark or a
publication performance claim. They do not change the unsupported-claim judge selection, and later
held-out evaluation would still be required before making any performance claim. The internal
contradiction checker still does not exist.

## Simultaneous-compatibility reframing probe

Ordinary claim-to-claim NLI failed the Part 4F2 high-precision gate. Part 4G changes the semantic
question rather than adding a score threshold: it asks whether two explicitly rendered memory claims
can both be true as stated. The premise wording, hypothesis, relation mapping, and aggregation rule
were frozen before model outputs. The premise is exactly `Memory claim 1: <CLAIM_1>`, a newline, and
`Memory claim 2: <CLAIM_2>`; the hypothesis is exactly
`These two memory claims can both be true as stated.` NLI entailment maps to `compatible`,
contradiction maps to `incompatible`, and neutral maps to `uncertain`; scores remain diagnostic only.
The pair is rendered in both claim orders. It is `incompatible` only when both renderings are
incompatible, `compatible` only when both are compatible, and otherwise `uncertain`.

The pre-frozen readiness gate requires at least five of six clear incompatibilities, no false
incompatibilities among twelve compatible pairs or six temporal-compatible pairs, at least ten of
twelve compatible pairs labeled compatible, and at least five of six temporal-compatible pairs
labeled compatible. This remains a development probe, not benchmark performance, and no internal
contradiction checker or production compatibility judge exists.

Part 2 `conflict_relation = exclusive_value` is mutation and gold-generation context, not
detector-visible evidence. A future checker must never use it. A controlled mutation is suitable for
semantic detector evaluation only when incompatibility is recoverable from detector-visible
normalized content and state. Cases whose incompatibility depends on hidden mutation assumptions
must be excluded or separately marked in future benchmark construction.

## Isolation

Production semantic code depends only on normalized models. It does not import checkers or mutation
code, inspect backend-native `raw`, read mutation templates or manifests, or contain prompt text.
No LLM client, remote provider adapter, or embedding system is implemented. The semantic package
contains no checker module; dependency direction runs only from the unsupported-claim checker to the
semantic contracts.
