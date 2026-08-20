# Semantic groundwork

The semantic layer separates transcript evidence resolution from semantic judgment. It contains no
defect checker. One optional implementation can run a pinned three-class NLI model locally on CPU;
there is no API-hosted or generative-model provider.

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
not mean entailed, neutral, unsupported, or clean. A future semantic checker must treat the absence
of text as an assessability condition.

The resolver does not decide which transcript roles are semantically authoritative. It preserves
each turn's exact role, text, and coordinates. A future semantic checker or composition policy will
decide how those segments are used. The resolver returns individual segments and does not
concatenate them into a premise.

Structural issues use the same `missing_transcript`, `missing_turn`, and `invalid_span` distinctions
as the orphaned-provenance checker. A broken reference is therefore not converted into a semantic
`neutral` or `contradiction` judgment. Resolution does not compare memory content with transcript
content.

Resolved segments contain transcript text because a future semantic judge needs its premise. They
are internal semantic inputs and are not automatically embedded into `CheckerResult`. Future
semantic checkers should expose minimal coordinates and scores through `EvidenceItem` unless source
text output is explicitly required.

## Semantic judgment contract

`SemanticJudge` is a provider-independent protocol with nonblank `judge_id` and `judge_version`
identities and one directional operation. `semantic_judge_identity()` provides reusable runtime
validation for those identity strings and returns the exact declared values without normalization:

```python
judgment = judge.judge(
    premise=segment.text,
    hypothesis=stored_claim,
)
```

The premise is evidence or context. The hypothesis is the claim being evaluated. A future
unsupported-claim checker will use resolved transcript evidence as the premise and memory content as
the hypothesis. A future contradiction checker may evaluate claims in both directions. Neither
checker exists yet.

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

There is still no unsupported-claim, internal-contradiction, instruction, or retrieval checker. The
local judge is semantic infrastructure only and is not registered with `memlint audit`.

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

## Isolation

Production semantic code depends only on normalized models. It does not import checkers or mutation
code, inspect backend-native `raw`, read mutation templates or manifests, or contain prompt text.
No LLM client, remote provider adapter, embedding system, or semantic checker is implemented.
