# Semantic groundwork

The semantic layer separates transcript evidence resolution from semantic judgment. It contains no
defect checker and no concrete model or provider implementation.

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

## Isolation

Production semantic code depends only on normalized models. It does not import checkers or mutation
code, inspect backend-native `raw`, read mutation templates or manifests, or contain prompt text.
No NLI model, LLM client, provider adapter, embedding system, or semantic checker is implemented in
this phase.
