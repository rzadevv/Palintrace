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
identities and one directional operation:

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

## Isolation

Production semantic code depends only on normalized models. It does not import checkers or mutation
code, inspect backend-native `raw`, read mutation templates or manifests, or contain prompt text.
No NLI model, LLM client, provider adapter, embedding system, or semantic checker is implemented in
this phase.
