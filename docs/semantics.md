# Semantic evidence

Palintrace separates transcript evidence resolution, evidence composition, and semantic judgment. The
semantic package is independent of checker and mutation code; checkers receive a `SemanticJudge`
through dependency injection.

## SemanticJudge contract

`SemanticJudge` declares a nonblank `judge_id`, a nonblank `judge_version`, and one directional
operation:

```python
judgment = judge.judge(premise=evidence, hypothesis=memory.content)
```

`SemanticJudgment` records one of three relations:

- `entailment`: the premise supports the hypothesis under the judge's decision rule;
- `contradiction`: the two inputs are incompatible under that rule; or
- `neutral`: neither relation was established.

Its score is a judge-specific value in `[0, 1]`, not automatically a calibrated probability.
`SemanticUsage` records model-call and token counts without prices, latency, timestamps, or input
text.

## Evidence resolution

`resolve_declared_evidence(memory, transcripts)` follows the memory's portable `SourceRef`
coordinates without making a semantic claim:

- a transcript reference produces one segment per turn;
- a turn reference produces the exact turn content;
- a character span uses exact Python character indexing; and
- missing transcripts, turns, or invalid spans become structural issues.

Every `EvidenceSegment` retains its transcript and turn coordinates, role, optional span, and exact
text. Repeated declarations remain visible at this stage. Memories with `known_absent` or
`unavailable` provenance produce no segments and no issues; that means they are not assessable from
declared evidence, not that they are supported or clean.

Broken references are not converted into neutral or contradictory semantic judgments. They belong
to provenance checking.

## Evidence composition

`compose_evidence()` accepts a nonempty tuple of resolved segments. It sorts them by source
coordinates and content, removes exact duplicate declarations, and reports both original and unique
segment counts.

Two rendering styles exist:

- `plain`: exact segment texts joined by newlines; and
- `role_labeled`: each segment rendered as `<role>: <text>`.

The public unsupported-claim checker uses `plain`. Composition does not filter roles, normalize
case, rewrite pronouns, summarize, truncate, or add instructions. Empty evidence raises
`SemanticCompositionError`.

## Bundled local NLI judge

`LocalNLISemanticJudge` is an optional CPU implementation:

```bash
python -m pip install -e '.[semantic-local]'
```

Construction requires an explicit Hugging Face model ID and revision. The implementation loads
safetensors with `trust_remote_code=False`, moves the classifier to CPU, and runs it in inference mode.
Torch and Transformers imports are lazy, so core installs and ordinary structural audits do not load
the model.

The model configuration must map labels unambiguously to contradiction, entailment, and neutral.
Each call encodes one complete premise/hypothesis pair with truncation disabled. Inputs above the
effective model/tokenizer limit raise `SemanticInputTooLongError`; Palintrace does not silently remove
evidence. The selected-class softmax value is reported as a classifier score, not factual
probability.

## Unsupported-claim semantics

`UnsupportedClaimChecker` assesses only memories with declared, resolvable, nonempty transcript
evidence. Its pipeline is:

1. resolve declared source coordinates;
2. reject structurally broken or empty evidence;
3. compose the evidence using `plain`;
4. call the configured semantic judge with evidence as premise and memory content as hypothesis;
5. emit no finding for entailment; and
6. emit an `unsupported_claim` finding for neutral or contradiction.

The result includes source coordinates, relation, score, judge identity, composition style, and
segment counts, but not transcript or memory text. Checker statistics expose skipped and assessed
records so that zero findings cannot be mistaken for complete coverage.
