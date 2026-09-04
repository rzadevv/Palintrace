# Checkers

Palintrace checkers inspect normalized memory stores and return deterministic findings. Five checkers
are supported by the public API and CLI.

| Checker | Defect class | Additional input | Method |
|---|---|---|---|
| `orphaned_provenance` | `orphaned_provenance` | transcripts | structural |
| `redundancy_bloat` | `redundancy_bloat` | none | structural |
| `stale_active` | `stale_active` | none | structural |
| `privacy_scope_violation` | `privacy_scope_violation` | scope policy | structural |
| `unsupported_claim` | `unsupported_claim` | transcripts and semantic model | semantic |

The remaining taxonomy classes—`internal_contradiction`, `retrieval_shadowing`, and
`injected_instruction`—do not have supported static production checkers. Retrieval shadowing is
assessed through paired retrieval experiments rather than store inspection.

## Common result model

Every checker returns a `CheckerResult` containing:

- checker identity and implementation version;
- rule identity, rule version, default severity, and defect class;
- zero or more deterministic `Finding` objects;
- model-call and token accounting;
- aggregate scan statistics.

A finding identifies affected memory IDs and structured evidence. Finding IDs are derived from the
checker identity, version, defect class, memory IDs, and evidence. Repeated checks over the same
normalized input therefore produce stable output.

Result severity expresses a rule's default importance. `Finding.confidence` expresses confidence in
one particular finding; it is not a severity level. CLI `--fail-on` gating compares only result
severity and finding presence, so confidence does not affect the exit status.

`CheckerResult` is also the canonical source for optional SARIF rendering: `rule_id` becomes SARIF
`ruleId`, severity determines its level, and `finding_id` is retained as a fingerprint. Memory IDs
remain logical memory identifiers rather than fabricated source-code locations.

Findings avoid serializing memory content and transcript text. Semantic findings contain hashes and
source coordinates instead. Mutation manifests and benchmark gold labels are never checker inputs.

## Orphaned provenance

`OrphanedProvenanceChecker` resolves declared `SourceRef` records against a supplied
`TranscriptSet`. It reports a memory when any declared reference has:

- a missing transcript;
- a missing turn; or
- a character span extending beyond the referenced turn.

Only `provenance_status: declared` records are eligible. A reference to an entire existing
transcript is valid when it intentionally omits a turn index. `unavailable` provenance is not
reported as orphaned. The checker requires transcripts and fails explicitly when they are absent.

## Redundancy bloat

`RedundancyBloatChecker` finds exact-content duplicates within the same observable normalized scope.
The scope key contains `user_id`, `agent_id`, and `session_id`. Completely unscoped memories are
skipped because their intended boundary is unknown.

Each duplicate group emits one finding for every distinct pair of memory IDs. Evidence contains a
content hash, content length, and scope—not the duplicated text. This checker does not attempt
paraphrase or semantic-equivalence detection.

## Stale active

`StaleActiveChecker` follows explicit `supersedes` links. It reports an older memory only when:

1. another memory explicitly names it in `supersedes`; and
2. the older memory has `active: true`.

Self-links and links to absent memories are skipped and counted. The checker does not infer
supersession from dates, wording, or conflicting values.

## Privacy scope violation

`PrivacyScopeViolationChecker` requires an explicit `ScopeIsolationPolicy`. Each policy rule names a
principal dimension (`user_id` or `agent_id`), an authoritative source principal, and prohibited
destination principals.

The checker compares portable normalized records after excluding the selected principal dimension
and the memory ID. A destination is reported only when it is an exact portable replica of a record
under the authoritative principal. Ordinary cross-scope differences are not violations without a
policy rule, and session isolation is not inferred.

Example policy:

```json
{
  "schema_version": "0.1",
  "rules": [
    {
      "dimension": "user_id",
      "authoritative_source_principal": "alice",
      "prohibited_destination_principals": ["bob"]
    }
  ]
}
```

Pass the policy to `palintrace audit` with `--scope-policy` when selecting this checker.

## Unsupported claim

`UnsupportedClaimChecker` assesses whether a memory's complete declared transcript evidence entails
its stored claim. It requires a `TranscriptSet` and an injected `SemanticJudge`. The supported local
judge uses a pinned three-way NLI model.

The checker processes each declared memory as follows:

1. resolve all declared source references;
2. abstain if resolution fails or yields no evidence;
3. compose evidence using the configured fixed composition style;
4. obtain one directional judgment with evidence as premise and memory content as hypothesis;
5. accept `entailment`; and
6. report `neutral` or `contradiction` as unsupported.

Inputs exceeding the model limit are counted as abstentions. Other model failures stop the check
instead of silently changing the relation. Statistics expose assessed and skipped populations plus
the three relation counts.

For the CLI, install the semantic extra and provide both a model ID and immutable revision:

```bash
palintrace audit \
  --store normalized.json \
  --transcripts transcripts.json \
  --checker unsupported_claim \
  --semantic-model-id cross-encoder/nli-MiniLM2-L6-H768 \
  --semantic-model-revision 4b32a82e99875f0bfedc1e20b854db455c540d57 \
  --output findings.json
```

The semantic judge is loaded only when this checker is selected. See [Semantic
checks](semantics.md) for the evidence and model contract.

## Optional identity-grounded candidate

The repository includes an evaluation candidate that prepends an explicitly trusted human-readable
speaker label to evidence from exactly attributed transcript turns. It accepts only explicit
turn-level bindings, abstains on unavailable or conflicting identity, and never infers identity from
roles, scope IDs, provider metadata, transcript prose, or the memory claim.

Current adapters do not automatically provide both exact turn attribution and a trustworthy semantic
speaker label. Callers can construct an explicit source-admission envelope, where:

- `TRUSTED_EXPLICIT` requires a turn, stable principal ID, and speaker label;
- `TRUSTED_CONFIGURED` requires a turn and operator-configured speaker label;
- `UNAVAILABLE` and `AMBIGUOUS` do not compile into bindings; and
- conflicting labels or principal IDs fail closed.

The candidate remains separate from `UnsupportedClaimChecker`, absent from public checker exports,
not selectable through the CLI, and not enabled by default. Its current readiness is
`OPTIONAL_EXPLICIT_API_READY`, not default readiness. See [Semantic checks](semantics.md) and
[Evaluation results](results.md) for the exact evidence and limits.

## Running checks

Select one or more checkers explicitly:

```bash
palintrace audit \
  --store normalized.json \
  --transcripts transcripts.json \
  --checker orphaned_provenance \
  --checker redundancy_bloat \
  --checker stale_active \
  --output findings.json
```

Input requirements are checker-specific. Palintrace rejects missing required inputs rather than
silently omitting a selected check.
