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

`RedundancyBloatChecker` finds duplicate claims within the same observable normalized scope. The
scope key contains `user_id`, `agent_id`, and `session_id`. Completely unscoped memories are skipped
because their intended boundary is unknown.

Memories are grouped by normalized content and scope. Normalization applies NFKC, case folding,
whitespace collapsing, and trailing-punctuation removal, so `User likes Python`,
`User likes Python.`, and `User likes Python ` fall into one group. Content consisting only of
punctuation is left unfolded so it stays distinguishable.

Each group emits one finding. Evidence reports `match_kind` as `exact` when every member stores
byte-identical content and `normalized` otherwise, alongside the normalized content hash, its
length, and the scope—not the duplicated text. The evidence kind is `exact_duplicate` or
`normalized_duplicate` to match. Statistics split `duplicate_groups` into `exact_duplicate_groups`
and `normalized_duplicate_groups`. This checker does not attempt paraphrase, embedding, or
semantic-equivalence detection.

## Stale active

`StaleActiveChecker` follows explicit `supersedes` links. It reports an older memory when:

1. another memory explicitly names it in `supersedes`; and
2. the older memory has `active: true`.

Resolved links also form a directed graph, from each superseder to the memory it supersedes. A
cyclic component means no record is the unambiguous replacement, so the checker reports the whole
cycle as one relational finding with evidence kind `supersession_cycle` and data listing `members`
and `active_members`, both sorted. A cycle with no active member is not reported. Cycle members do
not additionally receive the per-memory `active_superseded` finding, because the cycle already
describes their state. Memories outside every cycle are unaffected, including one that a cycle
member supersedes and one whose supersession chain leads into a cycle.

Links to absent memories produce no finding and no defect class of their own. They are counted in
`missing_targets_skipped` and listed by ID in the `dangling_supersession_targets` stat, a sorted
list of `superseder_id` and `missing_target_id` pairs. Self-links are skipped and counted;
`NormalizedMemory` rejects them, so they only arise from unvalidated input. The checker does not
infer supersession from dates, wording, or conflicting values.

## Privacy scope violation

`PrivacyScopeViolationChecker` requires an explicit `ScopeIsolationPolicy`. Each policy rule names a
principal dimension (`user_id` or `agent_id`), an authoritative source principal, and prohibited
destination principals.

A destination is reported when its normalized content matches a record under the authoritative
principal. Content normalization is the same as for [redundancy bloat](#redundancy-bloat), so
timestamps, provenance, source references, embeddings, supersession, and active state do not decide
whether a leak is found. The scope dimensions the rule does not name are not part of the match
either: a record leaking from one principal to a prohibited one is reported even when its
`agent_id` or `session_id` also differs. Ordinary cross-scope differences are still not violations
without a policy rule, and session isolation is not inferred.

Evidence uses kind `prohibited_scope_replica` and reports the authoritative memory ID, the scope
dimension and both principals, the normalized content hash, `match_kind` (`exact` or `normalized`),
and `differing_fields`. That list names the portable fields whose values differ, excluding the
memory ID and the rule's own dimension; differing scope dimensions appear as `scope.agent_id` or
`scope.session_id`. Field names are reported, never their values.

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

Select one checker explicitly:

```bash
palintrace audit \
  --store normalized.json \
  --transcripts transcripts.json \
  --checker orphaned_provenance \
  --output findings.json
```

Input requirements are checker-specific. Palintrace rejects missing required inputs rather than
silently omitting a selected check.
