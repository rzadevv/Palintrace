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

`retrieval_shadowing` is reported by `palintrace retrieval-audit` from a recorded retrieval
observation rather than by store inspection; see [Retrieval auditing](retrieval.md).
`internal_contradiction` and `injected_instruction` have no checker.

## Rule identifiers

Each result carries a stable `rule_id` of the form `memory.<area>.<defect>`, a numeric `rule_version`,
and a default severity:

| `checker_id` | `rule_id` | `rule_version` | Severity |
|---|---|---|---|
| `orphaned_provenance` | `memory.provenance.orphaned` | `1.0.0` | `error` |
| `redundancy_bloat` | `memory.duplication.equivalent-content` | `1.0.0` | `warning` |
| `stale_active` | `memory.state.explicit-stale` | `2.0.0` | `error` |
| `privacy_scope_violation` | `memory.scope.prohibited-replica` | `1.0.0` | `error` |
| `unsupported_claim` | `memory.claim.unsupported` | `1.0.0` | `error` |
| `retrieval_shadowing` | `memory.retrieval.shadowing` | `1.0.0` | `error` |

`checker_id` names the implementation, `rule_id` names the defect it reports, and `rule_version`
versions the rule's meaning. A custom checker outside this table must supply all three rule fields
explicitly.

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
source coordinates instead. Mutation manifests and gold labels are never checker inputs.

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

Run every checker in one pass with `--checker all`. Checkers whose required inputs are missing are
reported as skipped rather than failing the run. `palintrace preflight` reports which checkers a
store can support before running them.

## Adapter capabilities

`palintrace capabilities --adapter <name>` prints which normalized fields an adapter can populate.
Each field has one status:

- `supported`: the adapter has a direct normalized mapping;
- `conditional`: an optional source field or caller configuration is required; and
- `unsupported`: the adapter has no normalized mapping.

Status describes what the adapter can map, not whether every record carries a value.

## Exit status and SARIF

`audit` and `retrieval-audit` accept `--fail-on info|warning|error`. Severity is ordered
`info < warning < error`.

| Status | Meaning |
|---|---|
| `0` | The command succeeded and no configured gate failed |
| `1` | The audit succeeded, but findings met the `--fail-on` threshold |
| `2` | An argument, input, or configuration error occurred |

Without `--fail-on`, findings do not change the exit status. Outputs are written before the exit
status is decided.

`--sarif-output` writes a SARIF `2.1.0` rendering alongside the JSON result. Severity maps `info` to
`note`, `warning` to `warning`, and `error` to `error`.
