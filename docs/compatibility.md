# Compatibility

This document records the public identity and compatibility rules that apply as Palintrace evolves.
Package version `0.3.0` implements the first result-level rule identity contract.

## Package versions

The Python package follows Semantic Versioning. Before 1.0:

- Patch releases, such as `0.3.0` to `0.3.1`, are for bug fixes, documentation corrections, and
  internal changes that do not intentionally alter the public contract.
- Minor releases, such as `0.3.0` to `0.4.0`, are for new public checker or rule support, adapters,
  CLI or report capabilities, intentional API evolution, and clearly documented incompatible
  pre-1.0 contract changes.

Version `1.0.0` is reserved for the point at which Palintrace declares its core public compatibility
expectations stable. After 1.0, releases follow the usual Semantic Versioning compatibility rules.

The `v0.1.0` and `v0.2.0` tags are immutable historical releases. The first release planned under
this compatibility policy is `v0.3.0`; this policy does not itself create or move that tag.

## Public rule identifiers

Public semantic rules use lowercase, dot-separated identifiers in this form:

```text
memory.<area>.<defect>
```

A rule ID:

1. never contains the product name;
2. never depends on Python class or module names;
3. describes the defect or invariant rather than its implementation;
4. is not silently renamed or reused after public release;
5. remains reserved if its rule is deprecated; and
6. may be evaluated by more than one implementation.

The canonical rule mapping for the five supported public static checkers is:

| Current `checker_id` | Canonical `rule_id` |
|---|---|
| `orphaned_provenance` | `memory.provenance.orphaned` |
| `redundancy_bloat` | `memory.duplication.exact` |
| `stale_active` | `memory.state.explicit-stale` |
| `privacy_scope_violation` | `memory.scope.prohibited-exact-replica` |
| `unsupported_claim` | `memory.claim.unsupported` |

Recorded retrieval results use `retrieval_shadowing` with `memory.retrieval.shadowing`. This is a
projection of a recorded observation, not a sixth static production checker.

The experimental `unsupported_claim_identity_grounded` checker also uses
`memory.claim.unsupported`. It is an alternate implementation of the same semantic rule and remains
nondefault.

## Checker and rule identity

The three identity concepts have separate purposes:

```text
checker_id   = identity of the checker implementation
rule_id      = stable semantic identity of the defect or invariant
rule_version = version of that rule's public semantics
```

For example, the existing `unsupported_claim` checker maps to `memory.claim.unsupported` at rule
version `1.0.0`. Since checker-result schema `0.3`, `CheckerResult` serializes `rule_id`,
`rule_version`, and `severity` once on the result envelope.

Existing checker IDs participate in code, tests, result identities, evaluation, and frozen research
artifacts. They remain unchanged. A material change to a rule's public meaning uses a distinct
`rule_version`, while implementation-only changes may retain the same rule version.

## Independent version domains

The following versions serve different contracts and are not required to match:

```text
Palintrace package version
serialized result or schema version
rule version
taxonomy version
future Memory IR version
```

For example, package `0.3.0` can emit checker-result schema `0.3`, evaluate
`memory.claim.unsupported@1.0.0`, and use taxonomy `1.0`. A package release does not by itself require
changes to the other version domains.

Serialized formats carry their own schema versions. Compatibility is stated for each format rather
than inferred from the package version. The current checker-result schema version is `0.3`.

The defect taxonomy is independently versioned and is currently `1.0`. Its version changes when the
taxonomy contract changes, not whenever the package changes.

If a public Memory IR is introduced later, it will have an independent version. No Memory IR version
or schema is defined here.

## CLI exit status

The `audit` and `retrieval-audit` commands support an optional `--fail-on` threshold with severity
ordering `info < warning < error`.

| Status | Meaning |
|---|---|
| `0` | The command succeeded and no configured gate failed |
| `1` | The audit succeeded, but findings met the configured severity threshold |
| `2` | An argument, input, or configuration error occurred |

When `--fail-on` is omitted, findings do not change a successful exit status. Gating occurs after
the result is serialized and does not modify the `CheckerResult` content.

## SARIF reporting

Checker-result JSON schema `0.3` remains the canonical Palintrace result. `--sarif-output` writes a
deterministic derived representation using SARIF `2.1.0`; it does not replace or modify the canonical
result.

SARIF rule IDs come from `rule_id`. Severity maps as follows:

| Palintrace severity | SARIF level |
|---|---|
| `info` | `note` |
| `warning` | `warning` |
| `error` | `error` |

SARIF carries the existing finding ID as its fingerprint. Rendering does not change finding identity
or `--fail-on` behavior.

## Deprecation

Published rule IDs are deprecated in place rather than silently renamed or assigned a different
meaning. Their identifiers are not recycled. When a replacement is needed, it receives its own rule
ID and the relationship is documented.

Checker implementations and serialized schemas follow their own compatibility contracts. Any
deprecation affecting them must identify the affected contract and a documented transition; it must
not be implied solely by a package-version change.
