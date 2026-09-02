# Compatibility

This document records the public identity and compatibility rules that apply as Palintrace evolves.
It does not add fields or behavior to the current package.

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

These mappings define stable semantic names for future public rule support. Current result models do
not contain `rule_id` or `rule_version` fields.

## Checker and rule identity

The three identity concepts have separate purposes:

```text
checker_id   = identity of the checker implementation
rule_id      = stable semantic identity of the defect or invariant
rule_version = version of that rule's public semantics
```

For example, the existing `unsupported_claim` checker is mapped conceptually to
`memory.claim.unsupported` at rule version `1.0.0`. This example documents the policy; it is not a
serialized production contract in package version `0.2.0`.

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

For example, package `0.3.0` can emit checker-result schema `0.2`, evaluate
`memory.claim.unsupported@1.0.0`, and use taxonomy `1.0`. A package release does not by itself require
changes to the other version domains.

Serialized formats carry their own schema versions. Compatibility is stated for each format rather
than inferred from the package version. The current checker-result schema version is `0.2`.

The defect taxonomy is independently versioned and is currently `1.0`. Its version changes when the
taxonomy contract changes, not whenever the package changes.

If a public Memory IR is introduced later, it will have an independent version. No Memory IR version
or schema is defined here.

## Deprecation

Published rule IDs are deprecated in place rather than silently renamed or assigned a different
meaning. Their identifiers are not recycled. When a replacement is needed, it receives its own rule
ID and the relationship is documented.

Checker implementations and serialized schemas follow their own compatibility contracts. Any
deprecation affecting them must identify the affected contract and a documented transition; it must
not be implied solely by a package-version change.
