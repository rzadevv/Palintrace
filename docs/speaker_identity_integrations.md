# Speaker identity integration

Identity grounding is safe only when a caller or integration supplies trustworthy speaker
assertions for exact transcript turns. Palintrace does not infer those assertions from conversational
roles, memory scope, provider metadata, transcript prose, or memory claims.

## Trust levels

`SpeakerIdentitySourceAssertion` uses four trust classes:

- `TRUSTED_EXPLICIT`: an integration explicitly attributes an exact turn to a stable principal and
  supplies a human-readable speaker label;
- `TRUSTED_CONFIGURED`: an operator or caller explicitly assigns a speaker label to an exact turn;
- `UNAVAILABLE`: a required coordinate, attribution, or label is missing; and
- `AMBIGUOUS`: the available assertions conflict or do not identify one speaker uniquely.

Only trusted explicit or configured assertions can produce `SpeakerIdentityBindings`. Unavailable or
ambiguous identity never becomes trusted automatically. Conflicting labels or principal IDs for the
same turn fail closed.

## Current adapters

| Integration | Available source information | Automatic binding | Explicit configuration | Main limitation |
|---|---|---:|---:|---|
| File | Memory scope, source references, and a separately loaded `TranscriptSet` | No | Yes | Source references can identify a turn, but scope IDs, roles, and arbitrary metadata do not supply a trusted speaker label. |
| Mem0 | Memory `user_id`, optional `agent_id`, run/application IDs, and custom metadata | No | Yes | Exported memories do not provide the exact memory-to-message-turn mapping and display label required by the current adapter. |
| Letta | Agent/user configuration; message objects may expose `sender_id` or `name`; Identity objects have stable IDs and names | No | Yes | The current adapter reads core blocks and archival passages, not a complete and unique message-to-Identity join. Optional role or name fields are insufficient by themselves. |
| Graphiti | Group IDs, entity edges, episode UUIDs, and optional caller-provided episode mappings | No | Yes | Episodes identify ingestion events rather than structured speaker turns. Graph namespaces and names parsed from episode prose are not trusted identity. |

The field semantics come from the official [Mem0 memory
response](https://docs.mem0.ai/api-reference/memory/get-memory), [Mem0 add
request](https://docs.mem0.ai/api-reference/memory/add-memories), [Letta agent and identity
types](https://docs.letta.com/api/resources/agents), [Letta message retrieval
contract](https://docs.letta.com/api/resources/messages/methods/retrieve), [Graphiti episode
format](https://help.getzep.com/graphiti/core-concepts/adding-episodes), [Graphiti graph
namespacing](https://help.getzep.com/graphiti/core-concepts/graph-namespacing), and [Graphiti
`EntityEdge`](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py).

No live provider was called for this assessment. Adapter evidence comes from source inspection,
fixtures, fake transports, and the documented provider contracts.

## Binding requirements

A source assertion records:

- exact `transcript_id` and `turn_idx`;
- its trust class;
- source-system and source-record references;
- a human-readable `speaker_label` for trusted assertions; and
- a stable `principal_id` when the assertion is `TRUSTED_EXPLICIT`.

`TRUSTED_CONFIGURED` may omit the principal ID because the caller or operator is the authority for
the label. `UNAVAILABLE` and `AMBIGUOUS` assertions do not compile into bindings. Source provenance
stays in the caller-side assertion envelope and is not copied into checker findings.

## Identity resolution

`TranscriptSet` supplies stable transcript IDs and turn indices. A `TranscriptTurn` contains a role,
content, optional timestamp, and arbitrary metadata, but it does not define a portable principal ID
or human-readable speaker label.

These concepts remain separate:

- a role describes conversational function, such as `user`, `assistant`, or `tool`;
- a principal ID identifies a stable application or provider entity; and
- a speaker label is human-readable text used in the semantic premise.

An opaque principal ID can support a reliable join without being a suitable label. A label such as
`Alice` can be useful in a premise without being globally unique. Palintrace therefore retains both
when available and never converts one into the other implicitly.

## Current status

The demonstrated readiness is `OPTIONAL_EXPLICIT_API_READY`. Explicit caller or operator assertions
can be admitted deterministically and compiled into the existing binding contract without identity
inference.

`DEFAULT_READY` has not been established. The current File, Mem0, Letta, and Graphiti adapters do
not automatically provide both exact turn attribution and a trustworthy semantic speaker label.
No production identity-binding prevalence or coverage percentage has been measured.

The identity-grounded unsupported-claim checker remains a separate optional candidate. It is not
part of the public checker exports, is not selectable through the CLI, and is not enabled by
default.
