# Trusted speaker-binding feasibility

Part 6G-C supports the frozen identity-grounded representation on an independently constructed
synthetic held-out set when correct explicit speaker bindings are supplied. Part 6G-D audits the
separate integration assumption behind those bindings. It does not run a semantic model, estimate
production coverage, or promote the candidate checker.

## Trust classification

Identity-source assessments use exactly four trust classes:

- `TRUSTED_EXPLICIT`: a documented integration field explicitly attributes one exact transcript
  turn to a stable principal and supplies a human-readable speaker label;
- `TRUSTED_CONFIGURED`: an operator or caller explicitly configures one exact transcript turn and
  its human-readable speaker label; a stable principal ID may also be supplied;
- `UNAVAILABLE`: at least one required coordinate, attribution, or speaker label is absent;
- `AMBIGUOUS`: available assertions disagree or do not identify one unique principal/label.

Only the first two classes may produce `SpeakerIdentityBindings`. A backend field is not trusted
because its name resembles an identity. `role=user`, memory scope IDs, arbitrary metadata, raw
backend fields, transcript prose, and claim text never cross this admission boundary by themselves.

## Source-of-truth matrix

| Integration | Available identity fields | Turn attribution | Human label | Trust in the current path | Automatic binding | Explicit configuration | Important caveat |
|---|---|---|---|---|---:|---:|---|
| File | Memory `scope.user_id`/`agent_id`/`session_id`; explicit `source_refs`; separately loaded `TranscriptSet` | `SourceRef` can name a turn, but `TranscriptTurn` has only role/content/time/arbitrary metadata | None in the portable schema | `UNAVAILABLE` unless the caller supplies a separate exact binding | No | Yes | File input is controlled and therefore the simplest configured path, but scope IDs and turn roles are not labels. Unknown file metadata is retained under `raw` and is not admitted. |
| Mem0 | Documented memory `user_id`, optional `agent_id`, `run_id`, `app_id`; adapter normalizes user/agent/run into memory scope | The documented memory response has no source-message or turn coordinate. Add requests contain ordered role/content messages, but exported memories do not preserve a documented memory-to-message-turn mapping used by this adapter. | No documented display label in the memory response | `UNAVAILABLE` for automatic binding | No | Yes | Entity IDs scope memories; they do not prove which named person spoke a particular source turn. Custom metadata and fixture-only `source_refs` are not provider guarantees. |
| Letta | Current adapter configuration has `agent_id`/optional `user_id`; official message objects can expose optional `sender_id`, `agent_id`, and participant `name`; official Identity objects have stable IDs/identifier keys and names | Official messages are individually identified and may carry `sender_id`, but the current adapter fetches only core blocks and archival passages and does not ingest messages or identities | Official message `name` is optional; Identity `name` exists when an Identity is explicitly joined | Current path `UNAVAILABLE`; a future documented message/Identity join can be `TRUSTED_EXPLICIT` when unique and complete | No | Yes | `sender_id` may denote an Identity or agent and is optional. `name` is optional. Role alone is insufficient. Any future provider must resolve the documented sender type and reject missing or conflicting joins. |
| Graphiti | EntityEdge `group_id`, entity endpoints, fact, and episode UUIDs; explicit caller `episode_transcript_map` can create `SourceRef` values | Edge episodes identify ingestion events, not structured speaker turns. Graphiti message episodes encode `{role/name}: message` pairs in episode text. | No structured speaker label on the exported EntityEdge | `UNAVAILABLE` for automatic binding | No | Yes | `group_id` is a graph namespace, not a principal. Episode name/source description are ingestion metadata. Parsing names from episode prose or trusting LLM-extracted entities/facts would violate the no-inference rule. |

The official contracts supporting this audit are the [Mem0 memory response](https://docs.mem0.ai/api-reference/memory/get-memory),
[Mem0 add request](https://docs.mem0.ai/api-reference/memory/add-memories),
[Letta agent/message/Identity types](https://docs.letta.com/api/resources/agents),
[Letta message retrieval contract](https://docs.letta.com/api/resources/messages/methods/retrieve),
[Graphiti episode format](https://help.getzep.com/graphiti/core-concepts/adding-episodes),
[Graphiti graph namespacing](https://help.getzep.com/graphiti/core-concepts/graph-namespacing),
and the official [Graphiti EntityEdge source](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py).
The optional provider SDKs were not installed in the repository environment, and no live provider
was called. Local adapter evidence therefore consists of source inspection, fixtures, and
fake-transport tests; official documentation supplies field semantics, not live integration
validation.

## Normalized transcript boundary

`TranscriptSet` supplies stable transcript IDs and turn indices. `TranscriptTurn` supplies a
conversational role, content, optional timestamp, and arbitrary metadata. It does not define a
portable principal ID or speaker label. The identity resolver correctly ignores role, content, and
metadata. Adding suggestive metadata to a turn therefore cannot make it trusted without a separate
admission contract owned by the integration/caller.

This separation is intentional:

- role describes conversational function, such as `user`, `assistant`, or `tool`;
- principal ID identifies a stable backend/operator entity;
- speaker label is human-readable text suitable for the frozen semantic premise.

These values may coincide in a particular application, but MemLint must not assume that they do.

## Principal ID versus semantic label

Principal IDs and speaker labels must remain distinct. An opaque identifier such as
`usr_01H...` can be stable and useful for joining records while being a poor natural-language label.
Conversely, `Alice` can be a useful label while being non-unique. The integration boundary should
retain both whenever a stable principal is available, record how the assertion was obtained, and
emit only the label into the existing frozen `SpeakerIdentityBinding` consumed by semantics.

The frozen candidate and identity resolver do not need principal IDs and must remain unchanged.
Principal identity and admission provenance belong in a separate caller/integration envelope that
is compiled, fail-closed, to the existing binding set.

## Minimum safe integration boundary

The smallest safe future boundary preserves constructor-supplied `SpeakerIdentityBindings`. An
integration or operator first supplies a separate deterministic source assertion containing:

- exact `transcript_id` and `turn_idx`;
- trust class;
- stable source-system and source-record references;
- optional stable `principal_id`;
- required human-readable `speaker_label` for trusted assertions.

`TRUSTED_EXPLICIT` additionally requires a principal ID. `TRUSTED_CONFIGURED` may directly bind a
turn to a label when the operator is the authority. `UNAVAILABLE` and `AMBIGUOUS` never compile.
Conflicting trusted labels or principals for the same turn fail closed. The compiled object is the
unchanged version `0.1` `SpeakerIdentityBindings`; source provenance is retained by the caller-side
envelope and is not copied into checker findings.

No backend-specific automatic mapper is justified by the current evidence. A future Letta message
provider is plausible, but it requires a separately reviewed implementation and tests against the
documented message/Identity join. Mem0 and Graphiti would require their surrounding applications to
retain explicit source-turn identity assertions at ingestion time. File users can supply the
assertion envelope directly.

## Promotion rule frozen before implementation tests

Part 6G-D uses exactly three readiness states:

- `DEVELOPMENT_ONLY`: the explicit admission boundary is not deterministic/fail-closed, controlled
  trust/conflict/privacy tests fail, or a frozen predecessor/public behavior changes;
- `OPTIONAL_EXPLICIT_API_READY`: the frozen semantic evidence remains valid, explicit caller input
  can be admitted deterministically with source provenance and without inference, all controlled
  and regression tests pass, but automatic adapter availability or production prevalence remains
  unestablished;
- `DEFAULT_READY`: the optional criteria pass and trustworthy automatic bindings, measured
  real-world coverage, and production integration validation are established across the intended
  default population.

The decision is mechanical: choose the highest state whose complete conditions are demonstrated.
Synthetic 6G-C coverage is not evidence for the `DEFAULT_READY` availability requirements and will
not be reused as a real-world percentage.

## Part 6G-D feasibility result

The source-admission contract passed controlled tests for explicit principal/label input,
configured label-only authority, missing labels, transcript-level-only identity, ambiguous input,
multiple users, user/assistant turns, repeated turns from one principal, conflicting labels,
conflicting principals, deterministic serialization, and mixed-speaker resolution. Separate tests
against every current adapter normalizer confirmed the matrix above. No semantic judge was
constructed or called.

The highest demonstrated readiness state is:

```text
OPTIONAL_EXPLICIT_API_READY
```

The existing caller-supplied binding design is technically sufficient when a trusted integration
or operator explicitly provides the source assertions. It is not automatically satisfied by File,
Mem0, Letta, or Graphiti today. No production prevalence dataset was evaluated, so automatic
coverage has no percentage. The candidate remains nonpublic, non-CLI, and nondefault. The
`DEFAULT_READY` conditions are not met.
