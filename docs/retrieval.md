# Retrieval auditing

MemLint represents and audits retrieval behavior without choosing a production retrieval backend.
The caller supplies the target request and either runs a `Retriever` or provides an already-recorded
observation.

## Audit request

`RetrievalAuditRequest` declares:

- a nonblank request ID;
- the query text;
- one or more expected memory IDs; and
- a positive `top_k`.

Expected targets are explicit audit inputs. They are never passed to the retriever and are not
inferred from returned memories.

## Retriever protocol

A `Retriever` exposes a stable `retriever_id`, `retriever_version`, and:

```python
response = retriever.retrieve(query=request.query, top_k=request.top_k)
```

`run_retrieval_audit` checks the request against a normalized store, calls the retriever without
expected IDs, and then reconciles returned IDs with the audited snapshot. Duplicate, missing,
out-of-store, or inconsistent hits are rejected rather than silently repaired.

## Recorded observations

`RetrievalObservation` stores the request identity, SHA-256 of the query, expected targets,
retriever identity, `top_k`, ranked hits, and usage. It does not store the query text, memory text,
gold mutation metadata, or distractor IDs.

Hits have unique one-based ranks and memory IDs. Their optional scores are finite numbers and are
not interpreted across different retrievers.

## Sufficiency policies

The caller chooses one explicit `RetrievalSufficiencyPolicy`:

| Policy | Sufficient when |
|---|---|
| `all_expected` | Every expected target appears in the recorded top-k hits |
| `any_expected` | At least one expected target appears in the recorded top-k hits |

`assess_retrieval_sufficiency` returns the retrieved, present, and missing expected target sets. It
does not use scores or impose a rank-one requirement.

## Finding projection

`project_retrieval_shadowing_result` converts an insufficient observation into the standard
`CheckerResult` envelope. A sufficient observation produces no finding. An insufficient observation
produces one case-level `retrieval_shadowing` finding over the expected memory IDs, with evidence for
the request hash, policy, retrieved targets, missing targets, and retriever identity.

This projection does not run a retriever and does not claim that all retrieval failures are caused
by distractors. It reports that the explicit target was insufficient under the selected policy.

## Recorded-observation CLI

```bash
memlint retrieval-audit \
  --observation retrieval-observation.json \
  --policy all_expected \
  --output retrieval-findings.json
```

The input observation and output result must be different paths. The command never receives a
mutation manifest or expected answer text.

## Paired challenges

`assess_paired_retrieval_challenge` compares baseline and mutated observations for the same request.
Both observations must agree on query hash, expected targets, `top_k`, and retriever identity.

| Outcome | Baseline | Mutated |
|---|---|---|
| `induced_shadowing` | sufficient | insufficient |
| `resilient` | sufficient | sufficient |
| `baseline_insufficient` | insufficient | either state |

Only baseline-eligible challenges can show induced shadowing. Adding distractors to a store is a
challenge construction step, not proof that shadowing occurred.

## Evaluation retriever

The repository includes a deterministic lexical BM25-style retriever under `memlint.evaluation` for
reproducing controlled experiments. It tokenizes ASCII-alphanumeric terms, uses fixed scoring and
tie ordering, and reads only memory content. It is intentionally absent from `memlint.retrieval` and
the public CLI.

## Limitations

- MemLint does not ship a live production retriever.
- It does not automatically scan a store for retrieval-shadowing defects.
- Relevance targets must come from the audit scenario or caller.
- Observations bind retriever identity and query hash but not a cryptographic digest of every store
  record.
- Synthetic retrieval results are not production prevalence estimates.

See [Evaluation results](results.md) for the strong and matched retrieval experiments.
