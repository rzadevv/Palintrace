# Retrieval auditing

Palintrace audits retrieval behavior from recorded observations. It does not run a retriever: the
caller runs their own retrieval, records the result as a `RetrievalObservation` JSON file, and passes
it to `palintrace retrieval-audit`.

## Recorded observations

`RetrievalObservation` stores a nonblank request ID, the lowercase hex SHA-256 of the query, the
expected target memory IDs, the retriever identity, `top_k`, ranked hits, and usage. It does not
store the query text or memory text.

```json
{
  "request_id": "editor-preference",
  "query_sha256": "<sha256 of the query text>",
  "expected_memory_ids": ["editor-neovim"],
  "top_k": 3,
  "retriever_id": "my-retriever",
  "retriever_version": "1",
  "hits": [{"memory_id": "editor-vscode", "rank": 1, "score": 0.82}],
  "usage": {"retrieval_calls": 1, "candidate_count": 1}
}
```

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
palintrace retrieval-audit \
  --observation retrieval-observation.json \
  --policy all_expected \
  --output retrieval-findings.json
```

The input observation and output result must be different paths. The command never receives a
mutation manifest or expected answer text. It also accepts the optional `--fail-on` threshold used
by static audits, for example `--fail-on error`. Use `--sarif-output PATH` to write a SARIF projection
alongside the canonical retrieval result.

## Limitations

- Palintrace does not ship a live production retriever.
- It does not automatically scan a store for retrieval-shadowing defects.
- Relevance targets must come from the audit scenario or caller.
- Observations bind retriever identity and query hash but not a cryptographic digest of every store
  record.
