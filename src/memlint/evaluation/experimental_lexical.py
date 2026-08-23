"""Frozen evaluation-only lexical retrieval baseline for benchmark v0.1."""

from __future__ import annotations

import math
import re
from collections import Counter

from memlint.models import NormalizedStore
from memlint.retrieval import (
    RetrievalHit,
    RetrievalInputError,
    RetrievalResponse,
    RetrievalUsage,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
BM25_K1 = 1.2
BM25_B = 0.75


def tokenize_lexical(text: str) -> tuple[str, ...]:
    """Apply the exact benchmark v0.1 ASCII-alphanumeric tokenizer."""

    if not isinstance(text, str):
        raise TypeError("lexical tokenizer input must be a string")
    return tuple(match.group(0).lower() for match in TOKEN_RE.finditer(text))


class ExperimentalLexicalRetriever:
    """Deterministic BM25-style baseline over normalized memory content only."""

    retriever_id = "experimental_lexical"
    retriever_version = "0.1"

    def __init__(self, store: NormalizedStore) -> None:
        if not isinstance(store, NormalizedStore):
            raise RetrievalInputError("experimental lexical retrieval requires a NormalizedStore")
        self._store = store
        self._document_tokens = {
            memory.id: tokenize_lexical(memory.content) for memory in store.memories
        }

    def retrieve(self, *, query: str, top_k: int) -> RetrievalResponse:
        """Rank positive-scoring memory contents using the frozen BM25 formula."""

        if not isinstance(query, str) or not query.strip():
            raise RetrievalInputError("lexical retrieval query must be a nonblank string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise RetrievalInputError("lexical retrieval top_k must be a positive integer")

        candidate_count = len(self._store.memories)
        usage = RetrievalUsage(retrieval_calls=1, candidate_count=candidate_count)
        if candidate_count == 0:
            return RetrievalResponse(hits=(), usage=usage)

        query_tokens = tuple(sorted(set(tokenize_lexical(query))))
        if not query_tokens:
            return RetrievalResponse(hits=(), usage=usage)

        document_lengths = {
            memory_id: len(tokens) for memory_id, tokens in self._document_tokens.items()
        }
        avgdl = sum(document_lengths.values()) / candidate_count
        if avgdl == 0:
            return RetrievalResponse(hits=(), usage=usage)

        frequencies = {
            memory_id: Counter(tokens)
            for memory_id, tokens in self._document_tokens.items()
        }
        document_frequencies = {
            token: sum(token in frequency for frequency in frequencies.values())
            for token in query_tokens
        }

        scored: list[tuple[str, float]] = []
        for memory_id, frequency in frequencies.items():
            document_length = document_lengths[memory_id]
            score = 0.0
            for token in query_tokens:
                term_frequency = frequency[token]
                if term_frequency == 0:
                    continue
                document_frequency = document_frequencies[token]
                inverse_document_frequency = math.log(
                    1
                    + (candidate_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = term_frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * document_length / avgdl
                )
                score += inverse_document_frequency * (
                    term_frequency * (BM25_K1 + 1) / denominator
                )
            if score > 0:
                if not math.isfinite(score):  # pragma: no cover - finite inputs guarantee this
                    raise RetrievalInputError("lexical retrieval produced a nonfinite score")
                scored.append((memory_id, score))

        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return RetrievalResponse(
            hits=tuple(
                RetrievalHit(memory_id=memory_id, rank=rank, score=score)
                for rank, (memory_id, score) in enumerate(ranked, start=1)
            ),
            usage=usage,
        )
