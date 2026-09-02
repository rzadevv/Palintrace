from __future__ import annotations

import inspect
import math
from collections import Counter
from pathlib import Path

from palintrace.evaluation import (
    BM25_B,
    BM25_K1,
    TOKEN_RE,
    ExperimentalLexicalRetriever,
    tokenize_lexical,
)
from palintrace.models import MemoryScope, NormalizedMemory, NormalizedStore


def _store(*memories: NormalizedMemory) -> NormalizedStore:
    return NormalizedStore(
        schema_version="0.1",
        adapter="toy-lexical",
        memories=memories,
    )


def test_exact_tokenizer_constants_case_and_punctuation() -> None:
    assert TOKEN_RE.pattern == r"[A-Za-z0-9]+"
    assert BM25_K1 == 1.2
    assert BM25_B == 0.75
    assert tokenize_lexical("Hello, WORLD! café C3-PO _x") == (
        "hello",
        "world",
        "caf",
        "c3",
        "po",
        "x",
    )


def test_exact_known_bm25_score_uses_distinct_query_tokens() -> None:
    store = _store(
        NormalizedMemory(id="m1", content="alpha alpha beta"),
        NormalizedMemory(id="m2", content="alpha gamma"),
        NormalizedMemory(id="m3", content="delta"),
    )
    response = ExperimentalLexicalRetriever(store).retrieve(
        query="Alpha beta alpha",
        top_k=3,
    )
    scores = {hit.memory_id: hit.score for hit in response.hits}
    assert scores["m1"] is not None

    candidate_count = 3
    average_length = 2.0
    document_length = 3
    alpha_idf = math.log(1 + (candidate_count - 2 + 0.5) / (2 + 0.5))
    beta_idf = math.log(1 + (candidate_count - 1 + 0.5) / (1 + 0.5))
    normalization = BM25_K1 * (
        1 - BM25_B + BM25_B * document_length / average_length
    )
    expected = alpha_idf * (2 * (BM25_K1 + 1) / (2 + normalization))
    expected += beta_idf * (1 * (BM25_K1 + 1) / (1 + normalization))
    assert math.isclose(scores["m1"], expected, rel_tol=1e-12)


def test_ranking_tie_break_positive_only_and_fewer_than_top_k() -> None:
    store = _store(
        NormalizedMemory(id="zeta", content="shared token"),
        NormalizedMemory(id="alpha", content="shared token"),
        NormalizedMemory(id="zero", content="unrelated words"),
    )
    response = ExperimentalLexicalRetriever(store).retrieve(
        query="SHARED",
        top_k=5,
    )
    assert tuple((hit.memory_id, hit.rank) for hit in response.hits) == (
        ("alpha", 1),
        ("zeta", 2),
    )
    assert all(hit.score is not None and hit.score > 0 for hit in response.hits)
    assert response.usage.retrieval_calls == 1
    assert response.usage.candidate_count == 3


def test_empty_store_and_zero_token_query_return_no_hits() -> None:
    empty_response = ExperimentalLexicalRetriever(_store()).retrieve(
        query="alpha",
        top_k=3,
    )
    assert empty_response.hits == ()
    assert empty_response.usage.retrieval_calls == 1
    assert empty_response.usage.candidate_count == 0

    punctuation_response = ExperimentalLexicalRetriever(
        _store(NormalizedMemory(id="m1", content="alpha"))
    ).retrieve(query="--- !!!", top_k=3)
    assert punctuation_response.hits == ()
    assert punctuation_response.usage.candidate_count == 1


def test_retrieval_is_deterministic_and_scores_are_finite() -> None:
    store = _store(
        NormalizedMemory(id="m1", content="blue blue circle"),
        NormalizedMemory(id="m2", content="blue square"),
    )
    retriever = ExperimentalLexicalRetriever(store)
    first = retriever.retrieve(query="blue circle", top_k=2)
    second = retriever.retrieve(query="blue circle", top_k=2)
    assert first == second
    assert all(hit.score is not None and math.isfinite(hit.score) for hit in first.hits)


def test_only_memory_content_affects_scoring() -> None:
    store = _store(
        NormalizedMemory(
            id="a",
            content="same words",
            scope=MemoryScope(user_id="query-only-in-scope"),
            embedding=(99.0,),
            raw={"query": "same same same same"},
        ),
        NormalizedMemory(
            id="b",
            content="same words",
            scope=MemoryScope(agent_id="different"),
            embedding=(0.0,),
            raw={"backend": "different"},
        ),
    )
    response = ExperimentalLexicalRetriever(store).retrieve(query="same", top_k=2)
    assert tuple(hit.memory_id for hit in response.hits) == ("a", "b")
    assert response.hits[0].score == response.hits[1].score


def test_retriever_boundary_is_target_blind_and_has_no_external_methodology() -> None:
    signature = inspect.signature(ExperimentalLexicalRetriever.retrieve)
    assert tuple(signature.parameters) == ("self", "query", "top_k")
    source = Path("src/palintrace/evaluation/experimental_lexical.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "expected_memory_ids",
        "distractor",
        "mutation_id",
        "gold_label",
        "embedding",
        "requests",
        "http",
        "transformers",
        "torch",
        "stopword",
        "stemming",
        "synonym",
    )
    assert not any(term in source.lower() for term in forbidden)
    assert Counter(tokenize_lexical("repeat repeat")) == {"repeat": 2}
