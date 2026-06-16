from collections.abc import Mapping
from typing import Any

import pytest

from trustrag.retrieval.bm25 import BM25Chunk, BM25SearchResult
from trustrag.retrieval.hybrid import (
    HybridConfigError,
    HybridQueryError,
    HybridRetriever,
    HybridRetrieverConfig,
    build_hybrid_retriever,
)
from trustrag.retrieval.vector_store import VectorChunk, VectorSearchResult


class _StaticBM25Retriever:
    def __init__(self, results: list[BM25SearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[BM25SearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "minimum_score": minimum_score,
            }
        )
        return self.results[:top_k]


class _StaticVectorRetriever:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
        minimum_score: float | None = None,
    ) -> list[VectorSearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "where": where,
                "minimum_score": minimum_score,
            }
        )
        return self.results[:top_k]


def _bm25_chunk(
    child_id: str,
    text: str,
    *,
    parent_id: str | None = None,
) -> BM25Chunk:
    return BM25Chunk(
        child_id=child_id,
        parent_id=parent_id or "parent_1",
        source_id="source_a",
        source_group="sebi_investor_education",
        title="Test Document",
        text=text,
        page_start=1,
        page_end=1,
        token_start=0,
        token_end=max(1, len(text.split())),
        token_count=max(1, len(text.split())),
        metadata={"authority": "SEBI"},
    )


def _vector_chunk(
    child_id: str,
    text: str,
    *,
    parent_id: str | None = None,
) -> VectorChunk:
    return VectorChunk(
        child_id=child_id,
        parent_id=parent_id or "parent_1",
        source_id="source_a",
        source_group="sebi_investor_education",
        title="Test Document",
        text=text,
        page_start=1,
        page_end=1,
        token_start=0,
        token_end=max(1, len(text.split())),
        token_count=max(1, len(text.split())),
        metadata={"authority": "SEBI"},
    )


def _bm25_result(
    child_id: str,
    rank: int,
    *,
    parent_id: str | None = None,
) -> BM25SearchResult:
    return BM25SearchResult(
        rank=rank,
        score=10.0 / rank,
        chunk=_bm25_chunk(
            child_id,
            f"BM25 text for {child_id}",
            parent_id=parent_id,
        ),
        matched_terms=["kyc"],
    )


def _vector_result(
    child_id: str,
    rank: int,
    *,
    parent_id: str | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        rank=rank,
        score=1.0 / rank,
        distance=rank * 0.1,
        chunk=_vector_chunk(
            child_id,
            f"Vector text for {child_id}",
            parent_id=parent_id,
        ),
    )


def test_hybrid_retriever_fuses_bm25_and_vector_ranks_with_rrf() -> None:
    bm25 = _StaticBM25Retriever(
        [
            _bm25_result("c1", 1),
            _bm25_result("c2", 2),
        ]
    )
    vector = _StaticVectorRetriever(
        [
            _vector_result("c2", 1),
            _vector_result("c3", 2),
        ]
    )
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        config=HybridRetrieverConfig(
            bm25_top_k=10,
            vector_top_k=10,
            final_top_k=3,
            rrf_k=1,
        ),
    )

    results = retriever.search("KYC process")

    assert [result.child_id for result in results] == ["c2", "c1", "c3"]
    assert results[0].score == pytest.approx((1 / 3) + (1 / 2))
    assert results[0].source_names == ["bm25", "vector"]
    assert results[0].sources[0].matched_terms == ["kyc"]
    assert results[0].sources[1].distance == pytest.approx(0.1)


def test_hybrid_retriever_applies_top_k_and_passes_source_options() -> None:
    bm25 = _StaticBM25Retriever(
        [
            _bm25_result("c1", 1),
            _bm25_result("c2", 2),
        ]
    )
    vector = _StaticVectorRetriever(
        [
            _vector_result("c3", 1),
            _vector_result("c4", 2),
        ]
    )
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        config=HybridRetrieverConfig(
            bm25_top_k=2,
            vector_top_k=2,
            final_top_k=3,
        ),
    )

    results = retriever.search(
        "KYC process",
        top_k=2,
        bm25_minimum_score=0.2,
        vector_minimum_score=0.3,
        vector_where={"source_group": "sebi_investor_education"},
    )

    assert len(results) == 2
    assert bm25.calls == [
        {
            "query": "KYC process",
            "top_k": 2,
            "minimum_score": 0.2,
        }
    ]
    assert vector.calls == [
        {
            "query": "KYC process",
            "top_k": 2,
            "where": {"source_group": "sebi_investor_education"},
            "minimum_score": 0.3,
        }
    ]


def test_hybrid_retriever_filters_by_minimum_rrf_score() -> None:
    retriever = HybridRetriever(
        bm25_retriever=_StaticBM25Retriever([_bm25_result("c1", 1)]),
        vector_retriever=_StaticVectorRetriever([_vector_result("c2", 1)]),
        config=HybridRetrieverConfig(
            bm25_top_k=1,
            vector_top_k=1,
            final_top_k=2,
            rrf_k=1,
            minimum_score=0.6,
        ),
    )

    assert retriever.search("KYC") == []


def test_parent_ids_for_results_deduplicates_in_rank_order() -> None:
    retriever = HybridRetriever(
        bm25_retriever=_StaticBM25Retriever(
            [
                _bm25_result("c1", 1, parent_id="p1"),
                _bm25_result("c2", 2, parent_id="p1"),
            ]
        ),
        vector_retriever=_StaticVectorRetriever(
            [_vector_result("c3", 1, parent_id="p2")]
        ),
        config=HybridRetrieverConfig(
            bm25_top_k=2,
            vector_top_k=1,
            final_top_k=3,
        ),
    )

    results = retriever.search("KYC")

    assert retriever.parent_ids_for_results(results) == ["p1", "p2"]


def test_hybrid_search_rejects_empty_query_and_invalid_top_k() -> None:
    retriever = build_hybrid_retriever(
        bm25_retriever=_StaticBM25Retriever([]),
        vector_retriever=_StaticVectorRetriever([]),
        config=HybridRetrieverConfig(
            bm25_top_k=1,
            vector_top_k=1,
            final_top_k=1,
        ),
    )

    with pytest.raises(HybridQueryError, match="query must not be empty"):
        retriever.search(" ")

    with pytest.raises(HybridQueryError, match="top_k"):
        retriever.search("KYC", top_k=0)


def test_hybrid_config_rejects_impossible_final_top_k() -> None:
    with pytest.raises(HybridConfigError, match="final_top_k"):
        HybridRetrieverConfig(
            bm25_top_k=1,
            vector_top_k=1,
            final_top_k=3,
        )
