from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trustrag.retrieval.bm25 import BM25Chunk, BM25SearchResult
from trustrag.retrieval.vector_store import VectorChunk, VectorSearchResult


class HybridRetrievalError(Exception):
    """Base class for hybrid retrieval failures."""


class HybridConfigError(HybridRetrievalError):
    """Raised when hybrid retrieval configuration is invalid."""


class HybridQueryError(HybridRetrievalError):
    """Raised when a hybrid query is invalid."""


class BM25SearchBackend(Protocol):
    """Search interface required from a BM25 retriever."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[BM25SearchResult]:
        """Return BM25 search results."""


class VectorSearchBackend(Protocol):
    """Search interface required from a vector retriever."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
        minimum_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """Return vector search results."""


class HybridRetrieverConfig(BaseModel):
    """Configuration for reciprocal-rank hybrid retrieval."""

    model_config = ConfigDict(frozen=True)

    bm25_top_k: int = Field(default=20, ge=1)
    vector_top_k: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=1.0, gt=0)
    vector_weight: float = Field(default=1.0, gt=0)
    minimum_score: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_candidate_pool(self) -> Self:
        if self.final_top_k > self.bm25_top_k + self.vector_top_k:
            raise HybridConfigError(
                "final_top_k must not be greater than the total candidate pool"
            )
        return self


class HybridChunk(BaseModel):
    """Common child chunk representation returned by hybrid retrieval."""

    model_config = ConfigDict(frozen=True)

    child_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=1)
    token_count: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_bm25_chunk(cls, chunk: BM25Chunk) -> HybridChunk:
        return cls.model_validate(chunk.model_dump(mode="python"))

    @classmethod
    def from_vector_chunk(cls, chunk: VectorChunk) -> HybridChunk:
        return cls.model_validate(chunk.model_dump(mode="python"))


class HybridSourceScore(BaseModel):
    """Contribution from one retrieval source to a fused result."""

    model_config = ConfigDict(frozen=True)

    source: Literal["bm25", "vector"]
    rank: int = Field(ge=1)
    score: float
    rrf_score: float = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)
    distance: float | None = None


class HybridSearchResult(BaseModel):
    """One result produced by BM25 plus vector fusion."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    score: float = Field(ge=0)
    chunk: HybridChunk
    sources: list[HybridSourceScore]

    @property
    def child_id(self) -> str:
        return self.chunk.child_id

    @property
    def parent_id(self) -> str:
        return self.chunk.parent_id

    @property
    def source_names(self) -> list[str]:
        return [source.source for source in self.sources]


DEFAULT_HYBRID_RETRIEVER_CONFIG = HybridRetrieverConfig()


class HybridRetriever:
    """Fuse BM25 and vector results using Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        bm25_retriever: BM25SearchBackend,
        vector_retriever: VectorSearchBackend,
        config: HybridRetrieverConfig = DEFAULT_HYBRID_RETRIEVER_CONFIG,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.config = config

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        bm25_minimum_score: float | None = None,
        vector_minimum_score: float | None = None,
        vector_where: Mapping[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """Return fused child chunks for a query."""

        final_top_k = self.config.final_top_k if top_k is None else top_k
        if final_top_k < 1:
            raise HybridQueryError("top_k must be at least 1")
        if not query.strip():
            raise HybridQueryError("query must not be empty")

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=self.config.bm25_top_k,
            minimum_score=bm25_minimum_score,
        )
        vector_results = self.vector_retriever.search(
            query,
            top_k=self.config.vector_top_k,
            where=vector_where,
            minimum_score=vector_minimum_score,
        )

        fused_results = _fuse_results(
            bm25_results,
            vector_results,
            config=self.config,
        )
        return fused_results[:final_top_k]

    def parent_ids_for_results(
        self,
        results: Sequence[HybridSearchResult],
    ) -> list[str]:
        """Return parent IDs in result order without duplicates."""

        seen: set[str] = set()
        parent_ids: list[str] = []

        for result in results:
            if result.parent_id in seen:
                continue
            seen.add(result.parent_id)
            parent_ids.append(result.parent_id)

        return parent_ids


def build_hybrid_retriever(
    *,
    bm25_retriever: BM25SearchBackend,
    vector_retriever: VectorSearchBackend,
    config: HybridRetrieverConfig = DEFAULT_HYBRID_RETRIEVER_CONFIG,
) -> HybridRetriever:
    """Build a hybrid retriever from already-built sparse and dense retrievers."""

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        config=config,
    )


@dataclass
class _FusedCandidate:
    chunk: HybridChunk
    score: float = 0.0
    sources: list[HybridSourceScore] = field(default_factory=list)

    @property
    def best_rank(self) -> int:
        return min(source.rank for source in self.sources)


def _fuse_results(
    bm25_results: Sequence[BM25SearchResult],
    vector_results: Sequence[VectorSearchResult],
    *,
    config: HybridRetrieverConfig,
) -> list[HybridSearchResult]:
    candidates: dict[str, _FusedCandidate] = {}

    _add_bm25_results(candidates, bm25_results, config=config)
    _add_vector_results(candidates, vector_results, config=config)

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.score,
            candidate.best_rank,
            candidate.chunk.child_id,
        ),
    )

    results: list[HybridSearchResult] = []
    for candidate in ranked_candidates:
        if config.minimum_score is not None and candidate.score < config.minimum_score:
            continue
        results.append(
            HybridSearchResult(
                rank=len(results) + 1,
                score=candidate.score,
                chunk=candidate.chunk,
                sources=sorted(
                    candidate.sources,
                    key=lambda source: (source.source, source.rank),
                ),
            )
        )

    return results


def _add_bm25_results(
    candidates: dict[str, _FusedCandidate],
    results: Sequence[BM25SearchResult],
    *,
    config: HybridRetrieverConfig,
) -> None:
    seen_child_ids: set[str] = set()

    for result in results:
        if result.child_id in seen_child_ids:
            continue
        seen_child_ids.add(result.child_id)

        rrf_score = _rrf_score(
            rank=result.rank,
            weight=config.bm25_weight,
            rrf_k=config.rrf_k,
        )
        candidate = candidates.setdefault(
            result.child_id,
            _FusedCandidate(chunk=HybridChunk.from_bm25_chunk(result.chunk)),
        )
        candidate.score += rrf_score
        candidate.sources.append(
            HybridSourceScore(
                source="bm25",
                rank=result.rank,
                score=result.score,
                rrf_score=rrf_score,
                matched_terms=list(result.matched_terms),
            )
        )


def _add_vector_results(
    candidates: dict[str, _FusedCandidate],
    results: Sequence[VectorSearchResult],
    *,
    config: HybridRetrieverConfig,
) -> None:
    seen_child_ids: set[str] = set()

    for result in results:
        if result.child_id in seen_child_ids:
            continue
        seen_child_ids.add(result.child_id)

        rrf_score = _rrf_score(
            rank=result.rank,
            weight=config.vector_weight,
            rrf_k=config.rrf_k,
        )
        candidate = candidates.setdefault(
            result.child_id,
            _FusedCandidate(chunk=HybridChunk.from_vector_chunk(result.chunk)),
        )
        candidate.score += rrf_score
        candidate.sources.append(
            HybridSourceScore(
                source="vector",
                rank=result.rank,
                score=result.score,
                rrf_score=rrf_score,
                distance=result.distance,
            )
        )


def _rrf_score(*, rank: int, weight: float, rrf_k: int) -> float:
    return weight / (rrf_k + rank)


__all__ = [
    "BM25SearchBackend",
    "DEFAULT_HYBRID_RETRIEVER_CONFIG",
    "HybridChunk",
    "HybridConfigError",
    "HybridQueryError",
    "HybridRetrievalError",
    "HybridRetriever",
    "HybridRetrieverConfig",
    "HybridSearchResult",
    "HybridSourceScore",
    "VectorSearchBackend",
    "build_hybrid_retriever",
]
