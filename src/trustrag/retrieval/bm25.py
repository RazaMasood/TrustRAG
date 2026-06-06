from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rank_bm25 import BM25Okapi

from trustrag.ingestion.chunking import ChildChunk

LOGGER = logging.getLogger(__name__)

DEFAULT_TOKEN_PATTERN = r"(?u)\b[\w][\w.-]*\b"
DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


class BM25Error(Exception):
    """Base class for BM25 retrieval failures."""


class BM25ConfigError(BM25Error):
    """Raised when BM25 configuration is invalid."""


class BM25IndexError(BM25Error):
    """Raised when a BM25 index cannot be built."""


class BM25QueryError(BM25Error):
    """Raised when a BM25 query is invalid."""


class BM25LoadError(BM25Error):
    """Raised when child chunks cannot be loaded from disk."""


class BM25Config(BaseModel):
    """Configuration for BM25 tokenization and ranking."""

    model_config = ConfigDict(frozen=True)

    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    epsilon: float = Field(default=0.25, ge=0)
    token_pattern: str = DEFAULT_TOKEN_PATTERN
    lowercase: bool = True
    remove_stopwords: bool = True
    stopwords: frozenset[str] = Field(default_factory=lambda: DEFAULT_STOPWORDS)
    min_token_length: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _validate_token_pattern(self) -> Self:
        try:
            re.compile(self.token_pattern)
        except re.error as exc:
            raise BM25ConfigError(f"Invalid BM25 token pattern: {exc}") from exc
        return self


class BM25Chunk(BaseModel):
    """Searchable representation of one child chunk."""

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

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be greater than token_start")
        return self

    @classmethod
    def from_child_chunk(cls, chunk: ChildChunk) -> BM25Chunk:
        return cls.model_validate(chunk.model_dump(mode="python"))


class BM25SearchResult(BaseModel):
    """One scored BM25 retrieval result."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    score: float
    chunk: BM25Chunk
    matched_terms: list[str]

    @property
    def child_id(self) -> str:
        return self.chunk.child_id

    @property
    def parent_id(self) -> str:
        return self.chunk.parent_id


class BM25IndexStats(BaseModel):
    """Small diagnostic summary for a BM25 index."""

    model_config = ConfigDict(frozen=True)

    chunk_count: int
    average_token_count: float
    unique_source_count: int
    unique_parent_count: int


DEFAULT_BM25_CONFIG = BM25Config()


class BM25Retriever:
    """In-memory BM25 retriever over child chunks."""

    def __init__(
        self,
        chunks: Sequence[BM25Chunk],
        *,
        config: BM25Config = DEFAULT_BM25_CONFIG,
    ) -> None:
        if not chunks:
            raise BM25IndexError("Cannot build BM25 index without chunks")

        self.config = config
        self._chunks = tuple(chunks)
        self._chunk_by_id = _index_chunks_by_id(self._chunks)
        self._tokenized_corpus = [tokenize(chunk.text, config) for chunk in self._chunks]

        if not any(self._tokenized_corpus):
            raise BM25IndexError("Cannot build BM25 index: all chunks tokenize to empty")

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=config.k1,
            b=config.b,
            epsilon=config.epsilon,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        config: BM25Config = DEFAULT_BM25_CONFIG,
    ) -> BM25Retriever:
        return cls(load_bm25_chunks(path), config=config)

    @classmethod
    def from_child_chunks(
        cls,
        chunks: Sequence[ChildChunk],
        *,
        config: BM25Config = DEFAULT_BM25_CONFIG,
    ) -> BM25Retriever:
        return cls(
            [BM25Chunk.from_child_chunk(chunk) for chunk in chunks],
            config=config,
        )

    @property
    def chunks(self) -> tuple[BM25Chunk, ...]:
        return self._chunks

    @property
    def stats(self) -> BM25IndexStats:
        token_counts = [len(tokens) for tokens in self._tokenized_corpus]
        return BM25IndexStats(
            chunk_count=len(self._chunks),
            average_token_count=sum(token_counts) / len(token_counts),
            unique_source_count=len({chunk.source_id for chunk in self._chunks}),
            unique_parent_count=len({chunk.parent_id for chunk in self._chunks}),
        )

    def get_chunk(self, child_id: str) -> BM25Chunk:
        try:
            return self._chunk_by_id[child_id]
        except KeyError as exc:
            raise BM25IndexError(f"Unknown child chunk id: {child_id}") from exc

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[BM25SearchResult]:
        """Return ranked child chunks for a keyword query."""

        if top_k < 1:
            raise BM25QueryError("top_k must be at least 1")
        if not query.strip():
            raise BM25QueryError("query must not be empty")

        query_tokens = tokenize(query, self.config)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )

        results: list[BM25SearchResult] = []
        query_terms = set(query_tokens)

        for index in ranked_indexes:
            score = float(scores[index])
            if minimum_score is not None and score <= minimum_score:
                continue

            chunk_tokens = set(self._tokenized_corpus[index])
            matched_terms = sorted(query_terms & chunk_tokens)
            if not matched_terms:
                continue

            results.append(
                BM25SearchResult(
                    rank=len(results) + 1,
                    score=score,
                    chunk=self._chunks[index],
                    matched_terms=matched_terms,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def parent_ids_for_results(
        self,
        results: Sequence[BM25SearchResult],
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


def load_bm25_chunks(path: Path) -> list[BM25Chunk]:
    """Load child chunks from the pipeline's children.jsonl output."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise BM25LoadError(f"BM25 chunks file not found: {path}")
    if not path.is_file():
        raise BM25LoadError(f"BM25 chunks path is not a file: {path}")

    chunks: list[BM25Chunk] = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                chunks.append(_load_bm25_chunk_line(line, path, line_number))
    except OSError as exc:
        raise BM25LoadError(f"Could not read BM25 chunks file: {path}") from exc

    if not chunks:
        raise BM25LoadError(f"BM25 chunks file contains no records: {path}")

    _index_chunks_by_id(chunks)
    return chunks


def tokenize(text: str, config: BM25Config = DEFAULT_BM25_CONFIG) -> list[str]:
    """Tokenize text consistently for indexing and querying."""

    if config.lowercase:
        text = text.lower()

    tokens = re.findall(config.token_pattern, text)
    filtered_tokens: list[str] = []

    for token in tokens:
        token = token.strip("._-")
        if len(token) < config.min_token_length:
            continue
        if config.remove_stopwords and token in config.stopwords:
            continue
        filtered_tokens.append(token)

    return filtered_tokens


def build_bm25_retriever(
    chunks: Sequence[BM25Chunk | ChildChunk],
    *,
    config: BM25Config = DEFAULT_BM25_CONFIG,
) -> BM25Retriever:
    """Build a BM25 retriever from validated BM25 chunks or child chunks."""

    bm25_chunks = [
        chunk if isinstance(chunk, BM25Chunk) else BM25Chunk.from_child_chunk(chunk)
        for chunk in chunks
    ]
    return BM25Retriever(bm25_chunks, config=config)


def _load_bm25_chunk_line(
    line: str,
    path: Path,
    line_number: int,
) -> BM25Chunk:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise BM25LoadError(
            f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
        ) from exc

    try:
        return BM25Chunk.model_validate(data)
    except ValidationError as exc:
        raise BM25LoadError(
            f"Invalid BM25 chunk record in {path} at line {line_number}"
        ) from exc


def _index_chunks_by_id(chunks: Iterable[BM25Chunk]) -> dict[str, BM25Chunk]:
    indexed: dict[str, BM25Chunk] = {}
    duplicates: set[str] = set()

    for chunk in chunks:
        if chunk.child_id in indexed:
            duplicates.add(chunk.child_id)
        indexed[chunk.child_id] = chunk

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise BM25IndexError(f"Duplicate child chunk id(s): {duplicate_list}")

    return indexed


__all__ = [
    "BM25Chunk",
    "BM25Config",
    "BM25ConfigError",
    "BM25Error",
    "BM25IndexError",
    "BM25IndexStats",
    "BM25LoadError",
    "BM25QueryError",
    "BM25Retriever",
    "BM25SearchResult",
    "DEFAULT_BM25_CONFIG",
    "build_bm25_retriever",
    "load_bm25_chunks",
    "tokenize",
]
