from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import chromadb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from trustrag.ingestion.chunking import ChildChunk
from trustrag.retrieval.bm25 import BM25Chunk
from trustrag.retrieval.embeddings import EmbeddingClient

LOGGER = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "trustrag_child_chunks"
DEFAULT_DISTANCE_METRIC = "cosine"
COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")
TOP_LEVEL_METADATA_KEYS = {
    "child_id",
    "parent_id",
    "source_id",
    "source_group",
    "title",
    "page_start",
    "page_end",
    "token_start",
    "token_end",
    "token_count",
    "metadata_json",
}


class VectorStoreError(Exception):
    """Base class for vector retrieval failures."""


class VectorStoreConfigError(VectorStoreError):
    """Raised when vector store configuration is invalid."""


class VectorIndexError(VectorStoreError):
    """Raised when vector indexing fails."""


class VectorLoadError(VectorStoreError):
    """Raised when vector chunks cannot be loaded from disk."""


class VectorQueryError(VectorStoreError):
    """Raised when vector search input is invalid."""


class VectorStoreConfig(BaseModel):
    """Configuration for Chroma-backed child chunk retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    collection_name: str = DEFAULT_COLLECTION_NAME
    persist_path: Path | None = None
    distance_metric: Literal["cosine", "l2", "ip"] = DEFAULT_DISTANCE_METRIC
    embedding_batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def _validate_collection_name(self) -> Self:
        if not COLLECTION_NAME_PATTERN.match(self.collection_name):
            raise VectorStoreConfigError(
                "Chroma collection_name must be 3-63 characters and contain only "
                "letters, numbers, dots, underscores, or hyphens"
            )
        return self


class VectorChunk(BaseModel):
    """Searchable child chunk stored in a vector index."""

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
    def from_child_chunk(cls, chunk: ChildChunk) -> VectorChunk:
        return cls.model_validate(chunk.model_dump(mode="python"))

    @classmethod
    def from_bm25_chunk(cls, chunk: BM25Chunk) -> VectorChunk:
        return cls.model_validate(chunk.model_dump(mode="python"))


class VectorSearchResult(BaseModel):
    """One vector retrieval result."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    score: float
    distance: float
    chunk: VectorChunk

    @property
    def child_id(self) -> str:
        return self.chunk.child_id

    @property
    def parent_id(self) -> str:
        return self.chunk.parent_id


class VectorIndexStats(BaseModel):
    """Small diagnostic summary for a vector index."""

    model_config = ConfigDict(frozen=True)

    collection_name: str
    chunk_count: int


DEFAULT_VECTOR_STORE_CONFIG = VectorStoreConfig()


class VectorRetriever:
    """Chroma-backed vector retriever over child chunks."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        config: VectorStoreConfig = DEFAULT_VECTOR_STORE_CONFIG,
        client: Any | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.config = config
        self._client = client or _create_chroma_client(config)
        self._collection = self._get_or_create_collection()

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        embedding_client: EmbeddingClient,
        config: VectorStoreConfig = DEFAULT_VECTOR_STORE_CONFIG,
        client: Any | None = None,
        rebuild: bool = False,
    ) -> VectorRetriever:
        retriever = cls(
            embedding_client=embedding_client,
            config=config,
            client=client,
        )
        retriever.index_chunks(load_vector_chunks(path), rebuild=rebuild)
        return retriever

    @property
    def stats(self) -> VectorIndexStats:
        return VectorIndexStats(
            collection_name=self.config.collection_name,
            chunk_count=int(self._collection.count()),
        )

    def index_chunks(
        self,
        chunks: Sequence[VectorChunk | ChildChunk | BM25Chunk],
        *,
        rebuild: bool = False,
    ) -> VectorIndexStats:
        """Embed and upsert child chunks into the Chroma collection."""

        vector_chunks = _coerce_vector_chunks(chunks)
        if not vector_chunks:
            raise VectorIndexError("Cannot build vector index without chunks")
        _index_chunks_by_id(vector_chunks)

        if rebuild:
            self._reset_collection()

        for batch in _batched(vector_chunks, self.config.embedding_batch_size):
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedding_client.embed_texts(texts)
            _validate_embedding_count(embeddings, batch)
            self._collection.upsert(
                ids=[chunk.child_id for chunk in batch],
                documents=texts,
                metadatas=[_chunk_to_metadata(chunk) for chunk in batch],
                embeddings=embeddings,
            )

        LOGGER.info("Indexed %s vector chunk(s)", len(vector_chunks))
        return self.stats

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
        minimum_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """Return semantically similar child chunks for a query."""

        if top_k < 1:
            raise VectorQueryError("top_k must be at least 1")
        if not query.strip():
            raise VectorQueryError("query must not be empty")
        if self._collection.count() == 0:
            raise VectorQueryError("Cannot search an empty vector index")

        query_embedding = self.embedding_client.embed_text(query)
        raw_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=dict(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        return _parse_query_results(
            raw_results,
            distance_metric=self.config.distance_metric,
            minimum_score=minimum_score,
        )

    def get_chunk(self, child_id: str) -> VectorChunk:
        if not child_id.strip():
            raise VectorQueryError("child_id must not be empty")

        raw_result = self._collection.get(
            ids=[child_id],
            include=["documents", "metadatas"],
        )
        ids = raw_result.get("ids") or []
        if not ids:
            raise VectorIndexError(f"Unknown child chunk id: {child_id}")

        documents = raw_result.get("documents") or []
        metadatas = raw_result.get("metadatas") or []
        return _metadata_to_chunk(metadatas[0], documents[0])

    def parent_ids_for_results(
        self,
        results: Sequence[VectorSearchResult],
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

    def _get_or_create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

    def _reset_collection(self) -> None:
        try:
            self._client.delete_collection(name=self.config.collection_name)
        except Exception:
            LOGGER.debug(
                "Vector collection did not exist before reset: %s",
                self.config.collection_name,
            )
        self._collection = self._get_or_create_collection()


def build_vector_retriever(
    chunks: Sequence[VectorChunk | ChildChunk | BM25Chunk],
    *,
    embedding_client: EmbeddingClient,
    config: VectorStoreConfig = DEFAULT_VECTOR_STORE_CONFIG,
    client: Any | None = None,
    rebuild: bool = False,
) -> VectorRetriever:
    """Build and populate a vector retriever from child chunks."""

    retriever = VectorRetriever(
        embedding_client=embedding_client,
        config=config,
        client=client,
    )
    retriever.index_chunks(chunks, rebuild=rebuild)
    return retriever


def load_vector_chunks(path: Path) -> list[VectorChunk]:
    """Load child chunks from the pipeline's children.jsonl output."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise VectorLoadError(f"Vector chunks file not found: {path}")
    if not path.is_file():
        raise VectorLoadError(f"Vector chunks path is not a file: {path}")

    chunks: list[VectorChunk] = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                chunks.append(_load_vector_chunk_line(line, path, line_number))
    except OSError as exc:
        raise VectorLoadError(f"Could not read vector chunks file: {path}") from exc

    if not chunks:
        raise VectorLoadError(f"Vector chunks file contains no records: {path}")

    _index_chunks_by_id(chunks)
    return chunks


def _create_chroma_client(config: VectorStoreConfig) -> Any:
    if config.persist_path is None:
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=str(config.persist_path.expanduser()))


def _coerce_vector_chunks(
    chunks: Sequence[VectorChunk | ChildChunk | BM25Chunk],
) -> list[VectorChunk]:
    coerced: list[VectorChunk] = []
    for chunk in chunks:
        if isinstance(chunk, VectorChunk):
            coerced.append(chunk)
        elif isinstance(chunk, ChildChunk):
            coerced.append(VectorChunk.from_child_chunk(chunk))
        else:
            coerced.append(VectorChunk.from_bm25_chunk(chunk))
    return coerced


def _load_vector_chunk_line(
    line: str,
    path: Path,
    line_number: int,
) -> VectorChunk:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise VectorLoadError(
            f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
        ) from exc

    try:
        return VectorChunk.model_validate(data)
    except ValidationError as exc:
        raise VectorLoadError(
            f"Invalid vector chunk record in {path} at line {line_number}"
        ) from exc


def _index_chunks_by_id(chunks: Iterable[VectorChunk]) -> dict[str, VectorChunk]:
    indexed: dict[str, VectorChunk] = {}
    duplicates: set[str] = set()

    for chunk in chunks:
        if chunk.child_id in indexed:
            duplicates.add(chunk.child_id)
        indexed[chunk.child_id] = chunk

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise VectorIndexError(f"Duplicate child chunk id(s): {duplicate_list}")

    return indexed


def _chunk_to_metadata(chunk: VectorChunk) -> dict[str, Any]:
    return {
        "child_id": chunk.child_id,
        "parent_id": chunk.parent_id,
        "source_id": chunk.source_id,
        "source_group": chunk.source_group,
        "title": chunk.title,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "token_start": chunk.token_start,
        "token_end": chunk.token_end,
        "token_count": chunk.token_count,
        "metadata_json": json.dumps(
            chunk.metadata,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _metadata_to_chunk(metadata: Mapping[str, Any], document: str) -> VectorChunk:
    try:
        raw_metadata = json.loads(str(metadata.get("metadata_json", "{}")))
    except json.JSONDecodeError as exc:
        raise VectorIndexError("Vector chunk metadata_json is invalid") from exc

    payload = {
        "child_id": metadata["child_id"],
        "parent_id": metadata["parent_id"],
        "source_id": metadata["source_id"],
        "source_group": metadata["source_group"],
        "title": metadata["title"],
        "text": document,
        "page_start": metadata["page_start"],
        "page_end": metadata["page_end"],
        "token_start": metadata["token_start"],
        "token_end": metadata["token_end"],
        "token_count": metadata["token_count"],
        "metadata": raw_metadata,
    }
    return VectorChunk.model_validate(payload)


def _validate_embedding_count(
    embeddings: Sequence[Sequence[float]],
    chunks: Sequence[VectorChunk],
) -> None:
    if len(embeddings) != len(chunks):
        raise VectorIndexError(
            f"Embedding client returned {len(embeddings)} vector(s) for "
            f"{len(chunks)} chunk(s)"
        )


def _parse_query_results(
    raw_results: Mapping[str, Any],
    *,
    distance_metric: str,
    minimum_score: float | None,
) -> list[VectorSearchResult]:
    ids = _first_result_list(raw_results, "ids")
    documents = _first_result_list(raw_results, "documents")
    metadatas = _first_result_list(raw_results, "metadatas")
    distances = _first_result_list(raw_results, "distances")

    results: list[VectorSearchResult] = []
    for index, child_id in enumerate(ids):
        document = documents[index]
        metadata = metadatas[index]
        distance = float(distances[index])
        score = _distance_to_score(distance, distance_metric)
        if minimum_score is not None and score < minimum_score:
            continue

        chunk = _metadata_to_chunk(metadata, document)
        if chunk.child_id != child_id:
            raise VectorIndexError(
                f"Vector result id mismatch: {child_id} != {chunk.child_id}"
            )

        results.append(
            VectorSearchResult(
                rank=len(results) + 1,
                score=score,
                distance=distance,
                chunk=chunk,
            )
        )

    return results


def _first_result_list(raw_results: Mapping[str, Any], key: str) -> list[Any]:
    values = raw_results.get(key) or []
    if not values:
        return []
    return list(values[0])


def _distance_to_score(distance: float, distance_metric: str) -> float:
    if distance_metric == "cosine":
        return 1.0 - distance
    if distance_metric == "l2":
        return 1.0 / (1.0 + distance)
    return distance


def _batched(items: Sequence[VectorChunk], batch_size: int) -> list[list[VectorChunk]]:
    return [
        list(items[start : start + batch_size])
        for start in range(0, len(items), batch_size)
    ]


__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_DISTANCE_METRIC",
    "DEFAULT_VECTOR_STORE_CONFIG",
    "VectorChunk",
    "VectorIndexError",
    "VectorIndexStats",
    "VectorLoadError",
    "VectorQueryError",
    "VectorRetriever",
    "VectorSearchResult",
    "VectorStoreConfig",
    "VectorStoreConfigError",
    "VectorStoreError",
    "build_vector_retriever",
    "load_vector_chunks",
]
