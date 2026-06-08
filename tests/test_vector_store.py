import json
from pathlib import Path
from collections.abc import Sequence

import pytest

from trustrag.retrieval.vector_store import (
    VectorChunk,
    VectorIndexError,
    VectorLoadError,
    VectorQueryError,
    VectorRetriever,
    VectorStoreConfig,
    build_vector_retriever,
    load_vector_chunks,
)


class _FakeEmbeddingClient:
    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_vector_for_text(text) for text in texts]


def _vector_for_text(text: str) -> list[float]:
    normalized = text.lower()
    if "kyc" in normalized or "identity" in normalized:
        return [1.0, 0.0, 0.0]
    if "commodity" in normalized or "derivative" in normalized:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _chunk(
    child_id: str,
    text: str,
    *,
    parent_id: str | None = None,
    source_id: str = "source_a",
) -> VectorChunk:
    return VectorChunk(
        child_id=child_id,
        parent_id=parent_id or f"{source_id}_p000001",
        source_id=source_id,
        source_group="sebi_investor_education",
        title="Test Document",
        text=text,
        page_start=1,
        page_end=1,
        token_start=0,
        token_end=max(1, len(text.split())),
        token_count=max(1, len(text.split())),
        metadata={
            "authority": "SEBI",
            "jurisdiction": "IN",
            "trust_level": "official",
            "document_type": "investor_education",
        },
    )


def _config(name: str) -> VectorStoreConfig:
    return VectorStoreConfig(collection_name=f"test_{name}")


def test_build_vector_retriever_indexes_and_searches_child_chunks() -> None:
    retriever = build_vector_retriever(
        [
            _chunk("c1", "KYC procedure requires identity verification."),
            _chunk("c2", "Commodity derivatives use futures contracts."),
            _chunk("c3", "A demat account can hold securities."),
        ],
        embedding_client=_FakeEmbeddingClient(),
        config=_config("search"),
    )

    results = retriever.search("how to complete KYC identity checks", top_k=2)

    assert [result.child_id for result in results][:1] == ["c1"]
    assert results[0].rank == 1
    assert results[0].parent_id == "source_a_p000001"
    assert results[0].score == pytest.approx(1.0)
    assert retriever.stats.chunk_count == 3


def test_vector_retriever_can_get_indexed_chunk() -> None:
    retriever = build_vector_retriever(
        [_chunk("c1", "KYC procedure requires identity verification.")],
        embedding_client=_FakeEmbeddingClient(),
        config=_config("get"),
    )

    chunk = retriever.get_chunk("c1")

    assert chunk.child_id == "c1"
    assert chunk.metadata["authority"] == "SEBI"


def test_parent_ids_for_results_deduplicates_in_rank_order() -> None:
    retriever = build_vector_retriever(
        [
            _chunk("c1", "KYC identity", parent_id="p1"),
            _chunk("c2", "KYC verification", parent_id="p1"),
            _chunk("c3", "identity document", parent_id="p2"),
        ],
        embedding_client=_FakeEmbeddingClient(),
        config=_config("parents"),
    )

    results = retriever.search("KYC identity", top_k=3)

    assert retriever.parent_ids_for_results(results) == ["p1", "p2"]


def test_vector_search_rejects_empty_query_and_invalid_top_k() -> None:
    retriever = build_vector_retriever(
        [_chunk("c1", "KYC procedure")],
        embedding_client=_FakeEmbeddingClient(),
        config=_config("query_validation"),
    )

    with pytest.raises(VectorQueryError, match="query must not be empty"):
        retriever.search(" ")

    with pytest.raises(VectorQueryError, match="top_k"):
        retriever.search("kyc", top_k=0)


def test_vector_search_rejects_empty_index() -> None:
    retriever = VectorRetriever(
        embedding_client=_FakeEmbeddingClient(),
        config=_config("empty"),
    )

    with pytest.raises(VectorQueryError, match="empty vector index"):
        retriever.search("kyc")


def test_duplicate_child_ids_are_rejected() -> None:
    retriever = VectorRetriever(
        embedding_client=_FakeEmbeddingClient(),
        config=_config("duplicates"),
    )

    with pytest.raises(VectorIndexError, match="Duplicate child chunk id"):
        retriever.index_chunks(
            [
                _chunk("c1", "KYC identity"),
                _chunk("c1", "demat account"),
            ]
        )


def test_load_vector_chunks_reads_children_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "children.jsonl"
    records = [
        _chunk("c1", "KYC procedure").model_dump(mode="json"),
        _chunk("c2", "demat account").model_dump(mode="json"),
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    chunks = load_vector_chunks(path)

    assert [chunk.child_id for chunk in chunks] == ["c1", "c2"]


def test_load_vector_chunks_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "children.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(VectorLoadError, match="Invalid JSON"):
        load_vector_chunks(path)
