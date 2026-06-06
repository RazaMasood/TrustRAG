import json
from pathlib import Path

import pytest

from trustrag.retrieval.bm25 import (
    BM25Chunk,
    BM25Config,
    BM25IndexError,
    BM25LoadError,
    BM25QueryError,
    BM25Retriever,
    build_bm25_retriever,
    load_bm25_chunks,
    tokenize,
)


def _chunk(
    child_id: str,
    text: str,
    *,
    parent_id: str | None = None,
    source_id: str = "source_a",
) -> BM25Chunk:
    return BM25Chunk(
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


def test_tokenize_normalizes_text_and_removes_stopwords() -> None:
    tokens = tokenize("What is KYC procedure for a demat-account?")

    assert tokens == ["kyc", "procedure", "demat-account"]


def test_bm25_search_ranks_matching_child_chunks() -> None:
    retriever = BM25Retriever(
        [
            _chunk("c1", "KYC procedure requires identity verification and PAN."),
            _chunk("c2", "Commodity derivatives use futures contracts."),
            _chunk("c3", "A demat account can hold securities electronically."),
        ]
    )

    results = retriever.search("KYC identity procedure", top_k=2)

    assert [result.child_id for result in results] == ["c1"]
    assert results[0].rank == 1
    assert results[0].parent_id == "source_a_p000001"
    assert results[0].matched_terms == ["identity", "kyc", "procedure"]


def test_bm25_search_returns_empty_for_stopword_only_query() -> None:
    retriever = BM25Retriever([_chunk("c1", "KYC procedure")])

    assert retriever.search("what is the", top_k=5) == []


def test_bm25_search_rejects_empty_query_and_invalid_top_k() -> None:
    retriever = BM25Retriever([_chunk("c1", "KYC procedure")])

    with pytest.raises(BM25QueryError, match="query must not be empty"):
        retriever.search(" ")

    with pytest.raises(BM25QueryError, match="top_k"):
        retriever.search("kyc", top_k=0)


def test_parent_ids_for_results_deduplicates_in_rank_order() -> None:
    retriever = BM25Retriever(
        [
            _chunk("c1", "KYC identity", parent_id="p1"),
            _chunk("c2", "KYC verification", parent_id="p1"),
            _chunk("c3", "KYC demat", parent_id="p2"),
        ]
    )

    results = retriever.search("KYC", top_k=3)

    assert retriever.parent_ids_for_results(results) == ["p1", "p2"]


def test_retriever_reports_stats() -> None:
    retriever = BM25Retriever(
        [
            _chunk("c1", "KYC identity", parent_id="p1", source_id="source_a"),
            _chunk("c2", "demat account", parent_id="p2", source_id="source_b"),
        ]
    )

    assert retriever.stats.chunk_count == 2
    assert retriever.stats.unique_source_count == 2
    assert retriever.stats.unique_parent_count == 2
    assert retriever.stats.average_token_count == 2


def test_duplicate_child_ids_are_rejected() -> None:
    with pytest.raises(BM25IndexError, match="Duplicate child chunk id"):
        BM25Retriever(
            [
                _chunk("c1", "KYC identity"),
                _chunk("c1", "demat account"),
            ]
        )


def test_load_bm25_chunks_reads_children_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "children.jsonl"
    records = [
        _chunk("c1", "KYC procedure").model_dump(mode="json"),
        _chunk("c2", "demat account").model_dump(mode="json"),
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    chunks = load_bm25_chunks(path)

    assert [chunk.child_id for chunk in chunks] == ["c1", "c2"]


def test_load_bm25_chunks_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "children.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(BM25LoadError, match="Invalid JSON"):
        load_bm25_chunks(path)


def test_build_bm25_retriever_accepts_bm25_chunks() -> None:
    retriever = build_bm25_retriever([_chunk("c1", "KYC procedure")])

    assert retriever.search("KYC")[0].child_id == "c1"


def test_custom_config_can_keep_stopwords() -> None:
    config = BM25Config(remove_stopwords=False, min_token_length=1)

    assert tokenize("what is kyc", config) == ["what", "is", "kyc"]
