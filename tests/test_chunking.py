from pathlib import Path

import pytest
from pydantic import ValidationError

from trustrag.ingestion.chunking import (
    ChunkingConfig,
    ChunkingConfigError,
    EmptyDocumentError,
    chunk_document,
    chunk_documents,
    dump_chunked_document,
    load_chunked_document,
)
from trustrag.ingestion.loaders import DocumentPage, LoadedDocument


def _loaded_document(
    pages: list[tuple[int, str]],
    *,
    source_id: str = "sebi_test",
) -> LoadedDocument:
    return LoadedDocument(
        source_id=source_id,
        source_group="sebi_investor_education",
        title="SEBI Test Document",
        file_path=Path("test.pdf"),
        pages=[
            DocumentPage(page_number=page_number, text=text)
            for page_number, text in pages
        ],
        metadata={
            "authority": "SEBI",
            "jurisdiction": "IN",
            "trust_level": "official",
            "document_type": "investor_education",
        },
    )


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{number:02d}" for number in range(1, count + 1))


def test_chunk_document_creates_parent_and_child_chunks_with_page_ranges() -> None:
    document = _loaded_document(
        [
            (1, _words("a", 8)),
            (2, _words("b", 8)),
        ]
    )
    config = ChunkingConfig(
        parent_chunk_tokens=10,
        parent_chunk_overlap=2,
        child_chunk_tokens=4,
        child_chunk_overlap=1,
        min_parent_tokens=2,
        min_child_tokens=2,
    )

    chunked = chunk_document(document, config=config)

    assert chunked.parent_count == 2
    assert chunked.child_count == 6
    assert chunked.parents[0].page_start == 1
    assert chunked.parents[0].page_end == 2
    assert chunked.parents[1].page_start == 2
    assert chunked.parents[1].page_end == 2
    assert all(child.parent_id in {parent.parent_id for parent in chunked.parents} for child in chunked.children)


def test_chunk_ids_are_stable_and_metadata_is_preserved() -> None:
    document = _loaded_document([(1, _words("w", 12))], source_id="source_a")
    config = ChunkingConfig(
        parent_chunk_tokens=8,
        parent_chunk_overlap=2,
        child_chunk_tokens=4,
        child_chunk_overlap=1,
        min_parent_tokens=2,
        min_child_tokens=2,
    )

    chunked = chunk_document(document, config=config)

    assert chunked.parents[0].parent_id == "source_a_p000001"
    assert chunked.children[0].child_id == "source_a_c000001"
    assert chunked.children[0].parent_id == "source_a_p000001"
    assert chunked.children[0].metadata["authority"] == "SEBI"
    assert chunked.children[0].metadata["chunk_type"] == "child"


def test_parent_overlap_repeats_boundary_tokens() -> None:
    document = _loaded_document([(1, _words("w", 10))])
    config = ChunkingConfig(
        parent_chunk_tokens=5,
        parent_chunk_overlap=2,
        child_chunk_tokens=5,
        child_chunk_overlap=2,
        min_parent_tokens=1,
        min_child_tokens=1,
    )

    chunked = chunk_document(document, config=config)

    first_parent_tokens = chunked.parents[0].text.split()
    second_parent_tokens = chunked.parents[1].text.split()
    assert first_parent_tokens[-2:] == second_parent_tokens[:2]


def test_short_tail_is_merged_into_previous_window() -> None:
    document = _loaded_document([(1, _words("w", 11))])
    config = ChunkingConfig(
        parent_chunk_tokens=5,
        parent_chunk_overlap=0,
        child_chunk_tokens=5,
        child_chunk_overlap=0,
        min_parent_tokens=3,
        min_child_tokens=1,
    )

    chunked = chunk_document(document, config=config)

    assert [parent.token_count for parent in chunked.parents] == [5, 6]
    assert chunked.parents[-1].text.endswith("w11")


def test_empty_document_raises_clear_error() -> None:
    document = _loaded_document([(1, "   "), (2, "")])

    with pytest.raises(EmptyDocumentError, match="no chunkable text"):
        chunk_document(document)


def test_duplicate_page_number_raises_error() -> None:
    document = _loaded_document([(1, "first page"), (1, "duplicate page")])

    with pytest.raises(Exception, match="Duplicate page number"):
        chunk_document(document)


def test_chunking_config_rejects_invalid_relationships() -> None:
    with pytest.raises((ChunkingConfigError, ValidationError)):
        ChunkingConfig(
            parent_chunk_tokens=10,
            parent_chunk_overlap=10,
            child_chunk_tokens=5,
            child_chunk_overlap=1,
            min_parent_tokens=1,
            min_child_tokens=1,
        )


def test_chunk_documents_chunks_multiple_documents() -> None:
    documents = [
        _loaded_document([(1, _words("a", 6))], source_id="doc_a"),
        _loaded_document([(1, _words("b", 6))], source_id="doc_b"),
    ]
    config = ChunkingConfig(
        parent_chunk_tokens=6,
        parent_chunk_overlap=0,
        child_chunk_tokens=3,
        child_chunk_overlap=0,
        min_parent_tokens=1,
        min_child_tokens=1,
    )

    chunked_documents = chunk_documents(documents, config=config)

    assert [chunked.source_id for chunked in chunked_documents] == ["doc_a", "doc_b"]
    assert [chunked.child_count for chunked in chunked_documents] == [2, 2]


def test_chunked_document_round_trip_dump_and_load() -> None:
    document = _loaded_document([(1, _words("w", 5))])
    config = ChunkingConfig(
        parent_chunk_tokens=5,
        parent_chunk_overlap=0,
        child_chunk_tokens=5,
        child_chunk_overlap=0,
        min_parent_tokens=1,
        min_child_tokens=1,
    )
    chunked = chunk_document(document, config=config)

    restored = load_chunked_document(dump_chunked_document(chunked))

    assert restored == chunked
