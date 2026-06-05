from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trustrag.ingestion.loaders import DocumentPage, LoadedDocument

LOGGER = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\S+")


class ChunkingError(Exception):
    """Base class for chunking failures."""


class ChunkingConfigError(ChunkingError):
    """Raised when chunking configuration is invalid."""


class EmptyDocumentError(ChunkingError):
    """Raised when a document has no chunkable text."""


class ChunkingConfig(BaseModel):
    """Controls parent-child chunk sizes using whitespace token counts."""

    model_config = ConfigDict(frozen=True)

    parent_chunk_tokens: int = Field(default=1500, ge=1)
    parent_chunk_overlap: int = Field(default=200, ge=0)
    child_chunk_tokens: int = Field(default=400, ge=1)
    child_chunk_overlap: int = Field(default=75, ge=0)
    min_parent_tokens: int = Field(default=120, ge=1)
    min_child_tokens: int = Field(default=50, ge=1)

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        if self.parent_chunk_overlap >= self.parent_chunk_tokens:
            raise ChunkingConfigError(
                "parent_chunk_overlap must be smaller than parent_chunk_tokens"
            )
        if self.child_chunk_overlap >= self.child_chunk_tokens:
            raise ChunkingConfigError(
                "child_chunk_overlap must be smaller than child_chunk_tokens"
            )
        if self.child_chunk_tokens > self.parent_chunk_tokens:
            raise ChunkingConfigError(
                "child_chunk_tokens must be less than or equal to parent_chunk_tokens"
            )
        if self.min_parent_tokens > self.parent_chunk_tokens:
            raise ChunkingConfigError(
                "min_parent_tokens must be less than or equal to parent_chunk_tokens"
            )
        if self.min_child_tokens > self.child_chunk_tokens:
            raise ChunkingConfigError(
                "min_child_tokens must be less than or equal to child_chunk_tokens"
            )
        return self


class ParentChunk(BaseModel):
    """A larger context chunk used for answer generation."""

    model_config = ConfigDict(frozen=True)

    parent_id: str
    source_id: str
    source_group: str
    title: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=1)
    token_count: int = Field(ge=1)
    metadata: dict[str, Any]


class ChildChunk(BaseModel):
    """A smaller retrieval chunk linked back to its parent chunk."""

    model_config = ConfigDict(frozen=True)

    child_id: str
    parent_id: str
    source_id: str
    source_group: str
    title: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=1)
    token_count: int = Field(ge=1)
    metadata: dict[str, Any]


class ChunkedDocument(BaseModel):
    """Parent and child chunks produced from one loaded document."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_group: str
    title: str
    parents: list[ParentChunk]
    children: list[ChildChunk]
    metadata: dict[str, Any]

    @property
    def parent_count(self) -> int:
        return len(self.parents)

    @property
    def child_count(self) -> int:
        return len(self.children)


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    page_number: int
    token_index: int


@dataclass(frozen=True, slots=True)
class _TokenWindow:
    tokens: list[_Token]
    start: int
    end: int


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


def chunk_documents(
    documents: list[LoadedDocument],
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[ChunkedDocument]:
    """Chunk loaded documents into parent-child chunk sets."""

    chunked_documents = [
        chunk_document(document, config=config) for document in documents
    ]
    LOGGER.info("Chunked %s document(s)", len(chunked_documents))
    return chunked_documents


def chunk_document(
    document: LoadedDocument,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> ChunkedDocument:
    """Create parent and child chunks from a loaded document."""

    tokens = _tokens_from_pages(document.pages)
    if not tokens:
        raise EmptyDocumentError(f"Document has no chunkable text: {document.source_id}")

    parent_windows = _build_windows(
        tokens,
        chunk_size=config.parent_chunk_tokens,
        overlap=config.parent_chunk_overlap,
        min_tokens=config.min_parent_tokens,
    )

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    child_number = 1

    for parent_number, parent_window in enumerate(parent_windows, start=1):
        parent_id = _format_chunk_id(document.source_id, "p", parent_number)
        parent = _build_parent_chunk(document, parent_id, parent_window)
        parents.append(parent)

        child_windows = _build_windows(
            parent_window.tokens,
            chunk_size=config.child_chunk_tokens,
            overlap=config.child_chunk_overlap,
            min_tokens=config.min_child_tokens,
        )

        for child_window in child_windows:
            child_id = _format_chunk_id(document.source_id, "c", child_number)
            children.append(
                _build_child_chunk(document, child_id, parent_id, child_window)
            )
            child_number += 1

    return ChunkedDocument(
        source_id=document.source_id,
        source_group=document.source_group,
        title=document.title,
        parents=parents,
        children=children,
        metadata=dict(document.metadata),
    )


def _build_parent_chunk(
    document: LoadedDocument,
    parent_id: str,
    window: _TokenWindow,
) -> ParentChunk:
    page_start, page_end = _page_range(window.tokens)
    return ParentChunk(
        parent_id=parent_id,
        source_id=document.source_id,
        source_group=document.source_group,
        title=document.title,
        text=_render_tokens(window.tokens),
        page_start=page_start,
        page_end=page_end,
        token_start=window.tokens[0].token_index,
        token_end=window.tokens[-1].token_index + 1,
        token_count=len(window.tokens),
        metadata={
            **document.metadata,
            "chunk_type": "parent",
            "parent_id": parent_id,
        },
    )


def _build_child_chunk(
    document: LoadedDocument,
    child_id: str,
    parent_id: str,
    window: _TokenWindow,
) -> ChildChunk:
    page_start, page_end = _page_range(window.tokens)
    return ChildChunk(
        child_id=child_id,
        parent_id=parent_id,
        source_id=document.source_id,
        source_group=document.source_group,
        title=document.title,
        text=_render_tokens(window.tokens),
        page_start=page_start,
        page_end=page_end,
        token_start=window.tokens[0].token_index,
        token_end=window.tokens[-1].token_index + 1,
        token_count=len(window.tokens),
        metadata={
            **document.metadata,
            "chunk_type": "child",
            "child_id": child_id,
            "parent_id": parent_id,
        },
    )


def _tokens_from_pages(pages: list[DocumentPage]) -> list[_Token]:
    tokens: list[_Token] = []
    seen_pages: set[int] = set()

    for page in sorted(pages, key=lambda item: item.page_number):
        if page.page_number in seen_pages:
            raise ChunkingError(f"Duplicate page number: {page.page_number}")
        seen_pages.add(page.page_number)

        for match in TOKEN_PATTERN.finditer(page.text):
            tokens.append(
                _Token(
                    text=match.group(0),
                    page_number=page.page_number,
                    token_index=len(tokens),
                )
            )

    return tokens


def _build_windows(
    tokens: list[_Token],
    *,
    chunk_size: int,
    overlap: int,
    min_tokens: int,
) -> list[_TokenWindow]:
    if not tokens:
        return []

    step = chunk_size - overlap
    bounds: list[tuple[int, int]] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))

        if bounds and end - start < min_tokens:
            previous_start, _ = bounds[-1]
            bounds[-1] = (previous_start, len(tokens))
            break

        bounds.append((start, end))

        if end == len(tokens):
            break
        start += step

    return [
        _TokenWindow(tokens=tokens[start:end], start=start, end=end)
        for start, end in bounds
    ]


def _render_tokens(tokens: list[_Token]) -> str:
    return " ".join(token.text for token in tokens).strip()


def _page_range(tokens: list[_Token]) -> tuple[int, int]:
    page_numbers = [token.page_number for token in tokens]
    return min(page_numbers), max(page_numbers)


def _format_chunk_id(source_id: str, chunk_type: str, number: int) -> str:
    return f"{source_id}_{chunk_type}{number:06d}"


def dump_chunked_document(chunked_document: ChunkedDocument) -> dict[str, Any]:
    """Return a JSON-serializable representation of a chunked document."""

    return chunked_document.model_dump(mode="json")


def load_chunked_document(data: dict[str, Any]) -> ChunkedDocument:
    """Load a chunked document from a JSON-compatible dictionary."""

    return ChunkedDocument.model_validate(data)


__all__ = [
    "ChildChunk",
    "ChunkedDocument",
    "ChunkingConfig",
    "ChunkingConfigError",
    "ChunkingError",
    "DEFAULT_CHUNKING_CONFIG",
    "EmptyDocumentError",
    "ParentChunk",
    "chunk_document",
    "chunk_documents",
    "dump_chunked_document",
    "load_chunked_document",
]
