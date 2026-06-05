from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf as fitz
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class LoaderError(Exception):
    """Base class for document loading failures."""


class SourceRegistryError(LoaderError):
    """Raised when a source registry cannot be read or validated."""


class SourceFileError(LoaderError):
    """Raised when source files are missing or unsafe."""


class PDFExtractionError(LoaderError):
    """Raised when PDF text extraction fails."""


class SourceDefaults(BaseModel):
    """Metadata applied to every document in a source registry."""

    model_config = ConfigDict(extra="allow", frozen=True)

    authority: str
    jurisdiction: str
    trust_level: str
    document_type: str

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_string_fields(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be empty")
        return value


class SourceDocument(BaseModel):
    """A single document entry from the source registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    title: str = Field(min_length=1)
    file: str = Field(min_length=1)
    enabled: bool = True

    @field_validator("title", "file", mode="before")
    @classmethod
    def _strip_required_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return value

    @field_validator("file")
    @classmethod
    def _validate_relative_file(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("file must be relative to raw_dir")
        if ".." in path.parts:
            raise ValueError("file must not contain parent directory traversal")
        return value


class SourceRegistry(BaseModel):
    """Validated registry describing a group of local source documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_group: str = Field(min_length=1)
    raw_dir: str = Field(min_length=1)
    processed_dir: str = Field(min_length=1)
    defaults: SourceDefaults
    documents: list[SourceDocument] = Field(min_length=1)

    @field_validator("source_group", "raw_dir", "processed_dir", mode="before")
    @classmethod
    def _strip_required_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return value

    @field_validator("raw_dir", "processed_dir")
    @classmethod
    def _validate_relative_directory(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("directory must be relative to the project root")
        if ".." in path.parts:
            raise ValueError("directory must not contain parent directory traversal")
        return value

    @property
    def enabled_documents(self) -> list[SourceDocument]:
        return [document for document in self.documents if document.enabled]


class DocumentPage(BaseModel):
    """Text extracted from one human-numbered PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


class LoadedDocument(BaseModel):
    """A fully loaded document ready for chunking."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_group: str
    title: str
    file_path: Path
    pages: list[DocumentPage]
    metadata: dict[str, Any]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_page_count(self) -> int:
        return sum(page.has_text for page in self.pages)


def load_source_registry(registry_path: Path) -> SourceRegistry:
    """Load and validate a YAML source registry."""

    registry_path = registry_path.resolve()
    if not registry_path.exists():
        raise SourceRegistryError(f"Source registry not found: {registry_path}")
    if not registry_path.is_file():
        raise SourceRegistryError(f"Source registry path is not a file: {registry_path}")

    try:
        with registry_path.open("r", encoding="utf-8") as file:
            raw_registry = yaml.safe_load(file) or {}
    except OSError as exc:
        raise SourceRegistryError(f"Could not read source registry: {registry_path}") from exc
    except yaml.YAMLError as exc:
        raise SourceRegistryError(f"Invalid YAML in source registry: {registry_path}") from exc

    try:
        registry = SourceRegistry.model_validate(raw_registry)
    except ValidationError as exc:
        raise SourceRegistryError(f"Invalid source registry schema: {registry_path}") from exc

    _validate_unique_document_ids(registry.documents)
    return registry


def load_documents(
    registry_path: Path,
    *,
    project_root: Path | None = None,
) -> list[LoadedDocument]:
    """Load all enabled documents from a source registry.

    Args:
        registry_path: Path to a YAML source registry.
        project_root: Optional project root. When omitted, it is inferred from
            ``data/sources/<registry>.yaml`` style registry paths.

    Returns:
        Loaded documents with page-level text and metadata.
    """

    registry_path = registry_path.resolve()
    registry = load_source_registry(registry_path)
    root = _resolve_project_root(registry_path, project_root)
    raw_dir = _resolve_under_root(root, registry.raw_dir)

    source_paths = resolve_source_paths(registry, root)
    loaded_documents: list[LoadedDocument] = []

    for source_document, file_path in source_paths:
        pages = load_pdf_pages(file_path)
        metadata = _build_document_metadata(registry, source_document)

        loaded_documents.append(
            LoadedDocument(
                source_id=source_document.id,
                source_group=registry.source_group,
                title=source_document.title,
                file_path=file_path,
                pages=pages,
                metadata=metadata,
            )
        )

    LOGGER.info(
        "Loaded %s document(s) from %s",
        len(loaded_documents),
        raw_dir,
    )
    return loaded_documents


def resolve_source_paths(
    registry: SourceRegistry,
    project_root: Path,
) -> list[tuple[SourceDocument, Path]]:
    """Resolve enabled registry entries to validated local PDF paths."""

    root = project_root.resolve()
    raw_dir = _resolve_under_root(root, registry.raw_dir)
    resolved: list[tuple[SourceDocument, Path]] = []
    missing_files: list[Path] = []

    for source_document in registry.enabled_documents:
        file_path = _resolve_under_root(raw_dir, source_document.file)
        if not file_path.exists():
            missing_files.append(file_path)
            continue
        if not file_path.is_file():
            raise SourceFileError(f"Source path is not a file: {file_path}")
        resolved.append((source_document, file_path))

    if missing_files:
        formatted_paths = "\n".join(f"- {path}" for path in missing_files)
        raise SourceFileError(f"Missing source file(s):\n{formatted_paths}")

    if not resolved:
        raise SourceFileError(
            f"No enabled source documents found in registry: {registry.source_group}"
        )

    return resolved


def load_pdf_pages(file_path: Path) -> list[DocumentPage]:
    """Extract sorted page text from a PDF using PyMuPDF."""

    file_path = file_path.resolve()
    if not file_path.exists():
        raise SourceFileError(f"PDF not found: {file_path}")
    if not file_path.is_file():
        raise SourceFileError(f"PDF path is not a file: {file_path}")

    try:
        with fitz.open(file_path) as pdf:
            if pdf.needs_pass:
                raise PDFExtractionError(f"PDF is encrypted: {file_path}")
            if pdf.page_count == 0:
                raise PDFExtractionError(f"PDF has no pages: {file_path}")

            pages = [
                DocumentPage(
                    page_number=page_number,
                    text=_normalize_page_text(page.get_text("text", sort=True)),
                )
                for page_number, page in enumerate(pdf, start=1)
            ]
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"Could not extract text from PDF: {file_path}") from exc

    if not any(page.has_text for page in pages):
        raise PDFExtractionError(f"No extractable text found in PDF: {file_path}")

    return pages


def _validate_unique_document_ids(documents: list[SourceDocument]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for document in documents:
        if document.id in seen:
            duplicates.add(document.id)
        seen.add(document.id)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise SourceRegistryError(f"Duplicate source document id(s): {duplicate_list}")


def _resolve_project_root(registry_path: Path, project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()

    # Expected repo layout: <project>/data/sources/<registry>.yaml
    try:
        return registry_path.parents[2].resolve()
    except IndexError as exc:
        raise SourceRegistryError(
            "Could not infer project root from registry path. Pass project_root explicitly."
        ) from exc


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = (resolved_root / path).resolve()

    if not resolved_path.is_relative_to(resolved_root):
        raise SourceFileError(f"Resolved path escapes root: {resolved_path}")

    return resolved_path


def _build_document_metadata(
    registry: SourceRegistry,
    source_document: SourceDocument,
) -> dict[str, Any]:
    return {
        **registry.defaults.model_dump(),
        "source_group": registry.source_group,
        "source_id": source_document.id,
        "title": source_document.title,
    }


def _normalize_page_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    normalized_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue

        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


__all__ = [
    "DocumentPage",
    "LoadedDocument",
    "LoaderError",
    "PDFExtractionError",
    "SourceDefaults",
    "SourceDocument",
    "SourceFileError",
    "SourceRegistry",
    "SourceRegistryError",
    "load_documents",
    "load_pdf_pages",
    "load_source_registry",
    "resolve_source_paths",
]
