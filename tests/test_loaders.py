from pathlib import Path

import fitz
import pytest

from trustrag.ingestion.loaders import (
    PDFExtractionError,
    SourceFileError,
    SourceRegistryError,
    load_documents,
    load_pdf_pages,
    load_source_registry,
    resolve_source_paths,
)


def _write_registry(
    project_root: Path,
    *,
    documents: str,
    raw_dir: str = "data/raw/sebi",
) -> Path:
    registry_path = project_root / "data" / "sources" / "sebi_sources.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        f"""
source_group: sebi_investor_education
raw_dir: {raw_dir}
processed_dir: data/processed/sebi

defaults:
  authority: SEBI
  jurisdiction: IN
  trust_level: official
  document_type: investor_education

documents:
{documents}
""".lstrip(),
        encoding="utf-8",
    )
    return registry_path


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)

    doc.save(path)
    doc.close()


def test_load_source_registry_validates_yaml(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        documents="""
  - id: financial_education_part_a
    title: Financial Education Part A
    file: Financial Education Part A.pdf
    enabled: true
""",
    )

    registry = load_source_registry(registry_path)

    assert registry.source_group == "sebi_investor_education"
    assert registry.defaults.authority == "SEBI"
    assert registry.enabled_documents[0].id == "financial_education_part_a"


def test_load_source_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        documents="""
  - id: duplicate
    title: First
    file: first.pdf
    enabled: true
  - id: duplicate
    title: Second
    file: second.pdf
    enabled: true
""",
    )

    with pytest.raises(SourceRegistryError, match="Duplicate source document id"):
        load_source_registry(registry_path)


def test_resolve_source_paths_reports_missing_files(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        documents="""
  - id: missing_document
    title: Missing Document
    file: missing.pdf
    enabled: true
""",
    )
    registry = load_source_registry(registry_path)

    with pytest.raises(SourceFileError, match="Missing source file"):
        resolve_source_paths(registry, tmp_path)


def test_load_pdf_pages_extracts_human_numbered_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(pdf_path, ["First page text", "Second page text"])

    pages = load_pdf_pages(pdf_path)

    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].text == "First page text"
    assert pages[1].text == "Second page text"


def test_load_pdf_pages_rejects_empty_text_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    _write_pdf(pdf_path, [""])

    with pytest.raises(PDFExtractionError, match="No extractable text"):
        load_pdf_pages(pdf_path)


def test_load_documents_returns_structured_documents(tmp_path: Path) -> None:
    pdf_path = tmp_path / "data" / "raw" / "sebi" / "sample.pdf"
    _write_pdf(pdf_path, ["Investor education text"])
    registry_path = _write_registry(
        tmp_path,
        documents="""
  - id: sample_document
    title: Sample Document
    file: sample.pdf
    enabled: true
""",
    )

    documents = load_documents(registry_path, project_root=tmp_path)

    assert len(documents) == 1
    document = documents[0]
    assert document.source_id == "sample_document"
    assert document.title == "Sample Document"
    assert document.page_count == 1
    assert document.text_page_count == 1
    assert document.pages[0].page_number == 1
    assert document.pages[0].text == "Investor education text"
    assert document.metadata["authority"] == "SEBI"
    assert document.metadata["source_id"] == "sample_document"


def test_registry_rejects_path_traversal(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        documents="""
  - id: unsafe_document
    title: Unsafe Document
    file: ../unsafe.pdf
    enabled: true
""",
    )

    with pytest.raises(SourceRegistryError, match="Invalid source registry schema"):
        load_source_registry(registry_path)
