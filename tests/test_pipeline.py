import json
from pathlib import Path

import pymupdf as fitz
import pytest

from trustrag.ingestion.chunking import ChunkingConfig
from trustrag.ingestion.pipeline import (
    CHILDREN_FILENAME,
    MANIFEST_FILENAME,
    PARENTS_FILENAME,
    PipelineConfigError,
    run_ingestion,
)


def _write_registry(project_root: Path, file_name: str = "sample.pdf") -> Path:
    registry_path = project_root / "data" / "sources" / "sebi_sources.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        f"""
source_group: sebi_investor_education
raw_dir: data/raw/sebi
processed_dir: data/processed/sebi

defaults:
  authority: SEBI
  jurisdiction: IN
  trust_level: official
  document_type: investor_education

documents:
  - id: sample_document
    title: Sample Document
    file: {file_name}
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    return registry_path


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_ingestion_writes_parent_child_and_manifest_outputs(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    _write_pdf(
        tmp_path / "data" / "raw" / "sebi" / "sample.pdf",
        " ".join(f"token{number}" for number in range(1, 13)),
    )
    chunking_config = ChunkingConfig(
        parent_chunk_tokens=8,
        parent_chunk_overlap=2,
        child_chunk_tokens=4,
        child_chunk_overlap=1,
        min_parent_tokens=2,
        min_child_tokens=2,
    )

    summary = run_ingestion(
        registry_path,
        project_root=tmp_path,
        chunking_config=chunking_config,
    )

    parent_path = tmp_path / "data" / "processed" / "sebi" / PARENTS_FILENAME
    child_path = tmp_path / "data" / "processed" / "sebi" / CHILDREN_FILENAME
    manifest_path = tmp_path / "data" / "processed" / "sebi" / MANIFEST_FILENAME
    parents = _read_jsonl(parent_path)
    children = _read_jsonl(child_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary.documents_loaded == 1
    assert summary.parent_chunks == len(parents) == 2
    assert summary.child_chunks == len(children) == 5
    assert summary.parent_path == parent_path
    assert summary.child_path == child_path
    assert summary.manifest_path == manifest_path
    assert parents[0]["parent_id"] == "sample_document_p000001"
    assert children[0]["child_id"] == "sample_document_c000001"
    assert children[0]["parent_id"] == "sample_document_p000001"
    assert manifest["documents_loaded"] == 1
    assert manifest["parent_chunks"] == 2
    assert manifest["child_chunks"] == 5


def test_run_ingestion_supports_explicit_output_dir(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    _write_pdf(
        tmp_path / "data" / "raw" / "sebi" / "sample.pdf",
        "short test document",
    )
    output_dir = tmp_path / "custom-output"

    summary = run_ingestion(
        registry_path,
        project_root=tmp_path,
        output_dir=output_dir,
        chunking_config=ChunkingConfig(
            parent_chunk_tokens=10,
            parent_chunk_overlap=0,
            child_chunk_tokens=10,
            child_chunk_overlap=0,
            min_parent_tokens=1,
            min_child_tokens=1,
        ),
    )

    assert summary.output_dir == output_dir
    assert (output_dir / PARENTS_FILENAME).exists()
    assert (output_dir / CHILDREN_FILENAME).exists()
    assert (output_dir / MANIFEST_FILENAME).exists()


def test_run_ingestion_rejects_output_dir_outside_project_root(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    _write_pdf(
        tmp_path / "data" / "raw" / "sebi" / "sample.pdf",
        "short test document",
    )

    with pytest.raises(PipelineConfigError, match="Output directory escapes"):
        run_ingestion(
            registry_path,
            project_root=tmp_path,
            output_dir=tmp_path.parent / "outside-output",
        )
