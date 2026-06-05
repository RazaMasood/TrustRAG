from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trustrag.ingestion.chunking import (
    ChunkedDocument,
    ChunkingConfig,
    DEFAULT_CHUNKING_CONFIG,
    chunk_documents,
)
from trustrag.ingestion.loaders import load_documents, load_source_registry

LOGGER = logging.getLogger(__name__)

PARENTS_FILENAME = "parents.jsonl"
CHILDREN_FILENAME = "children.jsonl"
MANIFEST_FILENAME = "manifest.json"


class PipelineError(Exception):
    """Base class for ingestion pipeline failures."""


class PipelineConfigError(PipelineError):
    """Raised when pipeline configuration is invalid."""


class PipelineOutputError(PipelineError):
    """Raised when pipeline outputs cannot be written safely."""


class IngestionPipelineConfig(BaseModel):
    """Configuration for one ingestion pipeline run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    registry_path: Path
    project_root: Path | None = None
    output_dir: Path | None = None
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class IngestionRunSummary(BaseModel):
    """Machine-readable summary of a completed ingestion run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_group: str
    registry_path: Path
    output_dir: Path
    parent_path: Path
    child_path: Path
    manifest_path: Path
    documents_loaded: int
    parent_chunks: int
    child_chunks: int
    chunking: ChunkingConfig


def run_ingestion_pipeline(
    config: IngestionPipelineConfig,
) -> IngestionRunSummary:
    """Run loading, chunking, and JSONL writing for one source registry."""

    config = _resolve_config(config)
    registry = load_source_registry(config.registry_path)
    output_dir = _resolve_output_dir(config, registry.processed_dir)

    documents = load_documents(
        config.registry_path,
        project_root=config.project_root,
    )
    chunked_documents = chunk_documents(documents, config=config.chunking)

    parent_path = output_dir / PARENTS_FILENAME
    child_path = output_dir / CHILDREN_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    parent_count = _write_jsonl_atomic(
        parent_path,
        _iter_parent_records(chunked_documents),
    )
    child_count = _write_jsonl_atomic(
        child_path,
        _iter_child_records(chunked_documents),
    )

    summary = IngestionRunSummary(
        source_group=registry.source_group,
        registry_path=config.registry_path,
        output_dir=output_dir,
        parent_path=parent_path,
        child_path=child_path,
        manifest_path=manifest_path,
        documents_loaded=len(documents),
        parent_chunks=parent_count,
        child_chunks=child_count,
        chunking=config.chunking,
    )
    _write_json_atomic(manifest_path, summary.model_dump(mode="json"))

    LOGGER.info(
        "Ingested %s document(s): %s parent chunk(s), %s child chunk(s)",
        summary.documents_loaded,
        summary.parent_chunks,
        summary.child_chunks,
    )
    return summary


def run_ingestion(
    registry_path: Path,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    chunking_config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> IngestionRunSummary:
    """Convenience wrapper for running the ingestion pipeline."""

    return run_ingestion_pipeline(
        IngestionPipelineConfig(
            registry_path=registry_path,
            project_root=project_root,
            output_dir=output_dir,
            chunking=chunking_config,
        )
    )


def _iter_parent_records(
    chunked_documents: Iterable[ChunkedDocument],
) -> Iterable[dict[str, Any]]:
    for chunked_document in chunked_documents:
        for parent in chunked_document.parents:
            yield parent.model_dump(mode="json")


def _iter_child_records(
    chunked_documents: Iterable[ChunkedDocument],
) -> Iterable[dict[str, Any]]:
    for chunked_document in chunked_documents:
        for child in chunked_document.children:
            yield child.model_dump(mode="json")


def _write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> int:
    _validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_output_path(path)
    count = 0

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                file.write("\n")
                count += 1
        temp_path.replace(path)
    except OSError as exc:
        _cleanup_temp_path(temp_path)
        raise PipelineOutputError(f"Could not write JSONL output: {path}") from exc

    return count


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    _validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_output_path(path)

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        temp_path.replace(path)
    except OSError as exc:
        _cleanup_temp_path(temp_path)
        raise PipelineOutputError(f"Could not write JSON output: {path}") from exc


def _resolve_output_dir(
    config: IngestionPipelineConfig,
    registry_processed_dir: str,
) -> Path:
    project_root = _require_project_root(config.project_root)

    if config.output_dir is not None:
        output_dir = config.output_dir
    else:
        output_dir = (project_root / registry_processed_dir).resolve()

    if not output_dir.is_relative_to(project_root):
        raise PipelineConfigError(f"Output directory escapes project root: {output_dir}")

    return output_dir


def _resolve_config(config: IngestionPipelineConfig) -> IngestionPipelineConfig:
    registry_path = config.registry_path.expanduser().resolve()
    project_root = (
        config.project_root.expanduser().resolve()
        if config.project_root is not None
        else _infer_project_root(registry_path)
    )
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else None
    )

    return config.model_copy(
        update={
            "registry_path": registry_path,
            "project_root": project_root,
            "output_dir": output_dir,
        }
    )


def _infer_project_root(registry_path: Path) -> Path:
    try:
        return registry_path.parents[2].resolve()
    except IndexError as exc:
        raise PipelineConfigError(
            "Could not infer project root from registry path. Pass project_root explicitly."
        ) from exc


def _require_project_root(project_root: Path | None) -> Path:
    if project_root is None:
        raise PipelineConfigError("project_root is required after config validation")
    return project_root


def _validate_output_path(path: Path) -> None:
    if path.exists() and not path.is_file():
        raise PipelineOutputError(f"Output path is not a file: {path}")


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _cleanup_temp_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove temporary output file: %s", path)


__all__ = [
    "CHILDREN_FILENAME",
    "MANIFEST_FILENAME",
    "PARENTS_FILENAME",
    "IngestionPipelineConfig",
    "IngestionRunSummary",
    "PipelineConfigError",
    "PipelineError",
    "PipelineOutputError",
    "run_ingestion",
    "run_ingestion_pipeline",
]
