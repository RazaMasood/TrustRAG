from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_OLLAMA_USER_AGENT = "curl/8.5.0"


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingConfigError(EmbeddingError):
    """Raised when embedding client configuration is invalid."""


class EmbeddingInputError(EmbeddingError):
    """Raised when embedding input is invalid."""


class EmbeddingRequestError(EmbeddingError):
    """Raised when an embedding provider request fails."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when an embedding provider returns an invalid response."""


class EmbeddingClient(Protocol):
    """Minimal interface required by retrieval code."""

    def embed_text(self, text: str) -> list[float]:
        """Return one embedding vector for one input string."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""


class OllamaEmbeddingConfig(BaseModel):
    """Configuration for Ollama's embedding API."""

    model_config = ConfigDict(frozen=True)

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    batch_size: int = Field(default=DEFAULT_EMBEDDING_BATCH_SIZE, ge=1)
    user_agent: str = DEFAULT_OLLAMA_USER_AGENT
    truncate: bool = True
    dimensions: int | None = Field(default=None, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_base_url(self) -> Self:
        if not self.base_url.strip():
            raise EmbeddingConfigError("Ollama base_url must not be empty")
        return self


class OllamaEmbeddingResponse(BaseModel):
    """Validated subset of Ollama's /api/embed response."""

    model_config = ConfigDict(frozen=True)

    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None

    @model_validator(mode="after")
    def _validate_embeddings(self) -> Self:
        _validate_vectors(self.embeddings)
        return self


class OllamaEmbeddingClient:
    """HTTP client for Ollama's /api/embed endpoint."""

    def __init__(self, config: OllamaEmbeddingConfig | None = None) -> None:
        self.config = config or OllamaEmbeddingConfig()
        self._endpoint = _build_embed_endpoint(self.config.base_url)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        dotenv_path: Path | str | None = None,
    ) -> OllamaEmbeddingClient:
        """Create a client from .env and TRUSTRAG_* environment variables."""

        if env is None:
            load_dotenv(dotenv_path=dotenv_path, override=False)
            env = os.environ

        options: dict[str, Any] = {
            "base_url": env.get("TRUSTRAG_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            "model": env.get("TRUSTRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        }

        if timeout := env.get("TRUSTRAG_OLLAMA_TIMEOUT_SECONDS"):
            options["timeout_seconds"] = float(timeout)
        if batch_size := env.get("TRUSTRAG_EMBEDDING_BATCH_SIZE"):
            options["batch_size"] = int(batch_size)
        if user_agent := env.get("TRUSTRAG_OLLAMA_USER_AGENT"):
            options["user_agent"] = user_agent

        return cls(OllamaEmbeddingConfig(**options))

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings in provider-sized batches."""

        validated_texts = _validate_texts(texts)
        vectors: list[list[float]] = []

        for batch in _batched(validated_texts, self.config.batch_size):
            vectors.extend(self._embed_batch(batch))

        return vectors

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": list(texts),
            "truncate": self.config.truncate,
        }
        if self.config.dimensions is not None:
            payload["dimensions"] = self.config.dimensions
        if self.config.options:
            payload["options"] = self.config.options

        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                raw_response = response.read()
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            raise EmbeddingRequestError(
                f"Ollama embedding request failed with HTTP {exc.code}: {body}"
            ) from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise EmbeddingRequestError(
                f"Ollama embedding request failed: {exc}"
            ) from exc

        embedding_response = _parse_embedding_response(raw_response)
        if len(embedding_response.embeddings) != len(texts):
            raise EmbeddingResponseError(
                "Ollama returned "
                f"{len(embedding_response.embeddings)} embedding(s) for "
                f"{len(texts)} input(s)"
            )

        return embedding_response.embeddings


def _build_embed_endpoint(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/api"):
        return f"{normalized}/embed"
    return f"{normalized}/api/embed"


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if not texts:
        return []

    validated: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise EmbeddingInputError(f"Embedding input {index} is not a string")
        if not text.strip():
            raise EmbeddingInputError(f"Embedding input {index} is empty")
        validated.append(text)

    return validated


def _validate_vectors(vectors: Sequence[Sequence[float]]) -> None:
    if not vectors:
        raise EmbeddingResponseError("Embedding response did not contain vectors")

    expected_dimensions = len(vectors[0])
    if expected_dimensions == 0:
        raise EmbeddingResponseError("Embedding vectors must not be empty")

    for vector_index, vector in enumerate(vectors):
        if len(vector) != expected_dimensions:
            raise EmbeddingResponseError(
                "Embedding response contained inconsistent vector dimensions"
            )
        for value in vector:
            if not math.isfinite(float(value)):
                raise EmbeddingResponseError(
                    f"Embedding vector {vector_index} contains a non-finite value"
                )


def _parse_embedding_response(raw_response: bytes) -> OllamaEmbeddingResponse:
    try:
        data = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingResponseError("Ollama returned invalid JSON") from exc

    try:
        return OllamaEmbeddingResponse.model_validate(data)
    except ValidationError as exc:
        raise EmbeddingResponseError("Ollama returned an invalid embedding payload") from exc


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace")
    except OSError:
        return "<unreadable response body>"
    return body.strip() or "<empty response body>"


def _batched(items: Sequence[str], batch_size: int) -> list[list[str]]:
    return [
        list(items[start : start + batch_size])
        for start in range(0, len(items), batch_size)
    ]


__all__ = [
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_USER_AGENT",
    "EmbeddingClient",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingRequestError",
    "EmbeddingResponseError",
    "OllamaEmbeddingClient",
    "OllamaEmbeddingConfig",
    "OllamaEmbeddingResponse",
]
