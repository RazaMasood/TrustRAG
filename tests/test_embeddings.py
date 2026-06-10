import json
import urllib.error
from pathlib import Path

import pytest

import trustrag.retrieval.embeddings as embeddings_module
from trustrag.retrieval.embeddings import (
    EmbeddingConfigError,
    EmbeddingInputError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    OllamaEmbeddingClient,
    OllamaEmbeddingConfig,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _config(
    *,
    base_url: str = "https://ollama.alvision.in/",
    model: str = "embeddinggemma:latest",
    timeout_seconds: float = 30,
    batch_size: int = 32,
    user_agent: str = "TrustRAG/0.1.0",
) -> OllamaEmbeddingConfig:
    return OllamaEmbeddingConfig(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        user_agent=user_agent,
    )


def test_ollama_client_posts_to_embed_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = request.get_header("User-agent")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "model": "embeddinggemma:latest",
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            }
        )

    monkeypatch.setattr(embeddings_module.urllib.request, "urlopen", fake_urlopen)
    client = OllamaEmbeddingClient(
        _config(timeout_seconds=12)
    )

    vectors = client.embed_texts(["KYC process", "demat account"])

    assert client.endpoint == "https://ollama.alvision.in/api/embed"
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["timeout"] == 12
    assert captured["user_agent"] == "TrustRAG/0.1.0"
    assert captured["body"] == {
        "model": "embeddinggemma:latest",
        "input": ["KYC process", "demat account"],
        "truncate": True,
    }


def test_ollama_client_accepts_base_url_that_already_includes_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(
            {"model": "embeddinggemma:latest", "embeddings": [[0.1]]}
        ),
    )
    client = OllamaEmbeddingClient(
        _config(base_url="https://ollama.alvision.in/api")
    )

    client.embed_text("KYC")

    assert client.endpoint == "https://ollama.alvision.in/api/embed"


def test_ollama_client_can_be_configured_from_environment() -> None:
    client = OllamaEmbeddingClient.from_env(
        {
            "TRUSTRAG_OLLAMA_BASE_URL": "https://ollama.alvision.in",
            "TRUSTRAG_EMBEDDING_MODEL": "embeddinggemma:latest",
            "TRUSTRAG_OLLAMA_TIMEOUT_SECONDS": "15",
            "TRUSTRAG_EMBEDDING_BATCH_SIZE": "8",
            "TRUSTRAG_OLLAMA_USER_AGENT": "TrustRAG/0.1.0",
        }
    )

    assert client.config.base_url == "https://ollama.alvision.in"
    assert client.config.model == "embeddinggemma:latest"
    assert client.config.timeout_seconds == 15
    assert client.config.batch_size == 8
    assert client.config.user_agent == "TrustRAG/0.1.0"


def test_ollama_client_rejects_missing_environment_config() -> None:
    with pytest.raises(EmbeddingConfigError, match="TRUSTRAG_OLLAMA_BASE_URL"):
        OllamaEmbeddingClient.from_env({})


def test_ollama_client_loads_dotenv_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "TRUSTRAG_OLLAMA_BASE_URL=https://ollama.alvision.in",
                "TRUSTRAG_EMBEDDING_MODEL=bge-m3:latest",
                "TRUSTRAG_OLLAMA_TIMEOUT_SECONDS=20",
                "TRUSTRAG_EMBEDDING_BATCH_SIZE=4",
                "TRUSTRAG_OLLAMA_USER_AGENT=TrustRAG/0.1.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TRUSTRAG_OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("TRUSTRAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("TRUSTRAG_OLLAMA_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("TRUSTRAG_EMBEDDING_BATCH_SIZE", raising=False)
    monkeypatch.delenv("TRUSTRAG_OLLAMA_USER_AGENT", raising=False)

    client = OllamaEmbeddingClient.from_env(dotenv_path=dotenv_path)

    assert client.config.base_url == "https://ollama.alvision.in"
    assert client.config.model == "bge-m3:latest"
    assert client.config.timeout_seconds == 20
    assert client.config.batch_size == 4
    assert client.config.user_agent == "TrustRAG/0.1.0"


def test_ollama_client_rejects_blank_input() -> None:
    client = OllamaEmbeddingClient(_config())

    with pytest.raises(EmbeddingInputError, match="empty"):
        client.embed_text("   ")


def test_ollama_client_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(
            {"model": "embeddinggemma:latest", "embeddings": []}
        ),
    )
    client = OllamaEmbeddingClient(_config())

    with pytest.raises(EmbeddingResponseError, match="did not contain vectors"):
        client.embed_text("KYC")


def test_ollama_client_reports_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(embeddings_module.urllib.request, "urlopen", fake_urlopen)
    client = OllamaEmbeddingClient(_config())

    with pytest.raises(EmbeddingRequestError, match="request failed"):
        client.embed_text("KYC")
