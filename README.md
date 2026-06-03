# TrustRAG

TrustRAG is a planned agentic RAG system focused on source-grounded answers, hybrid retrieval, reranking, hallucination checks, and human-in-the-loop review. The target stack uses Ollama for local LLM and embedding models.

![TrustRAG target architecture](docs/assets/trustrag-agentic-rag-architecture.png)

## Status

Current status: project scaffold. Core RAG implementation is in progress.

## Target Features

- Document ingestion and chunking
- Local LLM and embeddings through Ollama
- Hybrid retrieval with ChromaDB and BM25
- Reranking with a cross-encoder model
- Source-cited answer generation
- Hallucination checking
- Web fallback for missing context
- LangGraph-based agent workflow
- Langfuse observability
