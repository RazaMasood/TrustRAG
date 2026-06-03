# Architecture

TrustRAG follows an agentic RAG pipeline. The target model stack is Ollama-first: local LLM calls for reasoning and answer generation, plus local embedding generation for retrieval.

Current status: scaffold. This diagram represents the planned architecture.

![TrustRAG target architecture](assets/trustrag-agentic-rag-architecture.png)

## Target Pipeline

1. Query analyzer with Ollama
2. Multi-query rewriter with Ollama
3. Hybrid retriever with ChromaDB, BM25, and Ollama embeddings
4. Reranker
5. Document grader with Ollama
6. Web fallback
7. Answer generator with Ollama
8. Hallucination checker
9. Memory store
10. Human-in-the-loop review

## Module Boundaries

- `ingestion/`: loading, cleaning, and chunking documents
- `retrieval/`: embeddings, BM25, vector search, hybrid search, reranking
- `generation/`: prompt assembly, answer generation, citations
- `evaluation/`: grading, hallucination checks, RAGAS metrics
- `graph/`: LangGraph orchestration
- `memory/`: conversation memory
- `observability/`: Langfuse tracing

## Design Principle

Business logic should live in focused modules. LangGraph should orchestrate those modules, not hide the core logic inside graph nodes.
