# Architecture

TrustRAG follows an agentic RAG pipeline.

Current status: scaffold. This diagram represents the planned architecture.

![TrustRAG target architecture](assets/trustrag-agentic-rag-architecture.png)

## Target Pipeline

1. Query analyzer
2. Multi-query rewriter
3. Hybrid retriever
4. Reranker
5. Document grader
6. Web fallback
7. Answer generator
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