# Roadmap

## Phase 0: Project Setup

- [x] Create uv project
- [x] Add package structure
- [x] Add docs, prompts, scripts, and tests folders
- [ ] Add first working tests

## Phase 1: Core RAG MVP

- [ ] Implement text chunking
- [ ] Implement BM25 retrieval
- [ ] Implement vector storage with ChromaDB
- [ ] Generate cited answers
- [ ] Add `scripts/ingest.py`
- [ ] Add `scripts/ask.py`

## Phase 2: Trust Layer

- [ ] Add reranker
- [ ] Add document grader
- [ ] Add citation validation
- [ ] Add hallucination checker

## Phase 3: Agentic Workflow

- [ ] Add LangGraph state
- [ ] Add graph nodes
- [ ] Add web fallback
- [ ] Add human-in-the-loop review

## Phase 4: Observability and Evaluation

- [ ] Add Langfuse tracing
- [ ] Add RAGAS evaluation
- [ ] Add benchmark dataset