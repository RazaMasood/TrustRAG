from pathlib import Path

# You are already inside the TrustRAG folder
ROOT = Path.cwd()

dirs = [
    "src/trustrag/graph",
    "src/trustrag/ingestion",
    "src/trustrag/retrieval",
    "src/trustrag/generation",
    "src/trustrag/evaluation",
    "src/trustrag/web",
    "src/trustrag/memory",
    "src/trustrag/observability",
    "tests",
    "data/raw",
    "data/processed",
    "storage/chroma",
    "prompts",
    "scripts",
    "docs",
]

files = [
    "src/trustrag/__init__.py",
    "src/trustrag/config.py",

    "src/trustrag/graph/state.py",
    "src/trustrag/graph/workflow.py",
    "src/trustrag/graph/nodes.py",

    "src/trustrag/ingestion/loaders.py",
    "src/trustrag/ingestion/chunking.py",
    "src/trustrag/ingestion/pipeline.py",

    "src/trustrag/retrieval/embeddings.py",
    "src/trustrag/retrieval/vector_store.py",
    "src/trustrag/retrieval/bm25.py",
    "src/trustrag/retrieval/hybrid.py",
    "src/trustrag/retrieval/reranker.py",

    "src/trustrag/generation/prompts.py",
    "src/trustrag/generation/answer.py",
    "src/trustrag/generation/citations.py",

    "src/trustrag/evaluation/grader.py",
    "src/trustrag/evaluation/hallucination.py",
    "src/trustrag/evaluation/ragas_eval.py",

    "src/trustrag/web/search.py",

    "src/trustrag/memory/conversation.py",

    "src/trustrag/observability/langfuse.py",

    "src/trustrag/cli.py",

    "tests/test_chunking.py",
    "tests/test_retrieval.py",
    "tests/test_citations.py",

    "prompts/query_analyzer.md",
    "prompts/answer_generator.md",
    "prompts/document_grader.md",

    "scripts/ingest.py",
    "scripts/ask.py",
    "scripts/evaluate.py",

    "docs/architecture.md",
    "docs/roadmap.md",

    ".env.example",
    ".gitignore",
    "pyproject.toml",
    "README.md",
    "LICENSE",
]

for directory in dirs:
    path = ROOT / directory
    path.mkdir(parents=True, exist_ok=True)

for file in files:
    path = ROOT / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print(f"Project structure created inside: {ROOT}")