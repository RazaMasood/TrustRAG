from pathlib import Path
from trustrag.ingestion.pipeline import run_ingestion

summary = run_ingestion(Path("data/sources/sebi_sources.yaml"))
print(summary)