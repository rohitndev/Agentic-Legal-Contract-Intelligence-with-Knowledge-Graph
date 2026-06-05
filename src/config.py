"""Central configuration.

Every external/cloud service is *optional*. The platform runs fully offline with
pure-Python fallbacks; setting the relevant environment variables transparently
upgrades a component to its production backend (Neo4j, Groq, ChromaDB, AWS S3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
STORAGE_DIR.mkdir(exist_ok=True)


def _flag(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    """Resolved runtime settings derived from environment variables."""

    # --- Neo4j knowledge graph (optional) ---
    neo4j_uri: str = field(default_factory=lambda: _flag("NEO4J_URI"))
    neo4j_user: str = field(default_factory=lambda: _flag("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _flag("NEO4J_PASSWORD"))

    # --- Groq LLM (optional) ---
    groq_api_key: str = field(default_factory=lambda: _flag("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _flag("GROQ_MODEL", "mixtral-8x7b-32768"))

    # --- HuggingFace Legal-BERT (optional) ---
    legal_bert_model: str = field(
        default_factory=lambda: _flag("LEGAL_BERT_MODEL", "nlpaueb/legal-bert-base-uncased")
    )
    enable_transformers: bool = field(
        default_factory=lambda: _flag("ENABLE_TRANSFORMERS", "0") == "1"
    )

    # --- ChromaDB vector store (optional) ---
    chroma_path: str = field(default_factory=lambda: _flag("CHROMA_PATH", str(STORAGE_DIR / "chroma")))
    enable_chroma: bool = field(default_factory=lambda: _flag("ENABLE_CHROMA", "0") == "1")

    # --- AWS S3 storage (optional) ---
    aws_bucket: str = field(default_factory=lambda: _flag("AWS_BUCKET"))
    aws_region: str = field(default_factory=lambda: _flag("AWS_REGION", "us-east-1"))

    # --- Celery async broker (optional) ---
    celery_broker_url: str = field(default_factory=lambda: _flag("CELERY_BROKER_URL"))
    celery_result_backend: str = field(default_factory=lambda: _flag("CELERY_RESULT_BACKEND"))

    # --- API ---
    api_host: str = field(default_factory=lambda: _flag("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_flag("API_PORT", "8000")))

    # ---- Capability flags (computed) ----
    @property
    def neo4j_enabled(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)

    @property
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def s3_enabled(self) -> bool:
        return bool(self.aws_bucket)

    @property
    def celery_enabled(self) -> bool:
        return bool(self.celery_broker_url)

    def summary(self) -> dict:
        """Human-readable capability map for /health and the CLI banner."""
        return {
            "knowledge_graph": "neo4j" if self.neo4j_enabled else "in-memory (networkx)",
            "llm": "groq" if self.groq_enabled else "template engine (offline)",
            "ner": "legal-bert" if self.enable_transformers else "rule-based (offline)",
            "vector_store": "chromadb" if self.enable_chroma else "tf-idf (offline)",
            "storage": "aws-s3" if self.s3_enabled else "local filesystem",
            "async": "celery" if self.celery_enabled else "synchronous",
        }


settings = Settings()
