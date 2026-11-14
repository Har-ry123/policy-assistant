"""Settings come from environment variables (Week 1: no keys in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader so the project needs no extra dependency."""
    env = path or ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = None
    model: str = "openai/gpt-oss-20b"
    embedder: str = "hash"          # "hash" (offline) or "minilm" (sentence-transformers)
    reranker: str = "lexical"       # "lexical" (offline) or "cross-encoder"
    top_k: int = 4
    qdrant_url: str | None = None   # None = in-memory Qdrant
    data_dir: Path = ROOT / "data" / "policies"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            model=os.getenv("MODEL", cls.model),
            embedder=os.getenv("EMBEDDER", cls.embedder),
            reranker=os.getenv("RERANKER", cls.reranker),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            data_dir=Path(os.getenv("DATA_DIR", str(cls.data_dir))),
        )
