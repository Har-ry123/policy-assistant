import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.rag import build_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def pipeline():
    # Force the offline configuration so tests never need keys or downloads.
    return build_pipeline(Settings(groq_api_key=None, embedder="hash", reranker="lexical"))
