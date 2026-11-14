"""Rerankers (Week 4). A reranker re-scores a short candidate list more carefully."""
from __future__ import annotations

from .chunking import Chunk
from .text import tokens


class LexicalReranker:
    """Offline: query-term coverage + bigram matches + title match. Not a neural model."""

    def score(self, query: str, chunk: Chunk) -> float:
        q = tokens(query)
        if not q:
            return 0.0
        body = tokens(chunk.search_text)
        body_set = set(body)
        coverage = sum(1 for t in set(q) if t in body_set) / len(set(q))
        q_bigrams = set(zip(q, q[1:]))
        bigrams = len(q_bigrams & set(zip(body, body[1:]))) / (len(q_bigrams) or 1)
        title_hits = len(set(q) & set(tokens(chunk.title + " " + chunk.section))) / len(set(q))
        return coverage + 0.5 * bigrams + 0.3 * title_hits


class CrossEncoderReranker:
    """Neural reranker (sentence-transformers cross-encoder); downloads a model on first use."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, chunk: Chunk) -> float:
        return float(self.model.predict([(query, chunk.text)])[0])


def get_reranker(kind: str):
    if kind == "cross-encoder":
        return CrossEncoderReranker()
    if kind == "lexical":
        return LexicalReranker()
    raise ValueError(f"Unknown reranker: {kind}")
