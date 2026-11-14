"""Retrieval modes (Weeks 3 and 4): vector, bm25, hybrid (RRF) and hybrid + rerank.

Access control is enforced HERE, before anything reaches the LLM: a user's roles decide which
chunks may be retrieved at all (Week 7: never rely on the model to hide restricted data).
"""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .chunking import Chunk
from .store import VectorStore

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: merge ranked lists using only ranks."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def allowed_access(roles) -> set[str]:
    return {"all"} | {r.strip().lower() for r in roles if r and r.strip()}


class Retriever:
    def __init__(self, chunks: list[Chunk], store: VectorStore, bm25: BM25, reranker, candidate_k: int = 20):
        self.by_id = {c.chunk_id: c for c in chunks}
        self.chunks, self.store, self.bm25, self.reranker = chunks, store, bm25, reranker
        self.candidate_k = candidate_k

    def search(self, query: str, mode: str = "hybrid_rerank", roles=(), k: int = 4) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        allowed = allowed_access(roles)
        allowed_ids = {c.chunk_id for c in self.chunks if c.access in allowed}
        n = self.candidate_k
        if mode == "vector":
            ranked = self.store.search(query, k, allowed)
        elif mode == "bm25":
            ranked = self.bm25.search(query, k, allowed_ids)
        else:
            vec = self.store.search(query, n, allowed)
            kw = self.bm25.search(query, n, allowed_ids)
            ranked = rrf([[c for c, _ in vec], [c for c, _ in kw]])[:n]
            if mode == "hybrid_rerank":
                scored = [(cid, self.reranker.score(query, self.by_id[cid])) for cid, _ in ranked]
                ranked = sorted(scored, key=lambda x: -x[1])
            ranked = ranked[:k]
        return [Hit(self.by_id[cid], float(score)) for cid, score in ranked if cid in allowed_ids]
