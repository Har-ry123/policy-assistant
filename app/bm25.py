"""BM25 keyword search (Week 4: the keyword half of hybrid search)."""
from __future__ import annotations

import math
from collections import Counter

from .chunking import Chunk
from .text import tokens


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [c.chunk_id for c in chunks]
        self.docs = [Counter(tokens(c.search_text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int, allowed_ids: set[str]) -> list[tuple[str, float]]:
        q = tokens(query)
        scored = []
        for i, cid in enumerate(self.ids):
            if cid not in allowed_ids:
                continue
            score = 0.0
            for t in q:
                tf = self.docs[i].get(t, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
                    score += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((cid, score))
        return sorted(scored, key=lambda x: -x[1])[:k]
