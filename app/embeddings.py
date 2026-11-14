"""Embedding models (Week 3).

``HashEmbedder`` is an OFFLINE, lexical baseline (hashed bag of stemmed words + bigrams). It does not
understand synonyms; it exists so the project runs and tests without downloads. For real semantic
search set EMBEDDER=minilm (sentence-transformers).
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .text import tokens


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokens(text)
        grams = [(t, 1.0) for t in toks] + [(f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:])]
        for gram, weight in grams:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 100) & 1 else -1.0
            vec[h % self.dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def get_embedder(kind: str) -> Embedder:
    if kind == "minilm":
        return SentenceTransformerEmbedder()
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedder: {kind}")
