"""Vector stores (Week 3).

``QdrantStore``     - Qdrant (in memory by default, or a server via QDRANT_URL). Uses a payload filter
                      on ``access`` so a user only retrieves documents their roles allow.
``BruteForceStore`` - exact search in pure Python. Same interface: shows how a vector DB can be
                      swapped (Chroma, FAISS, pgvector ...) and gives an exact baseline to compare
                      against HNSW.
"""
from __future__ import annotations

from typing import Protocol

from .chunking import Chunk
from .embeddings import Embedder


class VectorStore(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int, allowed: set[str]) -> list[tuple[str, float]]: ...


class QdrantStore:
    def __init__(self, embedder: Embedder, url: str | None = None, collection: str = "policies"):
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.collection = collection
        self.url = url
        self.client =QdrantClient(url=url) if url else QdrantClient(":memory:")

    def index(self, chunks: list[Chunk]) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=i, vector=self.embedder.embed(c.search_text),
                    payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "access": c.access,
                             "department": c.department, "text": c.text},
                )
                for i, c in enumerate(chunks)
            ],
        )
        if self.url:  # a payload index speeds up filtering on a real server (no effect in :memory: mode)
            self.client.create_payload_index(self.collection, "access", models.PayloadSchemaType.KEYWORD)

    def search(self, query: str, k: int, allowed: set[str]) -> list[tuple[str, float]]:
        from qdrant_client import models

        flt = models.Filter(must=[models.FieldCondition(key="access", match=models.MatchAny(any=sorted(allowed)))])
        points = self.client.query_points(
            collection_name=self.collection, query=self.embedder.embed(query), limit=k,
            query_filter=flt, with_payload=True,
        ).points
        return [(p.payload["chunk_id"], float(p.score)) for p in points]


class BruteForceStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.rows: list[tuple[str, str, list[float]]] = []   # chunk_id, access, vector

    def index(self, chunks: list[Chunk]) -> None:
        self.rows = [(c.chunk_id, c.access, self.embedder.embed(c.search_text)) for c in chunks]

    def search(self, query: str, k: int, allowed: set[str]) -> list[tuple[str, float]]:
        q = self.embedder.embed(query)
        scored = [(cid, sum(a * b for a, b in zip(q, v))) for cid, access, v in self.rows if access in allowed]
        return sorted(scored, key=lambda x: -x[1])[:k]
