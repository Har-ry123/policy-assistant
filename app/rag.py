"""The RAG pipeline (Weeks 2, 3, 4, 7, 8): guard -> retrieve -> generate -> check, with a trace.

Prompt-caching note (Week 8): SYSTEM_PROMPT is a fixed prefix and the retrieved context + question
come last, so providers that cache prompt prefixes can reuse the stable part.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from .bm25 import BM25
from .chunking import chunk_all, load_documents
from .config import Settings
from .embeddings import get_embedder
from .guardrails import REFUSAL, check_input, check_output, pii_in
from .llm import IDK, SYSTEM_PROMPT, FakeLLM, GroqLLM
from .rerank import get_reranker
from .retrieval import Hit, Retriever
from .store import BruteForceStore, QdrantStore
from .tracing import Trace

# Example prices per million tokens (input, output). Replace with your provider's current prices.
PRICE_PER_M = (0.59, 0.79)


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    blocked: bool = False
    reason: str = ""
    mode: str = ""
    usage: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n".join(
        f"[{i}] ({h.chunk.title}) {' '.join(h.chunk.text.split())}" for i, h in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},                       # stable prefix
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


class RAGPipeline:
    def __init__(self, llm, retriever: Retriever, top_k: int = 4, mode: str = "hybrid_rerank"):
        self.llm, self.retriever, self.top_k, self.mode = llm, retriever, top_k, mode

    def _prepare(self, question: str, roles, mode: str, trace: Trace):
        """Steps before the model: check the question, then search. Returns an early Answer, or (question, hits, sources)."""
        with trace.span("guard_input") as span:
            guard = check_input(question)
            span["allowed"] = guard.allowed
        if not guard.allowed:
            trace.log()
            message = guard.text if guard.reason in ("sensitive", "not english") else REFUSAL
            return Answer(message, blocked=True, reason=guard.reason, mode=mode, trace=trace.to_dict())
        guard.text = self.retriever.fix_typos(guard.text)

        with trace.span("retrieve", mode=mode) as span:
            hits = self.retriever.search(self.retriever.expand(guard.text), mode=mode, roles=roles, k=self.top_k)
            span["hits"] = len(hits)
        if not hits:
            trace.log()
            return Answer(IDK, sources=[], mode=mode, trace=trace.to_dict())
        sources = [{"n": i, "doc_id": h.chunk.doc_id, "title": h.chunk.title, "section": h.chunk.section,
                    "text": h.chunk.text, "score": round(h.score, 4)} for i, h in enumerate(hits, start=1)]
        return guard.text, hits, sources

    def _finish(self, text: str, hits, sources, prompt_tokens: int, completion_tokens: int, mode: str, trace: Trace) -> Answer:
        """Steps after the model: the answer must cite a real passage, and personal data is hidden again."""
        with trace.span("guard_output") as span:
            out = check_output(text, len(hits), keep=set().union(*(pii_in(h.chunk.text) for h in hits)))
            span["ok"] = out.allowed
        usage = {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "cost_usd": round((prompt_tokens * PRICE_PER_M[0] + completion_tokens * PRICE_PER_M[1]) / 1e6, 6),
        }
        trace.log()
        return Answer(out.text, sources=sources if out.text != IDK else [], reason=out.reason, mode=mode,
                      usage=usage, trace=trace.to_dict())

    def ask(self, question: str, roles=(), mode: str | None = None) -> Answer:
        mode = mode or self.mode
        trace = Trace()
        prepared = self._prepare(question, roles, mode, trace)
        if isinstance(prepared, Answer):
            return prepared
        text, hits, sources = prepared
        with trace.span("generate", model=self.llm.name) as span:
            result = self.llm.complete(build_messages(text, hits))
            span["prompt_tokens"], span["completion_tokens"] = result.prompt_tokens, result.completion_tokens
        return self._finish(result.text, hits, sources, result.prompt_tokens, result.completion_tokens, mode, trace)

    def ask_stream(self, question: str, roles=(), mode: str | None = None) -> Iterator[dict]:
        """Like ask(), but yields events while the model writes: ``sources``, then ``token`` events, then ``done``.

        The output check needs the whole answer, so it runs at the end ("stream, then check"). The ``done``
        event carries the checked answer; ``replaced`` is true when the check changed what was streamed
        (for example, no valid citation), and the client must then show the checked answer instead."""
        mode = mode or self.mode
        trace = Trace()
        prepared = self._prepare(question, roles, mode, trace)
        if isinstance(prepared, Answer):
            yield {"type": "done", **asdict(prepared), "replaced": False}
            return
        text, hits, sources = prepared
        yield {"type": "sources", "sources": sources}
        usage: dict = {}
        parts: list[str] = []
        with trace.span("generate", model=self.llm.name, streamed=True) as span:
            for token in self.llm.stream(build_messages(text, hits), usage):
                parts.append(token)
                yield {"type": "token", "text": token}
            span["prompt_tokens"] = usage.get("prompt_tokens", 0)
            span["completion_tokens"] = usage.get("completion_tokens", 0)
        raw = "".join(parts)
        answer = self._finish(raw, hits, sources, span["prompt_tokens"], span["completion_tokens"], mode, trace)
        yield {"type": "done", **asdict(answer), "replaced": answer.answer != raw.strip()}


def build_pipeline(settings: Settings | None = None, llm=None, store_kind: str = "qdrant") -> RAGPipeline:
    settings = settings or Settings.from_env()
    chunks = chunk_all(load_documents(settings.data_dir))
    embedder = get_embedder(settings.embedder)
    store = QdrantStore(embedder, settings.qdrant_url) if store_kind == "qdrant" else BruteForceStore(embedder)
    store.index(chunks)
    retriever = Retriever(chunks, store, BM25(chunks), get_reranker(settings.reranker))
    if llm is None:
        llm = GroqLLM(settings.groq_api_key, settings.model) if settings.groq_api_key else FakeLLM()
    return RAGPipeline(llm, retriever, top_k=settings.top_k)
