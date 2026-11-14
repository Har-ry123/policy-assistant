"""FastAPI service (Week 8): /health, /ask, /stream, /agent and /agent/approve.

Roles come from the ``X-User-Roles`` header (comma separated) for this demo. In production, take
them from a verified token (OIDC / JWT), never from a client-controlled header.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .graph import build_graph, resume_agent, run_agent
from .rag import RAGPipeline, build_pipeline
from .tools import LeaveStore, ToolRegistry


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    blocked: bool = False
    usage: dict = {}
    trace: dict = {}


class AgentRequest(BaseModel):
    message: str = Field(min_length=3, max_length=500)
    user: str = Field(default="employee", max_length=80)
    thread_id: str | None = None


class ApproveRequest(BaseModel):
    thread_id: str
    approve: bool


def parse_roles(header: str | None) -> list[str]:
    return [r.strip().lower() for r in (header or "").split(",") if r.strip()]


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Company Policy Assistant")
    pipeline = pipeline or build_pipeline()
    store = LeaveStore()
    graph = build_graph(pipeline, ToolRegistry(pipeline, store))
    app.state.pipeline, app.state.store, app.state.graph = pipeline, store, graph

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(pipeline.retriever.chunks), "llm": pipeline.llm.name}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, x_user_roles: str | None = Header(default=None)):
        try:
            ans = pipeline.ask(req.question, roles=parse_roles(x_user_roles), mode=req.mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            raise HTTPException(status_code=502, detail="The model service is unavailable.")
        return AskResponse(answer=ans.answer, sources=ans.sources, blocked=ans.blocked, usage=ans.usage, trace=ans.trace)

    @app.get("/stream")
    def stream(q: str, x_user_roles: str | None = Header(default=None)):
        try:
            ans = pipeline.ask(q, roles=parse_roles(x_user_roles))
        except Exception:
            raise HTTPException(status_code=502, detail="The model service is unavailable.")

        def gen():   # note: sends the finished answer word by word; true token streaming needs stream=True on the model call
            for word in ans.answer.split(" "):
                yield word + " "
        return StreamingResponse(gen(), media_type="text/plain")

    @app.post("/agent")
    def agent(req: AgentRequest, x_user_roles: str | None = Header(default=None)):
        thread_id = req.thread_id or uuid.uuid4().hex
        try:
            out = run_agent(graph, req.message, user=req.user, roles=parse_roles(x_user_roles), thread_id=thread_id)
        except Exception:
            raise HTTPException(status_code=502, detail="The model service is unavailable.")
        return {"thread_id": thread_id, **out}

    @app.post("/agent/approve")
    def approve(req: ApproveRequest):
        try:
            return {"thread_id": req.thread_id, **resume_agent(graph, req.thread_id, req.approve)}
        except Exception:
            raise HTTPException(status_code=404, detail="No pending approval for this thread.")

    @app.get("/leave-requests")
    def leave_requests():
        return store.requests

    return app


def app_factory() -> FastAPI:   # uvicorn app.api:app_factory --factory
    return create_app()
