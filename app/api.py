"""FastAPI service (Week 8): /health, /ask, /stream, /agent, /agent/stream and /agent/approve.

/stream and /agent/stream send real model tokens as server-sent events while the answer is written.

Roles come from the ``X-User-Roles`` header (comma separated) for this demo. In production, take
them from a verified token (OIDC / JWT), never from a client-controlled header.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .graph import build_graph, resume_agent, route_message, rule_route, run_agent
from .guardrails import check_input
from .rag import RAGPipeline, build_pipeline
from .retrieval import MODES
from .tools import LeaveStore, ToolRegistry


STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("policy_assistant")


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
    mode: str | None = None


class ApproveRequest(BaseModel):
    thread_id: str
    approve: bool


def parse_roles(header: str | None) -> list[str]:
    return [r.strip().lower() for r in (header or "").split(",") if r.strip()]


def sse(events: Iterator[dict]) -> StreamingResponse:
    """Server-sent events: one ``data: {json}`` line per event, with a blank line after each."""
    body = (f"data: {json.dumps(event)}\n\n" for event in events)
    return StreamingResponse(body, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def safe(events: Iterator[dict]) -> Iterator[dict]:
    """Once streaming has started the status code is sent, so a failure becomes an ``error`` event instead."""
    try:
        yield from events
    except Exception:
        log.exception("Streaming failed")
        yield {"type": "error", "detail": "The model service is unavailable."}


def check_mode(mode: str | None) -> None:
    if mode and mode not in MODES:
        raise HTTPException(status_code=422, detail="mode must be one of " + ", ".join(MODES))


def create_app(pipeline: RAGPipeline | None = None, use_ai: bool | None = None) -> FastAPI:
    app = FastAPI(title="Company Policy Assistant")
    pipeline = pipeline or build_pipeline()
    if use_ai is None:   # AGENT_AI=0 turns off the AI router and reviewer (saves tokens; the rules still work)
        use_ai = os.getenv("AGENT_AI", "1") != "0"
    store = LeaveStore()
    graph = build_graph(pipeline, ToolRegistry(pipeline, store), use_ai=use_ai)
    app.state.pipeline, app.state.store, app.state.graph = pipeline, store, graph

    def answer_events(question: str, roles, mode, router: str | None = None) -> Iterator[dict]:
        for event in safe(pipeline.ask_stream(question, roles=roles, mode=mode)):
            if event["type"] == "done":
                event = {**event, "status": "blocked" if event["blocked"] else "answered"}
                if router:
                    event["router"] = router
            yield event

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC_DIR / "index.html")

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
    def stream(q: str, mode: str | None = None, x_user_roles: str | None = Header(default=None)):
        """Real token streaming as server-sent events: sources, token, ..., done (see RAGPipeline.ask_stream)."""
        check_mode(mode)
        return sse(answer_events(q, parse_roles(x_user_roles), mode))

    @app.post("/agent/stream")
    def agent_stream(req: AgentRequest, x_user_roles: str | None = Header(default=None)):
        """The website's endpoint. Questions are streamed token by token; leave requests go through the agent
        graph and arrive as one ``agent`` event (they wait for approval anyway)."""
        check_mode(req.mode)
        roles = parse_roles(x_user_roles)
        thread_id = req.thread_id or uuid.uuid4().hex

        def events():
            guard = check_input(req.message)
            route = None
            if guard.allowed:
                route = route_message(pipeline.llm, req.message) if use_ai else rule_route(req.message)
            if route and route["intent"] == "question":
                yield from answer_events(req.message, roles, req.mode, router=route["router"])
                return
            # blocked messages and leave requests: the graph handles them (and pauses for approval)
            yield {"type": "agent", "thread_id": thread_id,
                   **run_agent(graph, req.message, user=req.user, roles=roles, thread_id=thread_id, mode=req.mode, route=route)}
        return sse(safe(events()))

    @app.post("/agent")
    def agent(req: AgentRequest, x_user_roles: str | None = Header(default=None)):
        if req.mode and req.mode not in MODES:
            raise HTTPException(status_code=422, detail="mode must be one of " + ", ".join(MODES))
        thread_id = req.thread_id or uuid.uuid4().hex
        try:
            out = run_agent(graph, req.message, user=req.user, roles=parse_roles(x_user_roles), thread_id=thread_id, mode=req.mode)
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
