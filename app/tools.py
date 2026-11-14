"""Tools for the agent (Weeks 5 and 7): an allowlist, validated arguments and least privilege.

The model never calls Python directly. Every tool call goes through ``ToolRegistry.run``, which
rejects unknown tools, validates arguments, and refuses risky actions without human approval.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

ALLOWED_TOOLS = {"search_policy", "submit_leave_request"}
RISKY_TOOLS = {"submit_leave_request"}      # need human approval


class ToolNotAllowed(Exception):
    pass


class ApprovalRequired(Exception):
    pass


class LeaveRequest(BaseModel):
    employee: str = Field(min_length=1, max_length=80)
    days: int = Field(ge=1, le=60)
    leave_type: Literal["annual", "sick"] = "annual"
    start_date: date | None = None


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300)


class LeaveStore:
    """Stand-in for the HR system. Replace with a real database or API."""

    def __init__(self):
        self.requests: list[dict] = []

    def submit(self, req: LeaveRequest) -> dict:
        record = {"id": f"LR-{len(self.requests) + 1:04d}", **req.model_dump(mode="json"), "status": "submitted"}
        self.requests.append(record)
        return record


class ToolRegistry:
    def __init__(self, pipeline, store: LeaveStore):
        self.pipeline, self.store = pipeline, store

    def run(self, name: str, args: dict, roles=(), approved: bool = False):
        if name not in ALLOWED_TOOLS:
            raise ToolNotAllowed(f"Tool not allowed: {name}")
        if name in RISKY_TOOLS and not approved:
            raise ApprovalRequired(f"{name} needs human approval")
        try:
            if name == "search_policy":
                a = SearchArgs(**args)
                hits = self.pipeline.retriever.search(a.query, roles=roles, k=self.pipeline.top_k)
                return [{"doc_id": h.chunk.doc_id, "section": h.chunk.section, "text": h.chunk.text} for h in hits]
            req = LeaveRequest(**args)
            return self.store.submit(req)
        except ValidationError as exc:
            raise ValueError(f"Invalid arguments for {name}: {exc.errors()[0]['msg']}") from exc
