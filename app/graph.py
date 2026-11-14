"""The agent workflow as a LangGraph graph (Weeks 5 and 6).

guard -> classify -+-> answer_question -> END
                   +-> draft_leave -> review (second agent) -> human_approval -+-> submit_leave -> END
                                                                              +-> reject_leave -> END

* State is shared; each node returns only the keys it changes.
* ``review`` is a second, specialised agent (compliance reviewer): the director graph hands the
  draft to it and it adds policy notes for the human approver.
* ``human_approval`` uses ``interrupt()``: the run pauses and the checkpointer stores the state
  until someone resumes it with ``Command(resume="approve" | "reject")``.
"""
from __future__ import annotations

import re
from datetime import date
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .guardrails import REFUSAL, check_input
from .tools import LeaveRequest, ToolRegistry

ANNUAL_LIMIT, SICK_LIMIT, HR_THRESHOLD = 24, 10, 10
_LEAVE_VERB = re.compile(r"\b(apply for|request|book|take|submit|raise|need)\b.*\bleave\b", re.I)
_DAYS = re.compile(r"(\d+)\s*(?:working\s+|sick\s+|annual\s+|casual\s+|paid\s+)?days?", re.I)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


class AgentState(TypedDict, total=False):
    message: str
    user: str
    roles: list
    intent: str
    answer: str
    sources: list
    leave: dict
    review: list
    approved: bool
    status: str
    reason: str
    result: dict
    usage: dict
    trace: dict


def parse_leave(message: str, user: str) -> tuple[dict | None, str]:
    """Turn free text into a validated leave request, or explain what is wrong."""
    m = _DAYS.search(message)
    if not m:
        return None, "Please say how many days of leave you want."
    leave_type = "sick" if "sick" in message.lower() else "annual"
    d = _ISO_DATE.search(message)
    try:
        start = date.fromisoformat(d.group(1)) if d else None
        req = LeaveRequest(employee=user or "employee", days=int(m.group(1)), leave_type=leave_type, start_date=start)
    except ValueError as exc:
        return None, f"I could not read that request: {exc}"
    limit = SICK_LIMIT if leave_type == "sick" else ANNUAL_LIMIT
    if req.days > limit:
        return None, f"{req.days} days is above the {leave_type} leave entitlement of {limit} days per year."
    return req.model_dump(mode="json"), ""


def review_notes(leave: dict) -> list[str]:
    """Compliance-reviewer agent: checks the draft against the leave policy rules."""
    notes = []
    days = leave["days"]
    notes.append("Notice needed: 2 working days." if days <= 3 else "Notice needed: 5 working days.")
    if leave["leave_type"] == "annual" and days > HR_THRESHOLD:
        notes.append("Longer than 10 consecutive days: HR approval is also required.")
    if leave["leave_type"] == "sick" and days > 2:
        notes.append("Sick leave over 2 consecutive days needs a medical certificate.")
    return notes


def build_graph(pipeline, registry: ToolRegistry, checkpointer=None):
    def guard(state: AgentState):
        g = check_input(state["message"])
        if not g.allowed:
            return {"status": "blocked", "answer": REFUSAL, "reason": g.reason}
        return {}

    def classify(state: AgentState):
        wants_leave = bool(_LEAVE_VERB.search(state["message"]) and _DAYS.search(state["message"]))
        return {"intent": "leave_request" if wants_leave else "question"}

    def answer_question(state: AgentState):
        ans = pipeline.ask(state["message"], roles=state.get("roles", []))
        return {"answer": ans.answer, "sources": ans.sources, "status": "answered", "usage": ans.usage, "trace": ans.trace}

    def draft_leave(state: AgentState):
        leave, problem = parse_leave(state["message"], state.get("user", ""))
        if problem:
            return {"status": "invalid_request", "answer": problem}
        return {"leave": leave}

    def review(state: AgentState):
        return {"review": review_notes(state["leave"])}

    def human_approval(state: AgentState):
        decision = interrupt({"action": "submit_leave_request", "request": state["leave"],
                              "review": state.get("review", [])})
        return {"approved": decision == "approve"}

    def submit_leave(state: AgentState):
        record = registry.run("submit_leave_request", state["leave"], roles=state.get("roles", []), approved=True)
        return {"status": "submitted", "result": record, "answer": f"Leave request {record['id']} submitted."}

    def reject_leave(state: AgentState):
        return {"status": "rejected", "answer": "The leave request was not approved."}

    g = StateGraph(AgentState)
    for name, fn in [("guard", guard), ("classify", classify), ("answer_question", answer_question),
                     ("draft_leave", draft_leave), ("review", review), ("human_approval", human_approval),
                     ("submit_leave", submit_leave), ("reject_leave", reject_leave)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", lambda s: "end" if s.get("status") == "blocked" else "classify",
                            {"end": END, "classify": "classify"})
    g.add_conditional_edges("classify", lambda s: "draft" if s["intent"] == "leave_request" else "answer",
                            {"draft": "draft_leave", "answer": "answer_question"})
    g.add_edge("answer_question", END)
    g.add_conditional_edges("draft_leave", lambda s: "end" if s.get("status") == "invalid_request" else "review",
                            {"end": END, "review": "review"})
    g.add_edge("review", "human_approval")
    g.add_conditional_edges("human_approval", lambda s: "submit" if s.get("approved") else "reject",
                            {"submit": "submit_leave", "reject": "reject_leave"})
    g.add_edge("submit_leave", END)
    g.add_edge("reject_leave", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def _summarise(result: dict) -> dict:
    pending = result.get("__interrupt__")
    if pending:
        return {"status": "pending_approval", "approval": pending[0].value}
    return {k: result.get(k) for k in ("status", "answer", "sources", "reason", "result", "usage", "trace") if result.get(k) is not None}


def run_agent(graph, message: str, user: str = "employee", roles=(), thread_id: str = "t1") -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    return _summarise(graph.invoke({"message": message, "user": user, "roles": list(roles)}, cfg))


def resume_agent(graph, thread_id: str, approve: bool) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    return _summarise(graph.invoke(Command(resume="approve" if approve else "reject"), cfg))
