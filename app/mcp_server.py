"""An MCP server that exposes policy search as a tool (Week 6, Section 24 of the handbook).

Install the MCP SDK (``pip install mcp``) and run:   python -m app.mcp_server
Then add it to an MCP host (for example Claude Desktop) with command ``python`` and args
``["-m", "app.mcp_server"]``.

The SDK renamed its server class between major versions (1.x ``FastMCP``, 2.x ``MCPServer``), so the
import below tries both. This file was written against the handbook's description and was NOT run
against a live MCP host while building this project; test it in your environment.
"""
from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP as _Server      # mcp 1.x
except ImportError:                                         # pragma: no cover
    from mcp.server.mcpserver import MCPServer as _Server   # mcp 2.x

from .rag import build_pipeline

mcp = _Server("policy-assistant")
_pipeline = build_pipeline()


@mcp.tool()
def search_policy(query: str) -> list[dict]:
    """Search the company policies (public documents only) and return the best matching passages."""
    hits = _pipeline.retriever.search(query, roles=(), k=4)      # no roles: restricted docs stay hidden
    return [{"doc": h.chunk.doc_id, "section": h.chunk.section, "text": h.chunk.text} for h in hits]


@mcp.tool()
def answer_question(question: str) -> str:
    """Answer a policy question with citations (public documents only)."""
    return _pipeline.ask(question).answer


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
