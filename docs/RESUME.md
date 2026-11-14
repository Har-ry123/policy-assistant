# Resume entry for this project

Only claim what you did and measured. The `[...]` values are placeholders: run `python -m eval.run_eval` with your real setup (Groq key, `EMBEDDER=minilm`, `RERANKER=cross-encoder`, a larger dataset) and replace them with your own numbers. Do not publish the offline baseline numbers as model results.

## Project entry

**Company Policy Assistant** | Python, Groq, Qdrant, LangGraph, LangChain, FastAPI, Docker | [GitHub link]

- Built a retrieval-augmented assistant that answers policy questions with cited sources and refuses out-of-scope requests (a polite "couldn't find that" reply), using Qdrant with HNSW and role-based payload filtering.
- Implemented hybrid retrieval (vector + BM25 with reciprocal rank fusion) and reranking; created a 71-question golden dataset over 20 documents and an evaluation harness reporting hit rate, precision, recall, MRR, faithfulness and correctness; hybrid search reached precision@3 of 0.72 versus 0.61 for keyword-only and 0.68 for vector search, with answer correctness of 0.91 (measured; re-score before quoting a newer number).
- Designed a LangGraph agent with a compliance-review step, validated Pydantic tool arguments, a tool allowlist and human approval (`interrupt`) before any leave request is submitted.
- Added guardrails and security tests (prompt injection, PII redaction, cross-role data leakage, forbidden tools): 56 automated tests run on every push (GitHub Actions workflow included).
- Deployed as a FastAPI service with streaming, tracing (timing, tokens, cost per request), Docker and a health check.

## One-line version (summary section)
AI engineer who builds measured RAG and agent applications with access control, guardrails and deployment.

## Skills block
Python, SQL | LLM APIs (Groq), prompt engineering, structured output (Pydantic), tool calling | RAG: embeddings, chunking, Qdrant, hybrid search, reranking, evaluation | LangChain, LangGraph, MCP | FastAPI, Docker, pytest, Git

## Checklist before you publish it
- [ ] Repo pushed to GitHub with this README and a short demo (GIF or video).
- [ ] Ran the evaluation with a real model and real embeddings; replaced every `[...]` above.
- [ ] No `.env` committed; `.env.example` present.
- [ ] You can explain every file in `app/` (see `docs/INTERVIEW.md`).
