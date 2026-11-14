# Company Policy Assistant

A retrieval-augmented assistant that answers employee questions about company policies **with citations**, enforces **role-based access** to restricted documents, can raise a **leave request through an agent with human approval**, refuses unsafe requests, is **measured** on a golden dataset, and ships as a **FastAPI + Docker** service.

It is the capstone for the AI Engineering Study Handbook: every week of the course appears in the code (see the coverage table).

## Architecture

```
            +-----------+   +-----------+   +------------------------------+   +--------+   +-------------+
 question ->| guard in  |-->| retrieve  |-->| prompt (stable prefix first) |-->|  LLM   |-->| guard out   |--> answer + sources
            | PII, inj. |   | roles     |   +------------------------------+   | Groq / |   | must cite   |
            +-----------+   | vector +  |                                      | Fake   |   +-------------+
                            | BM25 RRF  |                                      +--------+
                            | + rerank  |         every step -> trace (timing, tokens, cost)
                            +-----+-----+
                                  |
                         Qdrant (HNSW, payload filter on "access")

 agent (LangGraph):  guard -> classify -+-> answer_question
                                        +-> draft_leave -> review agent -> HUMAN APPROVAL -> submit / reject
```

## Topic coverage (handbook weeks -> this project)

| Week | Topics | Where |
|---|---|---|
| 1 | venv, API keys, LLM call, system role, temperature, tokens, streaming, retries, cost | `app/config.py`, `app/llm.py` (Groq client, retry with backoff, `temperature=0`), `app/rag.py` (usage + cost), `/stream` in `app/api.py` |
| 2 | prompt engineering, structured output, prompt chaining | `SYSTEM_PROMPT` in `app/llm.py`; Pydantic `LeaveRequest` in `app/tools.py`; the guard -> retrieve -> generate -> check chain in `app/rag.py` |
| 3 | RAG, embeddings, chunking, Qdrant, filters, HNSW, vector-DB alternatives | `app/chunking.py`, `app/embeddings.py`, `app/store.py` (`QdrantStore` with HNSW config and an `access` filter; `BruteForceStore` shows swapping the backend) |
| 4 | hybrid search, reranking, RAG evaluation | `app/bm25.py`, `app/retrieval.py` (RRF), `app/rerank.py`, `eval/run_eval.py`, `eval/golden.jsonl` |
| 5 | ReAct / agents / tool use | `app/tools.py` (allowlist, validated arguments, approval gate), `app/graph.py` |
| 6 | LangChain, LangGraph, multi-agent, MCP | `app/langchain_rag.py` (LCEL chain), `app/graph.py` (state, nodes, conditional edges, `interrupt` + checkpointer, reviewer agent), `app/mcp_server.py` |
| 7 | observability, guardrails, security | `app/tracing.py`, `app/guardrails.py`, access control in `app/retrieval.py`, `tests/test_guardrails_security.py` |
| 8 | prompt caching, cost, deployment, fine-tuning, capstone | stable prompt prefix in `app/rag.py`; `app/api.py`, `Dockerfile`, `docker-compose.yml`; `eval/finetune_export.py`; this project |
| Bonus | interview, resume | `docs/INTERVIEW.md`, `docs/RESUME.md` |

## Quick start (works offline, no keys)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q               # 56 tests
python -m app.cli "How many days of annual leave do I get?" --trace
python -m app.cli "What is the salary band for level 3?" --roles hr
python -m eval.run_eval           # writes eval/results.md
uvicorn app.api:app_factory --factory --reload
```

Without `GROQ_API_KEY` the assistant uses `FakeLLM`, a deterministic **extractive** stand-in (it copies the best-matching context sentences and cites them). It is *not* a language model; it lets tests and evaluation run offline.

### Use the real model

```bash
copy .env.example .env            # then set GROQ_API_KEY=...
# optional, real semantic search and a neural reranker:
pip install sentence-transformers
# in .env: EMBEDDER=minilm  and  RERANKER=cross-encoder
```

### Windows helpers

If you get `No module named 'langgraph'`, the terminal is not using the project's virtual environment (the prompt has no `(.venv)`). These scripts always use the right Python:

```powershell
.\run_api.ps1      # start the API on port 8000
.\run_tests.ps1    # run the tests
```

## Website

Start the server and open **http://127.0.0.1:8000/**. It is one HTML file (`app/static/index.html`, plain HTML/CSS/JavaScript, no build step):

- **Sidebar:** role switch (Employee / HR / Engineer) with a hint of what each role can see, your name for leave requests, search settings (hybrid + rerank, hybrid, vector, keyword), the saved leave requests, a live server status line (model and number of passages), API docs and GitHub links, and a light/dark toggle.
- **Welcome screen:** eight example cards. Role-specific cards switch the role for you (for example *Salary band* switches to HR).
- **Answers:** clickable `[1]` citation chips that open and highlight the source passage, cited sources shown first with other retrieved passages folded away, timing / token / cost chips, and a *How this answer was produced* timeline (check question, search, write answer, check answer).
- **Leave requests:** an approval card with the request details and the reviewer's notes; the request is saved only after **Approve**.
- **Polish:** typing indicator, Enter to send (Shift+Enter for a new line), character counter, clear error messages (for example when the model's daily limit is reached), a phone layout with a slide-out menu, keyboard focus styles and reduced-motion support.

It was exercised in a real browser against a real model: role switching, the approval flow, the attack example, the citation chips, the theme toggle, the phone menu and keyboard sending. Model text is always inserted as text nodes (never as HTML), which a test enforces. The role switch is a demo; use verified identity in production.

## API

```bash
curl -s localhost:8000/ask -H "content-type: application/json" \
     -d '{"question":"How many days of annual leave do I get?"}'

# restricted document only for the right role (demo header; use a verified token in production)
curl -s localhost:8000/ask -H "content-type: application/json" -H "X-User-Roles: hr" \
     -d '{"question":"What is the salary band for level 3?"}'

# agent with human approval
curl -s localhost:8000/agent -H "content-type: application/json" \
     -d '{"message":"I want to take 3 days of leave from 2026-12-21","user":"asha"}'
# -> {"status":"pending_approval","thread_id":"...","approval":{...}}
curl -s localhost:8000/agent/approve -H "content-type: application/json" \
     -d '{"thread_id":"<id from above>","approve":true}'
```

## Evaluation

`python -m eval.run_eval` compares four retrieval modes on 32 questions (27 answerable, 3 out of scope, 2 restricted) and writes `eval/results.md`. Numbers from the **offline baseline** (hash embedder, extractive model, 43 chunks), copied from that file:

| mode | hit rate | precision@3 | recall | MRR |
|---|---|---|---|---|
| vector | 1.00 | 0.69 | 1.00 | 0.96 |
| bm25 | 1.00 | 0.69 | 1.00 | 1.00 |
| hybrid | 1.00 | 0.70 | 1.00 | 0.98 |
| hybrid_rerank | 0.96 | 0.69 | 0.96 | 0.96 |

**Real model** (`groq:openai/gpt-oss-20b`, same 32 questions, same offline embedder; retrieval numbers are identical because retrieval does not use the model):

| mode | correctness | faithfulness (word-overlap proxy) | refusal accuracy | leak rate |
|---|---|---|---|---|
| vector | 0.89 | 0.70 | 1.00 | 0.00 |
| bm25 | 0.89 | 0.68 | 1.00 | 0.00 |
| hybrid | 0.89 | 0.70 | 1.00 | 0.00 |
| hybrid_rerank | 0.89 | 0.69 | 1.00 | 0.00 |

**With semantic embeddings and the final prompt** (`EMBEDDER=minilm`, `all-MiniLM-L6-v2`, real model `openai/gpt-oss-20b`, same 32 questions):

| mode | precision@3 | MRR | correctness | faithfulness (proxy) | refusals | leaks |
|---|---|---|---|---|---|---|
| vector | 0.79 (was 0.69) | 1.00 (was 0.96) | 0.93 | 0.71 | 1.00 | 0.00 |
| bm25 | 0.69 (unchanged) | 1.00 | 0.96 | 0.73 | 1.00 | 0.00 |
| hybrid | **0.81** (was 0.70) | 1.00 (was 0.98) | **0.96** | 0.76 | 1.00 | 0.00 |
| hybrid_rerank (lexical reranker) | 0.77 | 0.96 | 0.93 | 0.73 | 1.00 | 0.00 |

What changed the numbers: (1) semantic embeddings raised precision@3 by about 0.10 and fixed most synonym misses in retrieval; (2) the model was refusing some questions whose answer WAS in the retrieved text (for example "report harassment anonymously": the passage said "anonymous Ethics Hotline" but the word "harassment" sat in another passage), so the system prompt now allows combining passages and matching everyday wording, which raised correctness from about 0.89 to 0.93 to 0.96 while out-of-scope refusals stayed at 100%. Keep in mind: with 27 answerable questions, one question is worth about 0.04, and the real model is not perfectly repeatable, so differences between modes of one question are noise (BM25 does not even use embeddings). The offline lexical reranker lowers precision (it ignores the better semantic ranking); the next experiment is `RERANKER=cross-encoder`. The one question still failing in most modes is "save work files on a USB stick" (retrieval does not surface the IT-security passage). Raw files: `eval/results_offline.md` (offline model), `eval/results_hash_embedder.md` and `eval/results_minilm_final.md` (real model).

Of the 27 answerable questions with the word-based embedder, 3 failed in every mode: the model said "I don't know" for "doctor's note if I am ill" (policy: "medical certificate", "sick leave"), "save files on a USB stick" and "report harassment anonymously". These are synonym misses of the offline word-based search, the gap that real embeddings (`EMBEDDER=minilm`) are meant to close; they are the next thing to measure. An earlier run scored 0.74 because the checker did not normalise the non-breaking spaces and hyphens that real models emit; the checker now does.

**Read these honestly.** The corpus is tiny (9 documents), so all modes score near the ceiling and the differences are within noise; the lexical reranker even lost one case. The point of the harness is the method: change one thing, re-run, compare. To see real differences, add many more documents and harder paraphrased questions, switch to `EMBEDDER=minilm`, and re-run. Answer metrics (correctness, faithfulness, refusal accuracy, leak rate) are in `eval/results.md`; faithfulness there is a cheap word-overlap proxy, not an LLM judge.

### Smoke test against a running server

`python -m eval.smoke_check 8000` asks about 40 questions (normal, reworded, role-restricted, attacks, out of scope, PII) and runs the leave-request flows (approve, reject, invalid, injected). Last run with the real model `openai/gpt-oss-120b`: 56 of 57 checks passed. The one miss, "Can I fly business class?", is a model-consistency issue: search finds the right passage every time, but the model sometimes answers "I don't know" even at temperature 0. The same 41 questions were also typed into the website itself and all gave correct answers.

Two real bugs were found this way and fixed: some models write citations as `【1】` instead of `[1]` (the output check then wrongly replaced good answers with "I don't know"; citations and invisible special spaces are now normalised), and the privacy filter was hiding the company's own contact details (hotline number, security email) from answers (details that appear in the policy documents are now kept; other personal data is still hidden). The prompt also gained two lines (everyday wording such as "abroad" means "international"; placeholders like `[EMAIL]` should be ignored). **Those last prompt changes were checked with the 120b model only: the daily free token limit of `openai/gpt-oss-20b` was used up, so re-run `python -m eval.run_eval` once your quota resets before quoting new evaluation numbers.**

**Groq free-tier note:** each model has a daily token limit. If the site shows "The model service is unavailable", you have probably hit it; wait for the reset or set `MODEL=openai/gpt-oss-120b` in `.env` (models have separate limits).

## Security

* Restricted documents (`access: hr`, `access: engineering`) are filtered **inside retrieval**, so the LLM never sees them for an unauthorised user. Tested for every retrieval mode.
* Inputs: length limits, prompt-injection phrase check, PII redaction before the prompt. These are weak layers by design; the strong layers are access control, a tool allowlist, validated arguments and human approval for risky actions.
* Outputs must cite a real source or admit ignorance; PII is redacted again.
* The role header in `app/api.py` is a demo. In production derive roles from a verified identity token.

## Deployment

Note: the Docker image installs only `requirements.txt`. If you set `EMBEDDER=minilm` there, add `sentence-transformers` to `requirements.txt` first (it makes the image much larger), and mind that `/stream` currently sends the finished answer word by word rather than streaming model tokens.

```bash
docker build -t policy-assistant .
docker run -p 8000:8000 --env-file .env policy-assistant
# with a real Qdrant server and persistent storage:
docker compose up --build
```

## What was and was not verified

Verified here: all 56 tests (retrieval modes, access control, guardrails, the LangGraph approval flow, the API, a LangChain chain with a fake chat model), the evaluation script (offline and with the real Groq model), the fine-tuning export, semantic search with `all-MiniLM-L6-v2`, a 27-check end-to-end run against a live server, and the GitHub Actions workflow (passed on the first push).

**Not run:** a real Qdrant server (the project uses Qdrant in memory), Docker / docker compose, the neural cross-encoder reranker, and the MCP server (it needs `pip install mcp` and a host). Test those in your environment before depending on them.

## Limitations and next steps

* Tiny corpus; add real documents and a larger, harder golden dataset.
* The offline embedder and reranker are lexical baselines and miss synonyms: "What is the daily meal limit abroad?" returns "I don't know" because the policy says "international". This is exactly the gap real embeddings (`EMBEDDER=minilm`) close, and a good case to add to the golden dataset.
* In-memory conversation state and leave store; use a database and a persistent LangGraph checkpointer for production.
* Add an LLM-as-judge for faithfulness, and rate limiting per user.
