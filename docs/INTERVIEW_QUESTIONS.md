# Policy Assistant: Interview Question Bank

Every answer below matches the code in this repository. Answer with this pattern: **what it is, what I did, the trade-off, how I measured it.**

---

## 1. Project overview

**Q1. Tell me about this project in 60 seconds.**
It's a RAG assistant that answers employee questions about company policies and cites its sources. I chunk 20 policy documents into 87 passages and store them in Qdrant. Retrieval runs vector search and BM25 together, merges them with Reciprocal Rank Fusion, and can rerank the result. Access control happens inside retrieval, so HR-only or engineering-only documents never reach the model for the wrong user. A LangGraph agent handles leave requests: a review step, then a pause for human approval before anything is saved. I measured it on a 71-question golden set: 0% leaks, 100% refusal accuracy, 0.91 correctness. It runs as a FastAPI service in Docker, deployed on Render with Groq's `gpt-oss-20b`.

**Q2. What problem does it solve?**
Employees ask HR the same policy questions over and over. The assistant answers them instantly, with a citation they can check, and never shows confidential documents to people who shouldn't see them.

**Q3. Walk me through what happens when a user asks a question.**
1. **Input guard:** length limit (500 chars), prompt-injection patterns, self-harm check, English-only check, PII redaction.
2. **Query cleanup:** typo repair (`fix_typos`, using `difflib` against the policy vocabulary) and synonym expansion ("wfh" becomes "work from home").
3. **Retrieval:** role filter, vector top-20 plus BM25 top-20, RRF fusion, optional rerank, top-4 kept.
4. **Prompt:** the fixed system prompt comes first, then numbered context, then the question.
5. **LLM:** Groq at `temperature=0`, with retries.
6. **Output guard:** the answer must cite a real source number, otherwise it's replaced with the "I couldn't find that" message; PII is redacted again.
7. **Response:** answer, sources, tokens, cost, latency and a trace of every step.

**Q4. What's your tech stack and why?**
FastAPI (async, Pydantic validation, auto docs), Qdrant (payload filtering plus HNSW), LangGraph (state, branching, `interrupt` for human-in-the-loop), Groq (fast and cheap inference for open models), sentence-transformers (`all-MiniLM-L6-v2` embeddings), Docker, GitHub Actions CI, plain HTML/JS frontend (no build step).

**Q5. What was the hardest part?**
Pick one real story: (a) the model writing citations as `【1】`, which made the output check wrongly reject good answers; (b) the PII filter hiding the company's own hotline number; (c) the model refusing questions whose answer was split across two passages, fixed by a prompt change that raised correctness from 0.89 to 0.96.

**Q6. What would you do differently?**
Start with a bigger golden dataset and real embeddings from day one, use an LLM-as-judge for faithfulness, and build verified identity (JWT) into the design instead of a demo role header.

---

## 2. RAG fundamentals

**Q7. What is RAG and why not just ask the LLM?**
Retrieval-Augmented Generation fetches relevant documents and puts them in the prompt. The LLM doesn't know Acme's policies, and they change over time. RAG gives up-to-date, citable, access-controlled answers with no retraining.

**Q8. RAG vs fine-tuning?**
RAG adds **knowledge** (facts that change). Fine-tuning changes **behaviour or style** (format, tone, consistent refusals). I use RAG for the facts. `eval/finetune_export.py` exports chat-format data in case I later want to fine-tune refusal style.

**Q9. What are the main failure modes of RAG?**
Retrieval misses (the right chunk isn't found), chunking problems (the answer is split across chunks), the model ignoring context or hallucinating, the model over-refusing, and context leaks across permission boundaries. My evaluation measures each one separately: retrieval metrics, correctness, faithfulness, refusal accuracy and leak rate.

**Q10. How do you reduce hallucination?**
The prompt says to answer only from the numbered context, never guess, and reply with an exact refusal sentence when the context doesn't help. The output must cite `[n]` with a valid source number or it's replaced. `temperature=0`. And I measure faithfulness.

**Q11. What's the "I don't know" problem?**
Over-refusal. My model refused "report harassment anonymously" because "anonymous" and "harassment" were in different passages. I changed the prompt to allow combining passages and matching everyday wording to policy wording.

---

## 3. Chunking and embeddings

**Q12. How do you chunk documents?**
One chunk per `## section` (structure-aware). Sections longer than 600 characters are split on sentence boundaries with a **one-sentence overlap**. Each chunk keeps metadata: `doc_id`, `title`, `department`, `access`, `section`.

**Q13. Why section-based chunking instead of fixed-size?**
Policy documents already have meaningful sections. Fixed-size chunks can cut a rule in half. Section chunks keep each rule whole and make citations meaningful ("Leave Policy, Annual leave").

**Q14. Why the overlap?**
So a sentence that depends on the one before it doesn't lose its context at a chunk boundary.

**Q15. What text do you actually embed?**
`search_text` = title + section + text. Adding the title and section helps retrieval: a chunk saying "24 days per year" becomes findable for "annual leave".

**Q16. How do you pick chunk size?**
It's a trade-off. Small chunks are precise but lose context; large chunks carry noise and cost more tokens. 600 characters fits one policy rule, and with top-4 the prompt stays small. I'd tune it with the eval harness.

**Q17. What embedding model do you use?**
`all-MiniLM-L6-v2` (384 dimensions, fast, runs on CPU). There's also an offline `HashEmbedder` (512-dim hashed bag of words) so tests and CI run with no downloads.

**Q18. What did switching to real embeddings change?**
Precision@3 went from 0.70 to 0.81 for hybrid (on the 32-question set), and it fixed synonym misses like "doctor's note" vs "medical certificate".

**Q19. Cosine similarity vs dot product vs Euclidean?**
Cosine compares direction and ignores length. With normalised vectors all three give the same ranking. MiniLM is trained for cosine.

**Q20. How would you pick a better embedding model?**
Check the MTEB leaderboard for retrieval tasks, then compare candidates on *my own* golden set. Weigh dimension size (storage, speed), context length, latency and licence.

---

## 4. Vector database (Qdrant, HNSW)

**Q21. Why Qdrant?**
Payload filtering during search (which I need for access control), HNSW indexing, an in-memory mode for tests, and a server mode for production. The `VectorStore` protocol lets me swap it out; `BruteForceStore` proves that.

**Q22. What is HNSW?**
Hierarchical Navigable Small World: a layered proximity graph. Search starts in the sparse top layer and greedily moves closer to the query, descending layer by layer. It's approximate nearest-neighbour search with roughly logarithmic query time.

**Q23. What do `m=16` and `ef_construct=100` mean?**
`m` is the number of links per node: more links give better recall but use more memory. `ef_construct` is the candidate list size while building the index: higher means a better graph but slower indexing. `ef` at query time trades recall for speed.

**Q24. For 87 passages, do you even need HNSW?**
No. Brute force is instant at this size, which is why `BruteForceStore` exists. HNSW matters at hundreds of thousands of vectors and up. I configured it to show I understand the production setup.

**Q25. Pre-filtering vs post-filtering?**
Post-filtering (search, then drop forbidden results) can return fewer than k results and is a security risk if someone forgets the filter. Qdrant filters *during* the HNSW search on the `access` payload, so you always get k allowed results.

**Q26. Other vector databases you considered?**
FAISS (a library, not a database, no filtering), Chroma (simple, good for prototypes), pgvector (if you already run Postgres), Pinecone (managed, costs money), Weaviate and Milvus (scale). Qdrant has good filtering, is open-source and runs locally.

---

## 5. Hybrid search, RRF and reranking

**Q27. Why hybrid search?**
Vector search understands meaning ("time off" ≈ "leave"). BM25 catches exact terms (policy names, numbers, acronyms like "PTO"). Together they cover each other's blind spots.

**Q28. Explain BM25.**
A keyword ranking function: term frequency (with saturation, `k1`), inverse document frequency (rare words count more), and document length normalisation (`b`). I implemented it myself in `app/bm25.py`.

**Q29. What is Reciprocal Rank Fusion?**
`score(doc) = Σ 1 / (k + rank)` over each ranked list, with `k = 60`. It uses only ranks, so it doesn't matter that cosine scores and BM25 scores are on different scales. No tuning or normalisation needed.

**Q30. Why k=60 in RRF?**
It's the standard value from the original paper. It damps the advantage of rank 1 over rank 2, so a document ranked well in *both* lists beats one ranked first in only one.

**Q31. What is reranking and why do it?**
Two stages: fast retrieval for recall (20 candidates per retriever), then a slower, more accurate model scores the short list for precision. A cross-encoder reads the query and the passage *together*, so it's more accurate than a bi-encoder (separate embeddings).

**Q32. Bi-encoder vs cross-encoder?**
A bi-encoder embeds query and document separately, so document vectors can be precomputed: fast, scales to millions. A cross-encoder processes the pair jointly: more accurate, but you have to run it per pair at query time, so you only use it on the top-N.

**Q33. Your reranker made results worse. Why, and what did you do?**
The offline `LexicalReranker` is word-overlap based, so it threw away the better semantic ranking (precision@3 fell from 0.72 to 0.64). I kept hybrid as the best mode, reported the result honestly, and listed `cross-encoder/ms-marco-MiniLM-L-6-v2` as the next experiment. Lesson: measure, don't assume a component helps.

**Q34. How do you handle typos and synonyms?**
`fix_typos` uses `difflib.get_close_matches` (cutoff 0.84, same first letter, words of 5+ letters) against the vocabulary of the policies themselves. A small synonym map expands queries ("abroad" → "international", "pto" → "annual leave"). Real embeddings handle most synonyms; the map covers the remaining domain jargon.

**Q35. What is query expansion and what are other techniques?**
Adding related terms to the query. Other options: HyDE (embed a hypothetical answer), multi-query (the LLM writes several rewrites), and query decomposition for multi-part questions.

---

## 6. Evaluation

**Q36. How did you evaluate the system?**
A golden dataset (`eval/golden.jsonl`) of 71 questions: 58 answerable (including reworded and two-document questions), 6 out of scope, 7 access-leak attempts. I measure retrieval and answers separately.

**Q37. Explain your retrieval metrics.**
- **Hit rate:** was at least one correct chunk in the top-k?
- **Precision@k:** what fraction of the top-k is relevant?
- **Recall:** what fraction of the relevant chunks was found?
- **MRR:** mean of 1/rank of the first correct result (rewards ranking it first).

**Q38. Explain your answer metrics.**
- **Correctness:** expected keywords appear in the answer.
- **Faithfulness:** word-overlap proxy between the answer and the cited context.
- **Refusal accuracy:** out-of-scope questions get the refusal.
- **Leak rate:** restricted content appears for an unauthorised role (must be 0).

**Q39. What were your results?**
Hybrid: hit rate 1.00, precision@3 0.72, MRR 1.00. Answers with `gpt-oss-120b`: correctness 0.91, refusal accuracy 1.00, leak rate 0.00.

**Q40. Correctness dropped from 0.96 to 0.91. Did the system get worse?**
No. The dataset got harder (71 vs 32 questions, more paraphrases). Of the 4 misses, 3 were wording ("three days" vs "3 days"), which I fixed in the test keywords. One was a real miss ("Can I fly business class?"), where retrieval works but the model sometimes refuses.

**Q41. Why evaluate retrieval separately from generation?**
To locate the failure. If retrieval found the passage and the answer is still wrong, it's the prompt or the model. If retrieval missed, no prompt will fix it.

**Q42. Weaknesses of your evaluation?**
Small dataset (one question ≈ 0.02 correctness), keyword-match correctness is brittle, faithfulness is a word-overlap proxy, and the LLM isn't perfectly repeatable even at temperature 0. I'd add an LLM-as-judge, more adversarial questions, and multiple runs with confidence intervals.

**Q43. What is LLM-as-judge and its risks?**
A strong LLM grades answers against a rubric. Risks: bias toward longer or its own outputs, position bias, cost, and it needs its own validation against human labels.

**Q44. What are RAGAS / TruLens?**
Frameworks for RAG metrics (faithfulness, answer relevance, context precision and recall), mostly LLM-judged. I wrote my own to understand the metrics, but I'd consider them for scale.

**Q45. What's the smoke test?**
`eval/smoke_check.py` hits a running server with about 40 questions (normal, reworded, role-restricted, attacks, PII, out of scope) plus the leave flows. Last run: 56/57 passed. It found two real bugs (`【1】` citations, the hotline number being redacted).

---

## 7. Security and guardrails

**Q46. How do you stop users from seeing restricted documents?**
Every chunk has an `access` field (`all`, `hr`, `engineering`). Retrieval only searches chunks where `access ∈ {"all"} ∪ user_roles`. The LLM **never sees** restricted text, so no prompt trick can make it leak. Tested for every role pair and every retrieval mode.

**Q47. Why not just tell the LLM "don't reveal HR documents"?**
Because prompts aren't a security boundary. Prompt injection can override them. Never rely on the model to hide data it has been given.

**Q48. How do you handle prompt injection?**
Defence in depth:
1. Regex input check ("ignore previous instructions", "reveal system prompt", "developer mode", "verbatim", "DAN"...).
2. The system prompt says the context is data, not instructions.
3. Access control in retrieval.
4. A tool allowlist with Pydantic-validated arguments.
5. Human approval before any write action.
6. Output must cite a source.

The regex layer is weak by design; layers 3 to 5 do the real work.

**Q49. Direct vs indirect prompt injection?**
Direct: the user types the attack. Indirect: the attack hides inside retrieved documents or tool output. That's why the prompt marks context as untrusted data and why risky actions need a human.

**Q50. How do you handle PII?**
Regex redaction for email, card number and phone, applied **before** the prompt and again on the output. The card pattern runs before the phone pattern so a card number isn't mislabelled (a bug the tests caught). Contact details that appear in the policy documents themselves are kept, so the hotline number still shows.

**Q51. What other guardrails exist?**
Self-harm messages get a caring reply with the counselling number instead of a policy lookup. Non-English input gets a clear "please ask in English" reply. A 500-character limit. Requests to dump the system prompt or the passages are blocked.

**Q52. The role comes from a header. Isn't that insecure?**
Yes, and the README says so. It's a demo. In production roles come from a verified identity token (JWT/OIDC from the company SSO), checked server-side.

**Q53. How do you prevent XSS in the frontend?**
Model output is always inserted as text nodes, never as `innerHTML`. A test enforces it.

**Q54. What's the OWASP Top 10 for LLMs?**
Prompt injection, insecure output handling, training data poisoning, model DoS, supply chain, sensitive information disclosure, insecure plugin design, excessive agency, overreliance, model theft. My project addresses injection, disclosure, output handling and excessive agency (approval gate).

---

## 8. Agents and LangGraph

**Q55. What does the agent do?**
It routes a message: question → RAG answer; leave request → draft → review → **human approval** → submit or reject.

**Q56. Draw the graph.**
`START → guard → classify → (answer_question → END) | (draft_leave → review → human_approval → submit_leave | reject_leave → END)`. Invalid drafts and blocked inputs go straight to END.

**Q57. Why LangGraph instead of a plain function or a LangChain agent?**
The flow branches, validates, **pauses for a human, then resumes later**, possibly in another HTTP request. LangGraph gives explicit state, conditional edges, `interrupt()` and a checkpointer. A free-running ReAct agent is unpredictable for an action that writes data.

**Q58. How does `interrupt` work?**
`interrupt()` stops the graph and saves its state in the checkpointer under a `thread_id`. The API returns `pending_approval` and the `thread_id`. `/agent/approve` resumes with `Command(resume="approve")`, and the graph continues from that exact node.

**Q59. What checkpointer do you use? Is it production-ready?**
`InMemorySaver`, so state is lost on restart, and with several workers a resume could land on a different process. For production: `PostgresSaver` or `SqliteSaver`.

**Q60. How does classify decide the intent?**
With function calling. The model gets two tools, `answer_policy_question` and `request_leave`, and must call one; the leave details (days, annual or sick, start date) are the tool's arguments, validated by a Pydantic model. That handles phrasings regex can't, like "book me off sick for three days". The model isn't trusted blindly: a written date wins, a named weekday is recomputed in code (the 20b model turned "next Monday" into a Saturday), the entitlement limits still apply, and if the model fails or returns invalid arguments, the old rules take over (a leave verb plus a day count, not ending in "?"). Offline, only the rules run.

**Q61. What does the reviewer agent check?**
Two layers. Fixed rules always add the notice needed (2 working days up to 3 days, otherwise 5), HR approval for annual leave over 10 days, and a medical certificate for sick leave over 2 days. Then an AI reviewer reads the leave-policy passages and returns JSON notes plus a recommendation; every note must cite a real passage or it's dropped. If the AI step fails, the rule notes alone are shown.

**Q62. How are tools secured?**
`ToolRegistry`: an allowlist (an unknown tool raises `ToolNotAllowed`), arguments validated by Pydantic (`LeaveRequest`), and write tools raise `ApprovalRequired` unless `approved=True`. Least privilege.

**Q63. What is ReAct?**
Reason + Act: the LLM alternates thoughts, tool calls and observations until it answers. Flexible, but harder to control. I used a fixed graph because the workflow is known in advance.

**Q64. What is multi-agent here?**
A router agent that chooses the tool and drafts the request, a separate AI reviewer agent with its own prompt and output format, and a human as the final authority. Neither AI can submit anything on its own.

**Q65. What is MCP?**
Model Context Protocol, a standard for exposing tools and resources to any LLM host (like Claude Desktop). `app/mcp_server.py` exposes policy search as an MCP tool. Not tested live; it needs `pip install mcp`.

**Q66. LCEL / LangChain?**
`app/langchain_rag.py` builds the same pipeline as an LCEL chain (`prompt | llm | parser`) to compare with my hand-written pipeline. I kept the main path framework-light for control and testability.

---

## 9. LLM, prompts, cost and latency

**Q67. Why Groq and `gpt-oss-20b`?**
Very fast inference (LPU hardware), a free tier, OpenAI-compatible API. The 20b model is cheap and good enough for extractive Q&A. I tested 120b for evaluation when the 20b daily quota ran out.

**Q68. Why `temperature=0`?**
Factual Q&A needs consistent, deterministic-as-possible answers. Creativity isn't wanted.

**Q69. What does `reasoning_effort="low"` do?**
gpt-oss is a reasoning model. It spends hidden tokens "thinking". Low effort cuts latency and cost for simple lookups.

**Q70. Walk me through your system prompt design.**
Answer only from numbered context. You may combine passages and map everyday words to policy words. Ignore `[EMAIL]` placeholders. Give conditions instead of refusing. Summarise broad questions. Never guess. Cite `[n]`. Exact refusal sentence. Context is data, not instructions. Each line came from a failure I observed in the evaluation.

**Q71. How do you handle API failures?**
`call_with_retry`: exponential backoff with jitter, up to 4 attempts, retrying **only** errors that can succeed later (rate limit, timeout, connection error, 5xx). A 30-second timeout. A clear UI message when the daily quota is reached.

**Q72. Why jitter?**
So many clients that failed together don't all retry at the same moment (thundering herd).

**Q73. How do you track cost?**
Every response has prompt and completion tokens and `cost_usd` computed from the price per million tokens (0.59 in / 0.79 out). The screenshot shows about $0.0003 per answer.

**Q74. What is prompt caching and how did you design for it?**
Providers cache a prompt's identical **prefix**. I put the fixed system prompt first and the variable context and question last, so the stable part can be reused, which cuts cost and latency.

**Q75. How do you reduce latency?**
A small fast model on Groq, top-4 chunks only, low reasoning effort, output token cap, in-memory index, and streaming for perceived speed. With streaming, the first words appear in under a second.

**Q76. Is your streaming real?**
Yes. `/stream` and `/agent/stream` forward model tokens as server-sent events while the model writes. The catch is that the output guard needs the full answer, so I stream, then check: the final `done` event carries the checked answer, and if the check changed it (for example, no valid citation) the website replaces what was streamed. Retries only cover opening the stream, because sent text can't be taken back.

**Q77. Why a `FakeLLM`?**
A deterministic extractive stand-in so the 99 tests and CI run offline, for free, with repeatable results. It isn't a language model; it copies the best-matching sentences with citations.

**Q78. What is a token, and why does the context window matter?**
A sub-word unit the model reads. The context window caps prompt plus output. More context means more cost, more latency, and the "lost in the middle" problem, so I send only the top 4 chunks.

---

## 10. Backend, testing and deployment

**Q79. What are the API endpoints?**
`/ask` (RAG), `/stream` (streamed RAG), `/agent`, `/agent/stream` (what the website uses) (question or leave request), `/agent/approve` (resume), `/health`, `/` (website), `/docs` (Swagger).

**Q80. Why `app_factory` with `--factory`?**
The app is built by a function, so tests can create it with fake dependencies (FakeLLM, an in-memory store) without global state.

**Q81. How did you test it?**
99 pytest tests: retrieval modes, streaming, the AI router and reviewer with a scripted model (including every fallback), access control per mode, guardrails (injection, PII, card-vs-phone), the LangGraph approval flow (`test_leave_request_waits_for_human_then_submits`), API endpoints via the test client, the LangChain chain with a fake chat model, and the XSS-safety check. CI runs them on every push via GitHub Actions.

**Q82. How do you mock the LLM in tests?**
Dependency injection: the pipeline takes any object that implements the `LLM` protocol, so tests pass `FakeLLM`.

**Q83. How is it deployed?**
Docker image (`requirements.txt` + sentence-transformers) on Render. `docker-compose.yml` adds a real Qdrant server with a persistent volume and a health check.

**Q84. Why is the Docker image large, and how would you shrink it?**
sentence-transformers pulls in PyTorch. Options: CPU-only torch wheels, ONNX Runtime or `fastembed`, a hosted embedding API, multi-stage builds.

**Q85. Render free tier cold starts?**
Free instances sleep when idle, so the first request is slow (the embedding model loads too). Fixes: a paid instance, pre-baking the model into the image, or a keep-alive ping.

**Q86. How do you manage secrets?**
`.env` (git-ignored), with `.env.example` committed. On Render, environment variables. Never hard-coded.

**Q87. What is observability in your project?**
`app/tracing.py`: each step is a span with timing and attributes (guard, search, generate, check), returned in the response and shown in the UI timeline. Production: OpenTelemetry, Langfuse or LangSmith.

---

## 11. Scaling and production

**Q88. How would you scale to 100,000 documents and 10,000 users?**
Qdrant server/cloud with HNSW and sharding, an ingestion pipeline (incremental re-embedding when a document changes), batch embedding, a Postgres checkpointer and leave store, Redis caching of frequent questions, horizontal API replicas behind a load balancer, per-user rate limiting, async LLM calls.

**Q89. How do you keep the index up to date when a policy changes?**
Hash each document; on change, delete its chunks by `doc_id` and re-insert. Store a version or effective date in the payload, and show "last updated" in citations.

**Q90. How would you add multi-tenancy (many companies)?**
A `tenant_id` payload filter (or a collection per tenant), enforced server-side from the auth token, just like the role filter.

**Q91. How would you add conversation memory?**
Store chat history per thread, and rewrite follow-ups into standalone queries ("what about sick leave?" → "How many days of sick leave…") before retrieval.

**Q92. Where could this system fail in production?**
Weak roles from a header, in-memory state lost on restart, regex guardrails bypassed by paraphrase, LLM quota limits, stale documents, an English-only embedder.

**Q93. How would you support other languages?**
A multilingual embedder (e.g. `multilingual-e5`, `bge-m3`), language detection, and answering in the user's language. Right now non-English is politely refused because the search models are English-only.

**Q94. How would you monitor quality after launch?**
Thumbs up/down feedback, log refusals and low-relevance retrievals, sample conversations for LLM-judge scoring, alerts on leak checks and the error rate, and add failures to the golden set.

---

## 12. Behavioural and project story questions

**Q95. Tell me about a bug you found and fixed.**
Some models wrote citations as `【1】` instead of `[1]`, so my output check thought the answer had no citation and replaced correct answers with "I don't know". The smoke test caught it. I normalised citation styles and special spaces before checking.

**Q96. Tell me about a trade-off.**
Offline lexical baselines (fast, free, deterministic) vs real models (better quality). I kept both behind interfaces (`Embedder`, `VectorStore`, `LLM`) so tests run offline and production uses real models.

**Q97. Tell me about a result that surprised you.**
The reranker made precision worse. I expected it to help. It taught me to measure every component instead of assuming.

**Q98. How did you decide what to build first?**
The core RAG path with an offline fake model and tests, then evaluation, then security, then the agent, then the UI and deployment. Each step was measurable before moving on.

**Q99. What did you learn?**
Retrieval quality matters more than the model. Security must be architectural, not in the prompt. Evaluation needs honest reporting, including when a change hurts.

**Q100. Why should we trust your numbers?**
The README reports what was **not** verified (real Qdrant server, Docker Compose, the cross-encoder, MCP), notes that one question ≈ 0.02 correctness, and says which results came from which model. Every number is reproducible with `python -m eval.run_eval`.

---

## Quick-fire numbers to memorise

| Item | Value |
|---|---|
| Documents / passages | 20 / 87 |
| Golden questions | 71 (58 answerable, 6 out of scope, 7 leak attempts) |
| Chunk size / overlap | 600 chars / 1 sentence |
| Candidates per retriever / top_k | 20 / 4 |
| RRF k | 60 |
| HNSW | m=16, ef_construct=100 |
| Embedder | all-MiniLM-L6-v2 (384-dim); offline hash 512-dim |
| Reranker | lexical (offline), cross-encoder/ms-marco-MiniLM-L-6-v2 (optional) |
| LLM | Groq gpt-oss-20b (eval also on 120b), temperature 0 |
| Hybrid retrieval | hit 1.00, P@3 0.72, MRR 1.00 |
| Answers | correctness 0.91, refusal 1.00, leaks 0.00 |
| Tests | 99 pytest, CI on GitHub Actions |
| Smoke test | 56/57 |
| Max question length | 500 chars |
| Retries | 4 attempts, exponential backoff + jitter |
