# Interview talking points for this project

Use the pattern: **define, example, trade-off, how you measured it.** Everything below is something you can point to in the code.

## The 60-second pitch
"I built a policy assistant that answers employee questions with citations. Documents are chunked and stored in Qdrant; retrieval combines vector and BM25 search with reciprocal rank fusion and a reranker. Access is enforced in retrieval by role, so restricted HR or engineering documents never reach the model for the wrong user. An agent built with LangGraph handles leave requests with a reviewer step and human approval before anything is submitted. I measured retrieval and answers on a 71-question golden dataset over 20 documents, wrote security tests for injection and data leaks, and deployed it as a FastAPI service in Docker."

## Questions you should expect, and where the answer lives

| Question | Short answer | Code |
|---|---|---|
| Why hybrid search? | Vector search captures meaning; BM25 catches exact terms. RRF merges using ranks only, so scores on different scales do not matter. | `app/retrieval.py` |
| Why rerank? | The first stage is fast and approximate; a reranker scores the short list more carefully (precision). | `app/rerank.py` |
| How do you stop data leaks? | Filter by role inside retrieval, before the prompt. Do not rely on the model to hide anything. | `app/retrieval.py`, `tests/test_guardrails_security.py` |
| How do you handle prompt injection? | Layers: input check, treat context as data in the system prompt, least-privilege tools, approval for risky actions, output must cite a source. No single filter is enough. | `app/guardrails.py`, `app/tools.py` |
| Why LangGraph here? | The leave flow has branches, validation, a pause for a human and resume later. State plus checkpointer make that explicit. | `app/graph.py` |
| What does `interrupt` do? | Pauses the graph, saves state in the checkpointer, and resumes with the human's decision. | `app/graph.py`, `test_leave_request_waits_for_human_then_submits` |
| How do you evaluate RAG? | Golden dataset; retrieval metrics first (hit rate, precision, recall, MRR), then answer metrics. Change one thing at a time. | `eval/run_eval.py` |
| What did the evaluation show? | With 20 documents and 71 questions the modes separated: hybrid search had the best precision@3 (0.72 versus 0.68 for vector and 0.61 for keyword-only) and perfect ranking, while my offline lexical reranker lowered precision (0.64), so I kept hybrid and listed the neural cross-encoder as the next experiment. Answer correctness was 0.91 with the strict checker; re-asking the misses showed three of four were wording differences, which I fixed in the test data, and one real miss where the model refused although search found the passage. Refusals stayed at 100% and leaks at 0%. | `eval/results.md` |
| How do you control cost and latency? | Output limit, trimmed context (top-k chunks), stable prompt prefix for caching, token and cost in every response, streaming endpoint. | `app/rag.py`, `app/api.py` |
| What would you do differently in production? | Real embeddings, larger dataset, LLM judge, verified identity for roles, persistent checkpointer and database, per-user rate limits, tracing backend. | README "Limitations" |
| When would you fine-tune? | To change behaviour or style (for example consistent refusals), not to add facts; RAG handles facts. | `eval/finetune_export.py` |

## Stories to prepare
1. **A bug the tests caught** (for example: payload filter, card number mislabelled as a phone number, or an answer without a citation).
2. **A trade-off you made** (offline baseline vs real model; lexical reranker vs cross-encoder).
3. **A result you measured yourself** after running the evaluation with real embeddings: say the before and after numbers and what you changed.
