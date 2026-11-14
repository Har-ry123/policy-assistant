# Evaluation results

LLM: `groq:openai/gpt-oss-120b` | store: `qdrant` | questions: 71 (58 answerable, 6 out of scope, 7 restricted)

## Retrieval (k=3)

| mode | hit rate | precision@3 | recall | MRR |
|---|---|---|---|---|
| vector | 1.00 | 0.68 | 1.00 | 0.99 |
| bm25 | 1.00 | 0.61 | 1.00 | 0.97 |
| hybrid | 1.00 | 0.72 | 1.00 | 1.00 |
| hybrid_rerank | 0.98 | 0.64 | 0.98 | 0.97 |

## Answers

| mode | correctness | faithfulness | refusal accuracy | leak rate | est. cost (USD) |
|---|---|---|---|---|---|
| hybrid | 0.91 | 0.73 | 1.00 | 0.00 | 0.018439 |
| hybrid_rerank | 0.90 | 0.72 | 1.00 | 0.00 | 0.018372 |
