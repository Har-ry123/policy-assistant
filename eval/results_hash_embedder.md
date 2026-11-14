# Evaluation results

LLM: `groq:openai/gpt-oss-20b` | store: `qdrant` | questions: 32 (27 answerable, 3 out of scope, 2 restricted)

## Retrieval (k=3)

| mode | hit rate | precision@3 | recall | MRR |
|---|---|---|---|---|
| vector | 1.00 | 0.69 | 1.00 | 0.96 |
| bm25 | 1.00 | 0.69 | 1.00 | 1.00 |
| hybrid | 1.00 | 0.70 | 1.00 | 0.98 |
| hybrid_rerank | 0.96 | 0.69 | 0.96 | 0.96 |

## Answers

| mode | correctness | faithfulness | refusal accuracy | leak rate | est. cost (USD) |
|---|---|---|---|---|---|
| vector | 0.89 | 0.70 | 1.00 | 0.00 | 0.00661 |
| bm25 | 0.89 | 0.68 | 1.00 | 0.00 | 0.005866 |
| hybrid | 0.89 | 0.70 | 1.00 | 0.00 | 0.0066 |
| hybrid_rerank | 0.89 | 0.69 | 1.00 | 0.00 | 0.006651 |
