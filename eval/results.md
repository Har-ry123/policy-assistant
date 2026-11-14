# Evaluation results

LLM: `groq:openai/gpt-oss-20b` | store: `qdrant` | questions: 32 (27 answerable, 3 out of scope, 2 restricted)

## Retrieval (k=3)

| mode | hit rate | precision@3 | recall | MRR |
|---|---|---|---|---|
| vector | 1.00 | 0.79 | 1.00 | 1.00 |
| bm25 | 1.00 | 0.69 | 1.00 | 1.00 |
| hybrid | 1.00 | 0.81 | 1.00 | 1.00 |
| hybrid_rerank | 0.96 | 0.77 | 0.96 | 0.96 |

## Answers

| mode | correctness | faithfulness | refusal accuracy | leak rate | est. cost (USD) |
|---|---|---|---|---|---|
| vector | 0.93 | 0.71 | 1.00 | 0.00 | 0.00737 |
| bm25 | 0.96 | 0.73 | 1.00 | 0.00 | 0.006411 |
| hybrid | 0.96 | 0.76 | 1.00 | 0.00 | 0.007373 |
| hybrid_rerank | 0.93 | 0.73 | 1.00 | 0.00 | 0.007335 |
