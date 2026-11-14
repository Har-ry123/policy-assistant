# Evaluation results

LLM: `fake-extractive` | store: `qdrant` | questions: 32 (27 answerable, 3 out of scope, 2 restricted)

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
| vector | 0.78 | 0.81 | 1.00 | 0.00 | 0.002467 |
| bm25 | 0.81 | 0.80 | 1.00 | 0.00 | 0.002148 |
| hybrid | 0.78 | 0.81 | 1.00 | 0.00 | 0.002509 |
| hybrid_rerank | 0.81 | 0.80 | 1.00 | 0.00 | 0.002493 |
