"""Evaluation (Week 4): compare retrieval modes and score answers on the golden dataset.

    python -m eval.run_eval                 # offline (FakeLLM + hash embedder)
    python -m eval.run_eval --store brute   # exact search instead of Qdrant/HNSW

Metrics
  retrieval (k=3):  hit rate, precision@k (chunks from relevant docs / k), recall (relevant docs
                    found / relevant docs), MRR (1 / rank of the first relevant chunk)
  answers:          correctness (all keywords present), faithfulness (share of answer words that
                    appear in the cited context; a cheap offline proxy, NOT an LLM judge),
                    refusal accuracy on out-of-scope questions, leak rate on restricted ones.

With the offline FakeLLM the answer metrics measure the retrieval pipeline, not a language model.
Run with GROQ_API_KEY set to evaluate a real model.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from app.config import Settings
from app.llm import IDK
from app.rag import build_pipeline
from app.retrieval import MODES
from app.text import tokens

ROOT = Path(__file__).resolve().parent
K = 3


def norm(text: str) -> str:
    """Real models emit non-breaking spaces/hyphens that look identical but do not match plain text."""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("‐", "-").replace("‑", "-").replace("–", "-").lower()


def load_golden() -> list[dict]:
    lines = (ROOT / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def has(answer: str, key: str) -> bool:
    """True if the (normalised) answer contains the key; a key like "3 days|three days" accepts either wording."""
    a = norm(answer)
    return any(norm(alt) in a for alt in key.split("|"))


def retrieval_metrics(pipe, rows: list[dict], mode: str) -> dict:
    hit = prec = rec = mrr = 0.0
    n = 0
    for r in rows:
        if r["type"] != "normal":
            continue
        n += 1
        relevant = set(r["relevant_docs"])
        hits = pipe.retriever.search(r["question"], mode=mode, roles=r["roles"], k=K)
        docs = [h.chunk.doc_id for h in hits]
        good = [d in relevant for d in docs]
        hit += any(good)
        prec += sum(good) / K
        rec += len(relevant & set(docs)) / len(relevant)
        mrr += next((1 / (i + 1) for i, g in enumerate(good) if g), 0.0)
    return {"hit_rate": hit / n, "precision@3": prec / n, "recall": rec / n, "mrr": mrr / n}


def faithfulness(answer: str, sources: list[dict]) -> float:
    words = set(tokens(answer.replace("[", " ").replace("]", " ")))
    if not words:
        return 1.0
    context = set(tokens(" ".join(s["text"] for s in sources)))
    return len(words & context) / len(words)


def answer_metrics(pipe, rows: list[dict], mode: str) -> dict:
    correct = faith = 0.0
    n_norm = refused = n_oos = leaks = n_den = 0
    cost = 0.0
    for r in rows:
        ans = pipe.ask(r["question"], roles=r["roles"], mode=mode)
        cost += ans.usage.get("cost_usd", 0.0)
        if r["type"] == "normal":
            n_norm += 1
            correct += all(has(ans.answer, k) for k in r["keywords"])
            faith += faithfulness(ans.answer, ans.sources)
        elif r["type"] == "out_of_scope":
            n_oos += 1
            refused += ans.answer.startswith(IDK)
        else:
            n_den += 1
            text = ans.answer + " " + " ".join(s["text"] for s in ans.sources)
            leaks += any(has(text, f) for f in r["forbidden"]) or any(
                s["doc_id"] == r["restricted_doc"] for s in ans.sources)
    return {"correctness": correct / n_norm, "faithfulness": faith / n_norm,
            "refusal_accuracy": refused / n_oos, "leak_rate": leaks / n_den, "est_cost_usd": round(cost, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="qdrant", choices=["qdrant", "brute"])
    ap.add_argument("--answer-modes", default=",".join(MODES),
                    help="comma list of search modes to run the answer checks for (they spend model tokens); 'none' skips them")
    ap.add_argument("--out", default="results.md", help="report file name inside eval/")
    args = ap.parse_args()
    answer_modes = [] if args.answer_modes == "none" else [m for m in args.answer_modes.split(",") if m]
    unknown = [m for m in answer_modes if m not in MODES]
    if unknown:
        raise SystemExit(f"Unknown mode(s): {unknown}. Choose from {list(MODES)}")
    pipe = build_pipeline(Settings.from_env(), store_kind=args.store)
    rows = load_golden()
    lines = ["# Evaluation results\n",
             f"LLM: `{pipe.llm.name}` | store: `{args.store}` | questions: {len(rows)} "
             f"({sum(r['type'] == 'normal' for r in rows)} answerable, "
             f"{sum(r['type'] == 'out_of_scope' for r in rows)} out of scope, "
             f"{sum(r['type'] == 'denied' for r in rows)} restricted)\n",
             "## Retrieval (k=3)\n", "| mode | hit rate | precision@3 | recall | MRR |", "|---|---|---|---|---|"]
    for mode in MODES:
        m = retrieval_metrics(pipe, rows, mode)
        lines.append(f"| {mode} | {m['hit_rate']:.2f} | {m['precision@3']:.2f} | {m['recall']:.2f} | {m['mrr']:.2f} |")
    if answer_modes:
        lines += ["", "## Answers\n", "| mode | correctness | faithfulness | refusal accuracy | leak rate | est. cost (USD) |",
                  "|---|---|---|---|---|---|"]
        for mode in answer_modes:
            a = answer_metrics(pipe, rows, mode)
            lines.append(f"| {mode} | {a['correctness']:.2f} | {a['faithfulness']:.2f} | "
                         f"{a['refusal_accuracy']:.2f} | {a['leak_rate']:.2f} | {a['est_cost_usd']} |")
    report = "\n".join(lines) + "\n"
    (ROOT / args.out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:     # a short message instead of a long traceback (for example Groq's daily token limit)
        name = type(exc).__name__
        hint = " The free daily limit of this model is probably used up; wait for the reset, or set MODEL= to another model." if name == "RateLimitError" else ""
        raise SystemExit(f"Evaluation stopped: {name}: {str(exc)[:240]}{hint}\nNothing was saved.")
