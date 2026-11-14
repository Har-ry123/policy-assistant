"""Export the golden dataset as chat-format JSONL for fine-tuning (Week 8, Section 29).

Fine-tuning changes BEHAVIOUR (tone, format, refusing out-of-scope questions); it is not how this
project learns the policy facts (that is RAG). The file this script writes has one example per line:
{"messages": [system, user, assistant]}. Check your provider's fine-tuning guide for the exact format
it accepts. Training itself needs a provider or GPU and is not run here.

    python -m eval.finetune_export
"""
from __future__ import annotations

import json
from pathlib import Path

from app.llm import IDK, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    out = ROOT / "finetune_chat.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["type"] not in ("normal", "out_of_scope"):
                continue
            answer = r.get("reference_answer") or (IDK if r["type"] == "out_of_scope" else None)
            if answer is None:
                continue          # never train on a row without a reference answer
            example = {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": answer},
            ]}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} examples to {out}")


if __name__ == "__main__":
    main()
