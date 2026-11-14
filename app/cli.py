"""Command line demo:  python -m app.cli "How many leave days do I get?" --roles hr"""
from __future__ import annotations

import argparse
import json

from .rag import build_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the policy assistant a question.")
    ap.add_argument("question")
    ap.add_argument("--roles", default="", help="comma separated roles, e.g. hr,engineering")
    ap.add_argument("--mode", default="hybrid_rerank", choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    ap.add_argument("--trace", action="store_true", help="print the trace (timing per step)")
    args = ap.parse_args()
    pipe = build_pipeline()
    try:
        ans = pipe.ask(args.question, roles=[r for r in args.roles.split(",") if r], mode=args.mode)
    except Exception as exc:   # show a readable message instead of a long traceback
        raise SystemExit(f"The model call failed: {type(exc).__name__}: {str(exc)[:300]}\n"
                         "If the model was not found, set MODEL= in .env to a current model name.")
    print(ans.answer)
    for s in ans.sources:
        print(f"  [{s['n']}] {s['title']} / {s['section']}")
    if args.trace:
        print(json.dumps(ans.trace, indent=2))


if __name__ == "__main__":
    main()
