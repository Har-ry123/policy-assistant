"""Tiny tracing (Week 7): one trace per request, one span per step, with timing and attributes."""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager

log = logging.getLogger("policy_assistant")


class Trace:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        record = {"name": name, **attrs}
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.spans.append(record)

    def to_dict(self) -> dict:
        return {"trace_id": self.id, "spans": self.spans, "total_ms": round(sum(s["ms"] for s in self.spans), 2)}

    def log(self) -> None:
        log.info(json.dumps(self.to_dict()))
