"""Shared text helpers: tokenizer, light stemmer, sentence splitter."""
from __future__ import annotations

import re

STOP = set(
    """a an the of to for in on at by with from and or but if then than as is are was were be been
    being do does did can could should would will shall may might must i me my we our you your he she
    it its they their them this that these those what which who whom whose when where why how many much
    get gets got there here up out about into over after before also not no any each per""".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def stem(w: str) -> str:
    """A tiny rule-based stemmer (good enough for a baseline, not a linguistics tool)."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "sl":
        w = w[:-1]
    return w


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    for raw in _TOKEN.findall(text.lower()):
        if not keep_stop and raw in STOP:
            continue
        out.append(stem(raw))
    return out


def sentences(text: str) -> list[str]:
    parts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*# ").strip()
        if line:
            parts.extend(s.strip() for s in _SENT.split(line) if s.strip())
    return parts
