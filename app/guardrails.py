"""Guardrails (Week 7): input checks, PII redaction and output checks.

These are layers, not a complete defence. Pattern checks are easy to bypass, so access control,
least-privilege tools and human approval do the real work (see tools.py and graph.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import IDK

MAX_QUESTION_CHARS = 500
REFUSAL = "I can't help with that request."

PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),   # does not swallow a trailing full stop
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),   # before PHONE so cards are not mislabelled
    "PHONE": re.compile(r"\+?\d[\d\s().-]{8,}\d"),
}
INJECTION_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"disregard .{0,40}(instructions|rules)",
        r"(reveal|show|print|repeat) .{0,40}(system|hidden) prompt",
        r"\byou are now\b",
        r"\bdeveloper mode\b",
        r"act as .{0,40}(without|no) (restrictions|rules|limits)",
        r"(repeat|print|show|reveal|dump|output|list) .{0,30}(text|words|everything|instructions?|prompt) (above|before)",
        r"(print|show|reveal|dump|output|repeat) .{0,30}(all )?(the )?(passages|context|instructions)( you (were|are) given)?\b.{0,20}(verbatim|word for word|exactly)?",
        r"\bverbatim\b",
        r"(forget|drop) (everything|all) .{0,30}(told|instructions|rules)",
        r"\bjailbreak\b|(?-i:\bDAN\b)",
    )
]
SELF_HARM = re.compile(
    r"\b(hurt|harm|kill)(ing)? my ?self\b|\bsuicid|\bend (my|this) life\b|\bwant to die\b|\bself[- ]harm\b", re.I)
ENGLISH_ONLY = "Sorry, I can only answer questions written in English. Please ask your question in English."
SUPPORT = ("I'm really sorry you're going through this, and you don't have to face it alone. If you are in danger, "
           "please call your local emergency number now. Acme also offers 6 free confidential counselling sessions a year "
           "(call 1800-555-0142), and you can talk to HR or your manager.")
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reason: str = ""


def redact(text: str, keep=()) -> str:
    """Hide personal data. Values listed in ``keep`` (for example a company phone number that
    comes from the policy documents) are left alone."""
    keep = set(keep)

    def hide(label: str):
        def replace(match) -> str:
            return match.group(0) if match.group(0) in keep else f"[{label}]"
        return replace

    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(hide(label), text)
    return text


def pii_in(text: str) -> set:
    """All email / phone / card-like strings found in a piece of text."""
    return {m.group(0) for pattern in PII_PATTERNS.values() for m in pattern.finditer(text)}


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def check_input(text: str) -> GuardResult:
    text = (text or "").strip()
    if len(text) < 3:
        return GuardResult(False, text, "question too short")
    if len(text) > MAX_QUESTION_CHARS:
        return GuardResult(False, text, "question too long")
    if SELF_HARM.search(text):
        return GuardResult(False, SUPPORT, "sensitive")
    if looks_like_injection(text):
        return GuardResult(False, text, "possible prompt injection")
    if any(ch.isalpha() and ord(ch) > 127 for ch in text):   # the search models only understand English
        return GuardResult(False, ENGLISH_ONLY, "not english")
    return GuardResult(True, redact(text))


def clean_model_text(text: str) -> str:
    """Normalise what different models emit: full-width citation brackets and invisible special spaces."""
    text = re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)   # [1] written in full-width brackets
    for bad, good in ((" ", " "), (" ", " "), ("‑", "-"), ("‐", "-")):
        text = text.replace(bad, good)
    return text


def check_output(answer: str, n_sources: int, keep=()) -> GuardResult:
    """A grounded answer must cite a real source, or admit it does not know."""
    answer = redact(clean_model_text(answer).strip(), keep)
    if answer.startswith(IDK):
        return GuardResult(True, answer)
    cited = [int(n) for n in _CITATION.findall(answer)]
    if not cited or any(n < 1 or n > n_sources for n in cited):
        return GuardResult(False, IDK, "answer has no valid citation")
    return GuardResult(True, answer)
