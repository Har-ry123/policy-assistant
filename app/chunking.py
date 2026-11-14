"""Loading and chunking policy documents (Week 3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .text import sentences


@dataclass
class Document:
    id: str
    title: str
    department: str
    access: str   # "all" or a role such as "hr" / "engineering"
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    department: str
    access: str
    section: str
    text: str

    @property
    def search_text(self) -> str:
        """What gets embedded / keyword-indexed: title and section help retrieval."""
        return f"{self.title} {self.section} {self.text}"


def parse_document(raw: str) -> Document:
    """Parse a markdown file with a small front-matter block."""
    lines = raw.strip().splitlines()
    meta: dict[str, str] = {}
    body_start = 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body_start = i + 1
                break
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    return Document(
        id=meta["id"], title=meta.get("title", meta["id"]), department=meta.get("department", ""),
        access=meta.get("access", "all").lower(), text="\n".join(lines[body_start:]).strip(),
    )


def load_documents(data_dir: Path) -> list[Document]:
    return [parse_document(p.read_text(encoding="utf-8")) for p in sorted(Path(data_dir).glob("*.md"))]


def chunk_document(doc: Document, max_chars: int = 600) -> list[Chunk]:
    """One chunk per '## section'; long sections are split by sentences with a 1-sentence overlap."""
    sections: list[tuple[str, list[str]]] = []
    for line in doc.text.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip():
            sections.append((doc.title, [line]))
    chunks: list[Chunk] = []
    for section, body_lines in sections:
        sents = sentences(" ".join(body_lines))
        pieces, cur = [], []
        for s in sents:
            if cur and len(" ".join(cur + [s])) > max_chars:
                pieces.append(cur)
                cur = [cur[-1], s]          # overlap: repeat the last sentence
            else:
                cur.append(s)
        if cur:
            pieces.append(cur)
        for j, piece in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{doc.id}#{len(chunks)}", doc_id=doc.id, title=doc.title,
                department=doc.department, access=doc.access, section=section, text=" ".join(piece),
            ))
    return chunks


def chunk_all(docs: list[Document], max_chars: int = 600) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, max_chars)]
