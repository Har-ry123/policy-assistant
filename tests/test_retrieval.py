from app.chunking import chunk_all, load_documents, parse_document
from app.config import Settings
from app.rag import build_pipeline
from app.retrieval import MODES, allowed_access, rrf
from app.text import sentences, stem, tokens


def test_stem_and_tokens():
    assert stem("days") == "day" and stem("approved") == stem("approve")
    assert "the" not in tokens("the leave days") and tokens("Leaves")[0] == tokens("leave")[0]


def test_sentences_split_bullets_and_periods():
    assert sentences("- One. Two.\n- Three") == ["One.", "Two.", "Three"]


def test_parse_document_front_matter():
    doc = parse_document("---\nid: x\ntitle: X\naccess: HR\n---\n## A\nHello.")
    assert (doc.id, doc.access) == ("x", "hr") and "Hello." in doc.text


def test_chunks_follow_sections_and_keep_metadata():
    docs = load_documents(__import__("app.config", fromlist=["ROOT"]).ROOT / "data" / "policies")
    chunks = chunk_all(docs)
    assert len(chunks) >= len(docs) * 2
    assert all(c.chunk_id.startswith(c.doc_id) for c in chunks)
    assert {c.access for c in chunks} >= {"all", "hr", "engineering"}


def test_rrf_prefers_items_high_in_both_lists():
    merged = [cid for cid, _ in rrf([["d3", "d1", "d7"], ["d1", "d5", "d3"]])]
    assert merged == ["d1", "d3", "d5", "d7"]


def test_all_modes_find_the_leave_policy(pipeline):
    for mode in MODES:
        hits = pipeline.retriever.search("How many days of annual leave do I get?", mode=mode, k=3)
        assert hits and hits[0].chunk.doc_id == "leave-policy", mode


def test_restricted_documents_are_hidden_without_role(pipeline):
    for mode in MODES:
        docs = {h.chunk.doc_id for h in pipeline.retriever.search("salary band level 3", mode=mode, roles=(), k=10)}
        assert "salary-bands" not in docs, mode


def test_restricted_documents_visible_with_role(pipeline):
    hits = pipeline.retriever.search("salary band level 3", mode="hybrid", roles=("HR",), k=3)
    assert hits[0].chunk.doc_id == "salary-bands"


def test_roles_do_not_leak_across_roles(pipeline):
    hits = pipeline.retriever.search("pull request approval", mode="hybrid", roles=("hr",), k=10)
    assert "engineering-onboarding" not in {h.chunk.doc_id for h in hits}
    assert allowed_access(["HR", " "]) == {"all", "hr"}


def test_brute_force_store_matches_qdrant_top_hit():
    s = Settings(groq_api_key=None)
    qd, bf = build_pipeline(s), build_pipeline(s, store_kind="brute")
    q = "What is the daily meal limit for an international trip?"
    assert qd.retriever.search(q, "vector", k=1)[0].chunk.chunk_id == bf.retriever.search(q, "vector", k=1)[0].chunk.chunk_id
