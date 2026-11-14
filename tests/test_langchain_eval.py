import pytest

from eval.run_eval import load_golden, retrieval_metrics


def test_langchain_chain_runs_with_a_fake_chat_model(pipeline):
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from app.langchain_rag import build_chain

    model = GenericFakeChatModel(messages=iter([AIMessage(content="24 days. [1]")]))
    chain = build_chain(pipeline.retriever, model)
    assert chain.invoke("How many days of annual leave do I get?") == "24 days. [1]"


def test_golden_dataset_is_well_formed():
    rows = load_golden()
    assert len(rows) >= 30
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert {r["type"] for r in rows} == {"normal", "out_of_scope", "denied"}
    assert all(r["keywords"] for r in rows if r["type"] == "normal")


def test_retrieval_quality_floor(pipeline):
    """A regression gate: fail the build if retrieval drops below a floor (Week 8 CI idea)."""
    rows = load_golden()
    for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        assert retrieval_metrics(pipeline, rows, mode)["hit_rate"] >= 0.9, mode
