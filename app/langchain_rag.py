"""The same RAG flow written with LangChain (Week 6, Section 21 of the handbook).

Shows how the hand-written pipeline maps to LCEL: prompt | model | parser. Use ANY LangChain chat
model (for example ``ChatGroq`` from ``langchain-groq``) as ``chat_model``.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .llm import SYSTEM_PROMPT
from .retrieval import Retriever


def build_chain(retriever: Retriever, chat_model, roles=(), k: int = 4):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    def retrieve(question: str) -> str:
        hits = retriever.search(question, roles=roles, k=k)
        return "\n".join(f"[{i}] ({h.chunk.title}) {h.chunk.text}" for i, h in enumerate(hits, start=1))

    return (
        {"context": RunnableLambda(retrieve), "question": RunnableLambda(lambda q: q)}
        | prompt | chat_model | StrOutputParser()
    )
