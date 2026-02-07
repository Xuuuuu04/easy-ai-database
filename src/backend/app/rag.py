"""检索增强生成（RAG）相关辅助方法。"""

from __future__ import annotations

from typing import Any, Optional

from llama_index.core import VectorStoreIndex

from .config import settings
from .indexer import get_llm
from .db import search_chunks, list_chunks


def _enforce_citations(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """确保满足引用要求。

    Args:
        answer: 候选答案文本。
        citations: 候选引用列表。

    Returns:
        包含答案与引用的负载（或拒答信息）。
    """
    if settings.require_citations and len(citations) < settings.min_citations:
        return {
            "answer": "抱歉，当前知识库中没有足够的依据支持该问题的回答。请先导入相关资料。",
            "citations": [],
        }
    return {"answer": answer, "citations": citations}


def _build_citations_from_nodes(nodes) -> list[dict[str, Any]]:
    """从检索节点构建引用负载。

    Args:
        nodes: 检索到的节点或包装器。

    Returns:
        引用列表。
    """
    citations = []
    for node in nodes:
        base = getattr(node, "node", node)
        meta = base.metadata or {}
        citations.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "snippet": base.get_content()[:200],
            }
        )
    return citations


def _fallback_answer(question: str, hits: list[dict[str, Any]]) -> str:
    """基于原始分块命中生成兜底回答。

    Args:
        question: 用户问题。
        hits: 命中的分块记录。

    Returns:
        模型生成文本。
    """
    if not hits:
        return ""
    context = "\n\n".join(h["content"] for h in hits[: settings.max_context_chunks])
    llm = get_llm()
    prompt = (
        "请基于以下上下文回答问题，必须引用来源：\n"
        f"问题：{question}\n\n"
        f"上下文：\n{context}\n"
    )
    return llm.complete(prompt).text


def query_rag(index: Optional[VectorStoreIndex], question: str) -> dict[str, Any]:
    """执行 RAG 检索并返回响应负载。

    Args:
        index: 用于检索的向量索引。
        question: 用户问题。

    Returns:
        包含答案与引用的负载。
    """
    if settings.mock_mode or index is None:
        hits = search_chunks(question, limit=settings.max_context_chunks)
        citations = [
            {"source": "local", "page": h.get("page"), "snippet": h["content"][:200]}
            for h in hits
        ]
        answer = hits[0]["content"][:400] if hits else ""
        return _enforce_citations(answer, citations)

    retriever = index.as_retriever(similarity_top_k=settings.max_context_chunks)
    retrieved_nodes = retriever.retrieve(question)

    query_engine = index.as_query_engine(similarity_top_k=settings.max_context_chunks, llm=get_llm())
    response = query_engine.query(question)

    citations = _build_citations_from_nodes(retrieved_nodes)
    if not citations and getattr(response, "source_nodes", None):
        citations = _build_citations_from_nodes(response.source_nodes)

    if not citations:
        hits = search_chunks(question, limit=settings.max_context_chunks)
        if not hits:
            hits = list_chunks(limit=settings.max_context_chunks)
        citations = [
            {"source": "local", "page": h.get("page"), "snippet": h["content"][:200]}
            for h in hits
        ]
        answer = _fallback_answer(question, hits) or str(response)
        if settings.require_citations and len(citations) < settings.min_citations:
            return {
                "answer": answer,
                "citations": citations,
            }
        return {"answer": answer, "citations": citations}

    return _enforce_citations(str(response), citations)
