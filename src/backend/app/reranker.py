from __future__ import annotations

from typing import Any

import httpx

from .config import settings
from .indexer import _SYNC_HTTP_CLIENT


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """使用重排序模型对文档进行重排序。

    Args:
        query: 查询字符串
        documents: 待重排序的文档列表，每项包含 content 和 metadata
        top_k: 返回的最大文档数，默认使用 settings.rerank_top_k

    Returns:
        按相关性排序后的文档列表，包含 relevance_score
    """
    if not documents:
        return []

    if not settings.rerank_model:
        return documents[: top_k or settings.rerank_top_k]

    top_k = top_k or settings.rerank_top_k

    doc_texts = [doc.get("content", "") for doc in documents]

    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": doc_texts,
        "top_n": top_k,
        "return_documents": True,
    }

    try:
        client = _SYNC_HTTP_CLIENT or httpx.Client(timeout=30.0)
        response = client.post(
            f"{settings.rerank_base_url}/rerank",
            headers={
                "Authorization": f"Bearer {settings.rerank_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        reranked = []
        for item in result.get("results", []):
            idx = item.get("index", 0)
            if 0 <= idx < len(documents):
                doc = documents[idx].copy()
                doc["relevance_score"] = item.get("relevance_score", 0.0)
                if "document" in item and item["document"]:
                    doc["content"] = item["document"]
                reranked.append(doc)

        return reranked

    except Exception:
        return documents[:top_k]
