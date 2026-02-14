from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .db import get_conn


_LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
_CJK_SEGMENT_PATTERN = re.compile(r"[\u3400-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    normalized = text.lower().strip()
    if not normalized:
        return []

    tokens = _LATIN_TOKEN_PATTERN.findall(normalized)
    for segment in _CJK_SEGMENT_PATTERN.findall(normalized):
        tokens.extend(segment)
        if len(segment) > 1:
            tokens.extend(segment[idx : idx + 2] for idx in range(len(segment) - 1))
    return tokens


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Okapi | None:
    """从分块构建BM25索引。

    Args:
        chunks: 分块列表，每项包含 content

    Returns:
        BM25索引对象
    """
    if not chunks:
        return None

    tokenized_corpus = [_tokenize(chunk["content"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(
    query: str,
    chunks: list[dict[str, Any]],
    bm25_index: BM25Okapi | None,
    top_k: int = 10,
) -> list[tuple[dict[str, Any], float]]:
    """使用BM25检索分块。

    Args:
        query: 查询字符串
        chunks: 分块列表
        bm25_index: BM25索引
        top_k: 返回数量

    Returns:
        (分块, BM25分数) 列表
    """
    if not bm25_index or not chunks:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((chunks[idx], float(scores[idx])))

    return results


def reciprocal_rank_fusion(
    vector_results: list[tuple[dict[str, Any], float]],
    bm25_results: list[tuple[dict[str, Any], float]],
    k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """使用RRF融合向量检索和BM25结果。

    Args:
        vector_results: (分块, 相似度分数) 列表
        bm25_results: (分块, BM25分数) 列表
        k: RRF常数

    Returns:
        融合后的分块列表
    """
    scores: dict[int, float] = {}
    doc_map: dict[int, dict[str, Any]] = {}

    for rank, (doc, _) in enumerate(vector_results):
        doc_id = id(doc)
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = id(doc)
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
        doc_map[doc_id] = doc

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def hybrid_retrieve(
    query: str,
    index: Any,
    chunks: list[dict[str, Any]] | None = None,
    bm25_index: BM25Okapi | None = None,
    top_k: int = 6,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    candidate_multiplier: int = 2,
) -> list[dict[str, Any]]:
    """执行混合检索（向量 + BM25）。

    Args:
        query: 查询字符串
        index: 向量索引
        chunks: 所有分块（用于BM25）
        bm25_index: 预构建的BM25索引
        top_k: 返回数量

    Returns:
        融合后的分块列表
    """
    multiplier = max(1, candidate_multiplier)
    if not settings.enable_hybrid_search or not chunks or not bm25_index:
        retriever = index.as_retriever(similarity_top_k=top_k * multiplier)
        nodes = retriever.retrieve(query)
        return [node.node if hasattr(node, "node") else node for node in nodes][:top_k]

    retriever = index.as_retriever(similarity_top_k=top_k * multiplier)
    vector_nodes = retriever.retrieve(query)
    vector_results = [
        (
            node.node if hasattr(node, "node") else node,
            node.score if hasattr(node, "score") else 0.5,
        )
        for node in vector_nodes
    ]

    bm25_results = bm25_search(query, chunks, bm25_index, top_k=top_k * multiplier)

    fused = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=max(1, rrf_k),
        vector_weight=max(0.0, vector_weight),
        bm25_weight=max(0.0, bm25_weight),
    )
    return fused[:top_k]


def get_all_chunks_from_db(kb_id: int | None = 1) -> list[dict[str, Any]]:
    """从数据库获取所有分块。

    Returns:
        分块列表
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute("SELECT id, document_id, content, page FROM chunks")
    else:
        cur.execute(
            """
            SELECT c.id, c.document_id, c.content, c.page
            FROM chunks c
            INNER JOIN documents d ON d.id = c.document_id
            WHERE d.kb_id = ?
            """,
            (kb_id,),
        )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
