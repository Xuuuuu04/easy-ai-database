from __future__ import annotations

import re
from typing import Any, Generator, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .cache import semantic_cache
from .config import settings
from .db import get_chat_history, get_chat_summary, list_chunks, search_chunks
from .hybrid_search import build_bm25_index, get_all_chunks_from_db, hybrid_retrieve
from .indexer import get_llm
from .reranker import rerank_documents


STREAM_CHUNK_SIZE = 48

_SMALL_TALK_TEMPLATES = {
    "greeting": "你好！我是本机知识库助手。你可以直接问我知识库里的问题，我会基于本地资料回答并给出来源。",
    "presence": "我在。你可以直接提问，我会优先基于本地知识库给出可追溯答案。",
    "thanks": "不客气！如果你愿意，我可以继续帮你查知识库里的问题。",
}

_PLACEHOLDER_CHUNK_PATTERN = re.compile(r"^content\s+\d+$", re.IGNORECASE)
_TRAILING_PUNCTUATION_PATTERN = re.compile(r"[\?？!！。；;，,]+$")
_LEADING_NOISE_PATTERN = re.compile(r"^(请问|请你|麻烦|帮我|可以|能否|能不能|请)\s*")
_DEFINITION_PREFIX_PATTERN = re.compile(
    r"^(?:什么是|啥是|请介绍|介绍下|介绍一下|简要介绍|解释下|解释一下)(.+)$"
)
_TRAILING_SUFFIX_PATTERN = re.compile(
    r"(是什么|是啥|有哪些|有什么|怎么用|如何使用|如何|详解|介绍|简介)$"
)
_TRAILING_PARTICLE_PATTERN = re.compile(r"(吗|呢|呀|啊)$")


def _normalize_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(
        r"[\s\.,!?;:'\"，。！？；：、“”‘’()（）\[\]{}<>《》-]", "", cleaned
    )
    return cleaned


def get_smalltalk_answer(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return None

    greeting_tokens = {"你好", "您好", "嗨", "哈喽", "hello", "hi", "hey"}
    presence_tokens = {"在吗", "在么"}
    thanks_tokens = {"谢谢", "thanks", "thankyou"}

    if normalized in greeting_tokens:
        return _SMALL_TALK_TEMPLATES["greeting"]
    if normalized in presence_tokens:
        return _SMALL_TALK_TEMPLATES["presence"]
    if normalized in thanks_tokens:
        return _SMALL_TALK_TEMPLATES["thanks"]
    return None


def _split_stream_text(
    text: str, chunk_size: int = STREAM_CHUNK_SIZE
) -> Generator[str, None, None]:
    if not text:
        return
    size = max(1, chunk_size)
    for start in range(0, len(text), size):
        piece = text[start : start + size]
        if piece:
            yield piece


def _is_placeholder_chunk(content: str) -> bool:
    return bool(_PLACEHOLDER_CHUNK_PATTERN.fullmatch(content.strip()))


def _context_is_unusable(context_chunks: list[str]) -> bool:
    non_empty = [chunk for chunk in context_chunks if chunk.strip()]
    if not non_empty:
        return True
    return all(_is_placeholder_chunk(chunk) for chunk in non_empty)


def _filter_placeholder_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        hit
        for hit in hits
        if isinstance(hit.get("content"), str)
        and not _is_placeholder_chunk(hit["content"])
    ]


def _enforce_citations(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    if settings.require_citations and len(citations) < settings.min_citations:
        return {
            "answer": "抱歉，当前知识库中没有足够的依据支持该问题的回答。请先导入相关资料。",
            "citations": [],
        }
    return {"answer": answer, "citations": citations}


def _extract_content_and_metadata(node: Any) -> tuple[str, dict[str, Any]]:
    base = getattr(node, "node", node)

    if hasattr(base, "get_content"):
        content = base.get_content() or ""
        metadata = getattr(base, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return content, metadata

    if isinstance(base, dict):
        content = base.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        metadata: dict[str, Any] = {
            "source": base.get("source", "local"),
            "page": base.get("page"),
        }
        if base.get("document_id") is not None:
            metadata["document_id"] = base.get("document_id")
        return content, metadata

    return ("" if base is None else str(base), {})


def _build_citations_from_nodes(nodes) -> list[dict[str, Any]]:
    citations = []
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        citations.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "snippet": content[:200],
            }
        )
    return citations


def _build_context_with_history(
    question: str, chat_id: int | None, kb_id: int = 1
) -> str:
    if not settings.enable_multi_turn or chat_id is None:
        return question

    history = get_chat_history(chat_id, limit=settings.max_chat_history, kb_id=kb_id)
    if not history:
        return question

    summary = get_chat_summary(chat_id, kb_id=kb_id)
    if summary:
        return f"[对话摘要]{summary}\n\n当前问题: {question}"

    context_parts = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            context_parts.append(f"用户: {content}")
        elif role == "assistant":
            context_parts.append(f"助手: {content}")

    if context_parts:
        return f"历史对话:\n" + "\n".join(context_parts) + f"\n\n当前问题: {question}"

    return question


def _rewrite_queries(question: str) -> list[str]:
    text = question.strip()
    if not text:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        variants.append(normalized)

    _push(text)

    core = _TRAILING_PUNCTUATION_PATTERN.sub("", text).strip()
    core = _LEADING_NOISE_PATTERN.sub("", core).strip()
    definition_match = _DEFINITION_PREFIX_PATTERN.match(core)
    if definition_match:
        core = definition_match.group(1).strip()

    core = _TRAILING_SUFFIX_PATTERN.sub("", core).strip()
    core = _TRAILING_PARTICLE_PATTERN.sub("", core).strip()

    if core and core != text:
        _push(core)
        _push(f"{core} 简介")
        _push(f"{core} 是什么")
        _push(f"{core} 介绍")

    return variants


def _retrieve_candidates(
    index: VectorStoreIndex,
    query: str,
    bm25_index,
    chunks,
):
    if settings.enable_hybrid_search and bm25_index and chunks:
        return hybrid_retrieve(
            query,
            index,
            chunks,
            bm25_index,
            top_k=settings.rerank_top_k,
        )

    retriever = index.as_retriever(similarity_top_k=settings.rerank_top_k)
    return retriever.retrieve(query)


def _node_dedup_key(node: Any) -> str:
    content, meta = _extract_content_and_metadata(node)
    source = str(meta.get("source") or "")
    page = str(meta.get("page") or "")
    return f"{source}|{page}|{content[:200]}"


def _retrieve_with_rewrites(
    index: VectorStoreIndex,
    question: str,
    bm25_index,
    chunks,
) -> list[Any]:
    merged: list[Any] = []
    seen_keys: set[str] = set()
    limit = max(settings.rerank_top_k, settings.max_context_chunks) * 2

    for rewritten_query in _rewrite_queries(question):
        for node in _retrieve_candidates(index, rewritten_query, bm25_index, chunks):
            key = _node_dedup_key(node)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(node)
            if len(merged) >= limit:
                return merged

    return merged


def _iter_stream_chunks(stream) -> Generator[str, None, None]:
    emitted_text = ""
    for item in stream:
        delta = getattr(item, "delta", None)
        if isinstance(delta, str) and delta:
            emitted_text += delta
            yield from _split_stream_text(delta)
            continue

        text = getattr(item, "text", None)
        if not isinstance(text, str):
            text = "" if item is None else str(item)
        if not text:
            continue

        if emitted_text and text.startswith(emitted_text):
            chunk = text[len(emitted_text) :]
        else:
            chunk = text
        emitted_text = text
        if chunk:
            yield from _split_stream_text(chunk)


def _single_chunk_stream(text: str) -> Generator[str, None, None]:
    yield from _split_stream_text(text)


def _stream_answer_from_context(
    question: str, context: str
) -> Generator[str, None, None]:
    llm = get_llm()
    prompt = (
        "请基于以下上下文回答问题，必须引用来源：\n"
        f"问题：{question}\n\n"
        f"上下文：\n{context}\n"
    )
    yield from _iter_stream_chunks(llm.stream_complete(prompt))


def _apply_reranking(
    nodes: list[Any],
    query: str,
) -> list[Any]:
    if not nodes or not settings.rerank_model:
        return nodes

    docs = []
    for node in nodes:
        content, _ = _extract_content_and_metadata(node)
        if not content:
            continue
        docs.append(
            {
                "content": content[:1000],
                "node": node,
            }
        )

    if not docs:
        return nodes

    reranked = rerank_documents(query, docs, top_k=settings.max_context_chunks)
    return [doc["node"] for doc in reranked]


class RAGPipeline:
    def __init__(self):
        self._bm25_indexes: dict[int, Any] = {}
        self._chunks_cache: dict[int, list[dict[str, Any]]] = {}

    def _get_bm25_index(self, kb_id: int):
        if kb_id not in self._bm25_indexes or kb_id not in self._chunks_cache:
            chunks = get_all_chunks_from_db(kb_id=kb_id)
            self._chunks_cache[kb_id] = chunks
            self._bm25_indexes[kb_id] = build_bm25_index(chunks)
        return self._bm25_indexes.get(kb_id), self._chunks_cache.get(kb_id, [])

    def invalidate_cache(self, kb_id: int | None = None):
        if kb_id is None:
            self._bm25_indexes.clear()
            self._chunks_cache.clear()
            semantic_cache.clear()
            return

        self._bm25_indexes.pop(kb_id, None)
        self._chunks_cache.pop(kb_id, None)
        semantic_cache.clear(kb_id=kb_id)

    def query(
        self,
        index: Optional[VectorStoreIndex],
        question: str,
        chat_id: int | None = None,
        kb_id: int = 1,
    ) -> dict[str, Any]:
        smalltalk_answer = get_smalltalk_answer(question)
        if smalltalk_answer is not None:
            return {"answer": smalltalk_answer, "citations": []}

        enhanced_question = _build_context_with_history(question, chat_id, kb_id=kb_id)

        cached = semantic_cache.get(enhanced_question, kb_id=kb_id)
        if cached:
            return {"answer": cached["answer"], "citations": cached["citations"]}

        if settings.mock_mode or index is None:
            return self._mock_query(enhanced_question, kb_id=kb_id)

        bm25_index, chunks = self._get_bm25_index(kb_id)

        retrieved_nodes = _retrieve_with_rewrites(
            index,
            enhanced_question,
            bm25_index,
            chunks,
        )

        reranked_nodes = _apply_reranking(retrieved_nodes, enhanced_question)

        citations = _build_citations_from_nodes(
            reranked_nodes[: settings.max_context_chunks]
        )

        context_chunks = []
        for node in reranked_nodes[: settings.max_context_chunks]:
            content, _ = _extract_content_and_metadata(node)
            if content:
                context_chunks.append(content)

        if _context_is_unusable(context_chunks):
            context_chunks = []
            citations = []

        if not context_chunks:
            return {"answer": "未能检索到相关上下文。", "citations": []}

        context = "\n\n".join(context_chunks)
        llm = get_llm()
        prompt = (
            "请基于以下上下文回答问题，必须引用来源：\n"
            f"问题：{enhanced_question}\n\n"
            f"上下文：\n{context}\n"
        )
        response = llm.complete(prompt)
        answer = response.text

        if settings.require_citations and len(citations) < settings.min_citations:
            return {
                "answer": "抱歉，当前知识库中没有足够的依据支持该问题的回答。请先导入相关资料。",
                "citations": citations,
            }

        result = {"answer": answer, "citations": citations}
        semantic_cache.set(enhanced_question, answer, citations, kb_id=kb_id)
        return result

    def query_stream(
        self,
        index: Optional[VectorStoreIndex],
        question: str,
        chat_id: int | None = None,
        kb_id: int = 1,
    ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        smalltalk_answer = get_smalltalk_answer(question)
        if smalltalk_answer is not None:
            return _single_chunk_stream(smalltalk_answer), []

        enhanced_question = _build_context_with_history(question, chat_id, kb_id=kb_id)

        cached = semantic_cache.get(enhanced_question, kb_id=kb_id)
        if cached:
            return _single_chunk_stream(cached["answer"]), cached["citations"]

        if settings.mock_mode or index is None:
            return self._mock_query_stream(enhanced_question, kb_id=kb_id)

        bm25_index, chunks = self._get_bm25_index(kb_id)

        retrieved_nodes = _retrieve_with_rewrites(
            index,
            enhanced_question,
            bm25_index,
            chunks,
        )

        reranked_nodes = _apply_reranking(retrieved_nodes, enhanced_question)
        citations = _build_citations_from_nodes(
            reranked_nodes[: settings.max_context_chunks]
        )

        context_chunks = []
        for node in reranked_nodes[: settings.max_context_chunks]:
            content, _ = _extract_content_and_metadata(node)
            if content:
                context_chunks.append(content)

        if _context_is_unusable(context_chunks):
            context_chunks = []
            citations = []

        if not context_chunks:
            return _single_chunk_stream("未能检索到相关上下文。"), []

        context = "\n\n".join(context_chunks)

        def _stream_with_cache():
            chunks = []
            for chunk in _stream_answer_from_context(enhanced_question, context):
                chunks.append(chunk)
                yield chunk
            final_answer = "".join(chunks)
            semantic_cache.set(enhanced_question, final_answer, citations, kb_id=kb_id)

        return _stream_with_cache(), citations

    def _mock_query(self, question: str, kb_id: int = 1) -> dict[str, Any]:
        hits = search_chunks(question, limit=settings.max_context_chunks, kb_id=kb_id)
        if not hits:
            hits = list_chunks(limit=settings.max_context_chunks, kb_id=kb_id)
        hits = _filter_placeholder_hits(hits)
        if not hits:
            return {"answer": "未能检索到相关上下文。", "citations": []}
        citations = [
            {"source": "local", "page": h.get("page"), "snippet": h["content"][:200]}
            for h in hits
        ]
        answer = hits[0]["content"][:400] if hits else ""
        return _enforce_citations(answer, citations)

    def _mock_query_stream(
        self, question: str, kb_id: int = 1
    ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        hits = search_chunks(question, limit=settings.max_context_chunks, kb_id=kb_id)
        if not hits:
            hits = list_chunks(limit=settings.max_context_chunks, kb_id=kb_id)
        hits = _filter_placeholder_hits(hits)
        if not hits:
            return _single_chunk_stream("未能检索到相关上下文。"), []
        citations = [
            {"source": "local", "page": h.get("page"), "snippet": h["content"][:200]}
            for h in hits
        ]
        answer = hits[0]["content"][:400] if hits else ""
        return _single_chunk_stream(answer), citations


rag_pipeline = RAGPipeline()


def query_rag(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None = None,
    kb_id: int = 1,
) -> dict[str, Any]:
    return rag_pipeline.query(index, question, chat_id=chat_id, kb_id=kb_id)


def query_rag_with_chat(
    index: Optional[VectorStoreIndex], question: str, chat_id: int, kb_id: int = 1
) -> dict[str, Any]:
    return rag_pipeline.query(index, question, chat_id=chat_id, kb_id=kb_id)


def query_rag_stream(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None = None,
    kb_id: int = 1,
) -> Generator[str, None, None]:
    stream, _ = rag_pipeline.query_stream(index, question, chat_id=chat_id, kb_id=kb_id)
    yield from stream


def query_rag_stream_with_citations(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None = None,
    kb_id: int = 1,
) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
    return rag_pipeline.query_stream(index, question, chat_id=chat_id, kb_id=kb_id)


def invalidate_rag_cache(kb_id: int | None = None):
    rag_pipeline.invalidate_cache(kb_id=kb_id)


def retrieve_context(
    index: Optional[VectorStoreIndex],
    question: str,
    kb_id: int = 1,
    chat_id: int | None = None,
    top_k: int = 6,
) -> dict[str, Any]:
    effective_top_k = max(1, top_k)
    enhanced_question = _build_context_with_history(question, chat_id, kb_id=kb_id)
    rewrites = _rewrite_queries(enhanced_question)

    if settings.mock_mode or index is None:
        hits = search_chunks(enhanced_question, limit=effective_top_k, kb_id=kb_id)
        if not hits:
            hits = list_chunks(limit=effective_top_k, kb_id=kb_id)
        hits = _filter_placeholder_hits(hits)
        return {
            "query": enhanced_question,
            "rewrites": rewrites,
            "hits": [
                {
                    "source": "local",
                    "page": hit.get("page"),
                    "snippet": str(hit.get("content") or "")[:220],
                    "content": str(hit.get("content") or ""),
                }
                for hit in hits[:effective_top_k]
            ],
        }

    bm25_index, chunks = rag_pipeline._get_bm25_index(kb_id)
    retrieved_nodes = _retrieve_with_rewrites(
        index,
        enhanced_question,
        bm25_index,
        chunks,
    )
    reranked_nodes = _apply_reranking(retrieved_nodes, enhanced_question)

    payload_hits: list[dict[str, Any]] = []
    for node in reranked_nodes[:effective_top_k]:
        content, meta = _extract_content_and_metadata(node)
        payload_hits.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "document_id": meta.get("document_id"),
                "snippet": content[:220],
                "content": content,
            }
        )

    return {
        "query": enhanced_question,
        "rewrites": rewrites,
        "hits": payload_hits,
    }
