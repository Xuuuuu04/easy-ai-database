from __future__ import annotations

from typing import Any, Generator, Optional, TypedDict

from llama_index.core import VectorStoreIndex

from .config import settings
from .indexer import get_llm
from .ingest import extract_text_from_url
from .rag import _build_context_with_history, get_smalltalk_answer, query_rag


STREAM_CHUNK_SIZE = 48


class AgentState(TypedDict):
    question: str
    context: str
    steps: list[dict[str, Any]]
    citations: list[dict[str, Any]]


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


def _query_variants(question: str) -> list[str]:
    text = question.strip()
    if not text:
        return []

    candidates = [text]
    stripped = text.rstrip("?？!！。；;，,")
    if stripped and stripped != text:
        candidates.append(stripped)

    if stripped.startswith("请问"):
        trimmed = stripped[len("请问") :].strip()
        if trimmed:
            candidates.append(trimmed)

    if stripped.startswith("什么是"):
        noun = stripped[len("什么是") :].strip()
        if noun:
            candidates.extend([noun, f"{noun} 简介", f"{noun} 介绍"])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _merge_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for citation in citations:
        source = str(citation.get("source") or "")
        page = str(citation.get("page") or "")
        snippet = str(citation.get("snippet") or "")
        key = f"{source}|{page}|{snippet[:120]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(citation)
    return merged


def _fetch_first_url(question: str) -> tuple[str, str] | None:
    for token in question.split():
        if token.startswith("http://") or token.startswith("https://"):
            text = extract_text_from_url(token)
            return token, text
    return None


def _build_summary_prompt(question: str, context: str) -> str:
    return (
        "你是检索增强助手。请基于以下证据回答问题，给出清晰结论与要点。\n"
        f"问题：{question}\n\n"
        f"证据上下文：\n{context}\n"
    )


def _run_exploration(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None,
    kb_id: int,
) -> AgentState:
    variants = _query_variants(question)
    rounds = max(1, settings.agent_max_rounds)
    state: AgentState = {
        "question": question,
        "context": "",
        "steps": [],
        "citations": [],
    }

    gathered_context: list[str] = []
    gathered_citations: list[dict[str, Any]] = []
    fallback_answer = "未能检索到相关上下文。"

    for variant in variants[:rounds]:
        result = query_rag(index, variant, chat_id=chat_id, kb_id=kb_id)
        answer = str(result.get("answer") or "")
        citations = result.get("citations") or []
        gathered_citations.extend(citations)
        state["steps"].append(
            {
                "tool": "search_kb",
                "input": variant,
                "output": answer,
                "citations": citations,
            }
        )

        if answer:
            fallback_answer = answer
        if citations and "未能检索到相关上下文" not in answer:
            gathered_context.append(answer)

        if len(gathered_context) >= 2:
            break

    fetched = _fetch_first_url(question)
    if fetched is not None:
        url, text = fetched
        clipped = text[:2000]
        gathered_context.append(clipped)
        state["steps"].append(
            {
                "tool": "fetch_url",
                "input": url,
                "output": clipped[:500],
                "citations": [{"source": url, "page": None, "snippet": clipped[:200]}],
            }
        )
        gathered_citations.append(
            {"source": url, "page": None, "snippet": clipped[:200]}
        )

    state["citations"] = _merge_citations(gathered_citations)
    if gathered_context:
        state["context"] = "\n\n".join(gathered_context)
    else:
        state["context"] = fallback_answer
    return state


def run_agent(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None = None,
    kb_id: int = 1,
) -> dict[str, Any]:
    smalltalk_answer = get_smalltalk_answer(question)
    if smalltalk_answer is not None:
        return {"answer": smalltalk_answer, "steps": [], "citations": []}

    enhanced_question = _build_context_with_history(question, chat_id, kb_id=kb_id)
    state = _run_exploration(index, enhanced_question, chat_id=chat_id, kb_id=kb_id)

    if settings.mock_mode:
        answer = state["context"][:400]
    else:
        llm = get_llm()
        response = llm.complete(_build_summary_prompt(question, state["context"]))
        answer = response.text

    state["steps"].append(
        {
            "tool": "summarize",
            "input": question,
            "output": answer,
            "citations": state["citations"],
        }
    )
    return {
        "answer": answer,
        "steps": state["steps"],
        "citations": state["citations"],
    }


def run_agent_stream_with_metadata(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None = None,
    kb_id: int = 1,
) -> tuple[Generator[str, None, None], dict[str, Any]]:
    smalltalk_answer = get_smalltalk_answer(question)
    if smalltalk_answer is not None:
        metadata: dict[str, Any] = {
            "answer": smalltalk_answer,
            "steps": [],
            "citations": [],
        }

        def _smalltalk_stream() -> Generator[str, None, None]:
            yield from _split_stream_text(smalltalk_answer)

        return _smalltalk_stream(), metadata

    enhanced_question = _build_context_with_history(question, chat_id, kb_id=kb_id)
    state = _run_exploration(index, enhanced_question, chat_id=chat_id, kb_id=kb_id)

    metadata: dict[str, Any] = {
        "answer": "",
        "steps": [],
        "citations": state["citations"],
    }

    if settings.mock_mode:
        answer = state["context"][:400]
        state["steps"].append(
            {
                "tool": "summarize",
                "input": question,
                "output": answer,
                "citations": state["citations"],
            }
        )
        metadata["answer"] = answer
        metadata["steps"] = state["steps"]

        def _mock_stream() -> Generator[str, None, None]:
            yield from _split_stream_text(answer)

        return _mock_stream(), metadata

    llm = get_llm()
    prompt = _build_summary_prompt(question, state["context"])

    def _stream() -> Generator[str, None, None]:
        chunks: list[str] = []
        for chunk in _iter_stream_chunks(llm.stream_complete(prompt)):
            chunks.append(chunk)
            yield chunk

        final_answer = "".join(chunks)
        state["steps"].append(
            {
                "tool": "summarize",
                "input": question,
                "output": final_answer,
                "citations": state["citations"],
            }
        )
        metadata["answer"] = final_answer
        metadata["steps"] = state["steps"]

    return _stream(), metadata


def run_agent_stream(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None = None,
    kb_id: int = 1,
) -> Generator[str, None, None]:
    stream, _ = run_agent_stream_with_metadata(
        index,
        question,
        chat_id=chat_id,
        kb_id=kb_id,
    )
    yield from stream
