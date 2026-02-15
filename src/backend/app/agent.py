from __future__ import annotations

import json
import re
from typing import Any, Generator, Optional, TypedDict
from urllib.parse import urlparse

from llama_index.core import VectorStoreIndex

from .config import settings
from .indexer import get_llm
from .ingest import (
    build_documents_from_file,
    build_documents_from_url,
    extract_text_from_url,
)
from .source_access import resolve_allowed_source_path
from .rag import (
    _build_context_with_history,
    get_smalltalk_answer,
    query_rag,
    retrieve_context,
)


STREAM_CHUNK_SIZE = 48


class AgentState(TypedDict):
    question: str
    context: str
    steps: list[dict[str, Any]]
    citations: list[dict[str, Any]]


def _derive_keywords(text: str, limit: int = 8) -> list[str]:
    stopwords = {
        "请",
        "请问",
        "帮我",
        "一下",
        "如何",
        "怎么",
        "什么",
        "以及",
        "并且",
        "the",
        "and",
        "for",
        "with",
        "about",
    }
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_\-]{2,}", text.lower())
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def _read_source_content(
    source: str,
    kb_id: int,
    max_chars: int = 2200,
) -> dict[str, Any]:
    normalized_source = str(source or "").strip()
    if not normalized_source or normalized_source == "local":
        return {
            "source": normalized_source or "local",
            "content": "",
            "kind": "local",
            "preview_type": "text",
            "error": "source_not_readable",
        }

    parsed = urlparse(normalized_source)
    if parsed.scheme in {"http", "https"}:
        docs = build_documents_from_url(normalized_source)
        content = "\n\n".join(doc.text for doc in docs if doc.text)
        return {
            "source": normalized_source,
            "content": content[:max_chars],
            "kind": "url",
            "preview_type": "text",
            "error": None,
        }

    try:
        source_path = resolve_allowed_source_path(normalized_source, kb_id)
    except PermissionError:
        return {
            "source": normalized_source,
            "content": "",
            "kind": "file",
            "preview_type": "text",
            "error": "source_outside_kb_root",
        }
    except FileNotFoundError:
        return {
            "source": normalized_source,
            "content": "",
            "kind": "file",
            "preview_type": "text",
            "error": "source_file_not_found",
        }

    docs = build_documents_from_file(source_path, str(source_path))
    content = "\n\n".join(doc.text for doc in docs if doc.text)
    ext = source_path.suffix.lower().lstrip(".")
    preview_type = "markdown" if ext in {"md", "markdown"} else "text"
    if ext in {"json", "xml", "yaml", "yml", "toml", "ini", "cfg"}:
        preview_type = "code"

    return {
        "source": normalized_source,
        "content": content[:max_chars],
        "kind": "file",
        "preview_type": preview_type,
        "error": None,
    }


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

    keyword_seed = re.sub(r"[?？!！。；;,，:：]", " ", stripped)
    keyword_seed = re.sub(
        r"^(请问|请|帮我|麻烦|能否|可以|告诉我|我想知道|请你|请帮我)",
        "",
        keyword_seed,
    ).strip()
    if keyword_seed and keyword_seed != stripped:
        candidates.append(keyword_seed)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _query_token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_\-]{2,}", text.lower())
        if token
    }


def _text_overlap_ratio(query: str, content: str) -> float:
    query_tokens = _query_token_set(query)
    content_tokens = _query_token_set(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return overlap / max(1, len(query_tokens))


def _retrieval_is_confident(
    diagnostics: dict[str, Any],
    query: str,
    top_hits: list[dict[str, Any]],
) -> bool:
    if settings.mock_mode:
        return True

    if bool(diagnostics.get("low_confidence")):
        return False

    min_overlap = max(0.08, settings.retrieval_min_token_overlap)
    avg_overlap = diagnostics.get("avg_token_overlap")
    try:
        if avg_overlap is not None and float(avg_overlap) >= min_overlap:
            return True
    except Exception:
        pass

    for hit in top_hits:
        hit_content = str(hit.get("content") or hit.get("snippet") or "")
        if _text_overlap_ratio(query, hit_content) >= min_overlap:
            return True
    return False


def _query_is_novel(query: str, existing: set[str], threshold: float = 0.72) -> bool:
    normalized_query = " ".join(query.split())
    if not normalized_query or normalized_query in existing:
        return False

    query_tokens = _query_token_set(normalized_query)
    for item in existing:
        if item == normalized_query:
            return False
        item_tokens = _query_token_set(item)
        if not query_tokens or not item_tokens:
            continue
        overlap = len(query_tokens.intersection(item_tokens))
        union = len(query_tokens.union(item_tokens))
        if union <= 0:
            continue
        if (overlap / union) >= threshold:
            return False
    return True


def _derive_followup_queries_from_hits(
    question: str,
    top_hits: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    candidates: list[str] = []
    for hit in top_hits[:3]:
        snippet = str(hit.get("snippet") or "")
        if not snippet:
            continue
        keywords = _derive_keywords(snippet, limit=4)
        if not keywords:
            continue
        candidates.append(" ".join(keywords))
        candidates.append(f"{question} {' '.join(keywords[:2])}".strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def _next_round(steps: list[dict[str, Any]]) -> int:
    rounds = [int(step.get("round") or 0) for step in steps if step.get("round")]
    return (max(rounds) + 1) if rounds else 1


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _citation_key(citation: dict[str, Any]) -> str:
    source = str(citation.get("source") or "")
    page = str(citation.get("page") or "")
    snippet = str(citation.get("snippet") or "")
    return f"{source}|{page}|{snippet[:120]}"


def _extract_urls(question: str) -> list[str]:
    urls = re.findall(r"https?://[^\s\]\)\>\"']+", question)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        cleaned = url.rstrip(".,;:!?)]>")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _fetch_first_url(question: str) -> dict[str, Any] | None:
    for url in _extract_urls(question):
        try:
            text = extract_text_from_url(url)
            return {"url": url, "text": text, "error": None}
        except Exception as exc:
            return {"url": url, "text": "", "error": str(exc)}
    return None


def _plan_followup_queries(
    question: str,
    steps: list[dict[str, Any]],
    exclude: set[str],
    max_queries: int,
) -> list[str]:
    if max_queries <= 0 or settings.mock_mode:
        return []

    llm = get_llm()
    compact_steps = [
        {
            "query": step.get("input"),
            "answer": str(step.get("output") or "")[:260],
            "citations": len(step.get("citations") or []),
        }
        for step in steps
        if step.get("tool") == "search_kb"
    ]

    prompt = (
        "You are a retrieval strategist. Generate follow-up search queries to improve evidence coverage.\n"
        "Return JSON only with key 'queries' as a list of concise search queries.\n"
        "Avoid repeating failed/duplicate queries and keep language consistent with user question.\n"
        f"Question: {question}\n"
        f"Previous steps: {json.dumps(compact_steps, ensure_ascii=False)}\n"
        f"Max queries: {max_queries}"
    )

    try:
        raw = llm.complete(prompt).text
    except Exception:
        return []

    parsed = _extract_json_object(raw)
    if parsed is None:
        return []

    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        return []

    planned: list[str] = []
    for item in raw_queries:
        query = " ".join(str(item or "").split())
        if not query or query in exclude or query in planned:
            continue
        planned.append(query)
        if len(planned) >= max_queries:
            break
    return planned


def _build_summary_prompt(question: str, context: str) -> str:
    return (
        "你是严格的检索增强助手。\n"
        "仅允许使用给定证据，不得使用外部知识。\n"
        "如果证据不足，请只输出：未找到相关信息。\n"
        "必须保留证据中的关键术语、实体名、编号、代码与专有 token 原文，不可改写。\n"
        "若问题询问定义或结论，请优先引用证据中的原句并保留核心词。\n"
        "回答必须包含结论，并在结尾给出引用来源。\n"
        f"问题：{question}\n\n"
        f"证据上下文：\n{context}\n"
    )


def _run_exploration(
    index: Optional[VectorStoreIndex],
    question: str,
    chat_id: int | None,
    kb_id: int,
) -> AgentState:
    seed_question = question
    marker = "当前问题:"
    if marker in question:
        seed_question = question.split(marker)[-1].strip() or question

    variants: list[str] = []
    seen_variant_values: set[str] = set()
    for candidate in _query_variants(question) + _query_variants(seed_question):
        normalized_candidate = " ".join(candidate.split())
        if not normalized_candidate or normalized_candidate in seen_variant_values:
            continue
        seen_variant_values.add(normalized_candidate)
        variants.append(normalized_candidate)
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

    pending_queries: list[str] = []
    known_queries: set[str] = set()
    for variant in variants:
        normalized_variant = " ".join(variant.split())
        if _query_is_novel(normalized_variant, known_queries):
            pending_queries.append(normalized_variant)
            known_queries.add(normalized_variant)

    seen_queries: set[str] = set()
    seen_context_blocks: set[str] = set()
    seen_citation_keys: set[str] = set()
    seen_read_sources: set[str] = set()
    executed_rounds = 0

    while pending_queries and executed_rounds < rounds:
        variant = pending_queries.pop(0)
        if variant in seen_queries:
            continue
        seen_queries.add(variant)
        executed_rounds += 1

        retrieval_payload: dict[str, Any] = {}
        try:
            retrieval_payload = retrieve_context(
                index,
                variant,
                kb_id=kb_id,
                chat_id=chat_id,
                top_k=min(4, settings.max_context_chunks),
            )
        except Exception:
            retrieval_payload = {}

        rewrites = [
            str(item)
            for item in retrieval_payload.get("rewrites", [])
            if str(item).strip()
        ]
        top_hits = [
            {
                "source": hit.get("source"),
                "page": hit.get("page"),
                "snippet": str(hit.get("snippet") or "")[:220],
                "content": str(hit.get("content") or ""),
            }
            for hit in retrieval_payload.get("hits", [])[:3]
            if isinstance(hit, dict)
        ]
        retrieval_diagnostics = retrieval_payload.get("diagnostics") or {}
        retrieval_confident = _retrieval_is_confident(
            retrieval_diagnostics, variant, top_hits
        )
        keyword_seed = " ".join([variant] + rewrites[:2])
        keywords = _derive_keywords(keyword_seed)

        if retrieval_confident:
            for hit in top_hits:
                hit_content = str(hit.get("content") or "").strip()
                if not hit_content:
                    continue
                normalized_hit = "\n".join(
                    line.strip() for line in hit_content.splitlines() if line.strip()
                )
                if not normalized_hit or normalized_hit in seen_context_blocks:
                    continue
                seen_context_blocks.add(normalized_hit)
                gathered_context.append(hit_content[:1200])

        answer = ""
        new_citations: list[dict[str, Any]] = []

        if retrieval_confident and top_hits and not settings.mock_mode:
            answer = str(top_hits[0].get("content") or top_hits[0].get("snippet") or "")
            derived_citations: list[dict[str, Any]] = []
            for hit in top_hits:
                source = str(hit.get("source") or "")
                snippet = str(hit.get("snippet") or hit.get("content") or "")[:200]
                if not source or not snippet:
                    continue
                derived_citations.append(
                    {
                        "source": source,
                        "page": hit.get("page"),
                        "snippet": snippet,
                    }
                )

            for citation in derived_citations:
                key = _citation_key(citation)
                if key in seen_citation_keys:
                    continue
                seen_citation_keys.add(key)
                new_citations.append(citation)
        elif (
            index is None
            or settings.mock_mode
            or settings.agent_enable_llm_search_fallback
        ):
            try:
                result = query_rag(index, variant, chat_id=chat_id, kb_id=kb_id)
            except Exception as exc:
                state["steps"].append(
                    {
                        "tool": "search_kb",
                        "input": variant,
                        "output": "",
                        "citations": [],
                        "round": executed_rounds,
                        "rewrites": rewrites,
                        "keywords": keywords,
                        "top_hits": top_hits,
                        "retrieval_diagnostics": retrieval_diagnostics,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue

            answer = str(result.get("answer") or "")
            citations = result.get("citations") or []
            for citation in citations:
                key = _citation_key(citation)
                if key in seen_citation_keys:
                    continue
                seen_citation_keys.add(key)
                new_citations.append(citation)
        else:
            answer = "未找到相关信息。"

        gathered_citations.extend(new_citations)

        state["steps"].append(
            {
                "tool": "search_kb",
                "input": variant,
                "output": answer,
                "citations": new_citations,
                "round": executed_rounds,
                "rewrites": rewrites,
                "keywords": keywords,
                "top_hits": top_hits,
                "retrieval_diagnostics": retrieval_diagnostics,
                "retrieval_confident": retrieval_confident,
                "status": "ok",
            }
        )

        if new_citations:
            primary_source = str(new_citations[0].get("source") or "")
            if (
                primary_source
                and primary_source != "local"
                and primary_source not in seen_read_sources
            ):
                seen_read_sources.add(primary_source)
                source_payload = _read_source_content(primary_source, kb_id=kb_id)
                source_content = str(source_payload.get("content") or "")
                source_error = source_payload.get("error")
                preview_snippet = source_content[:500] if source_content else ""
                read_step: dict[str, Any] = {
                    "tool": "read_source",
                    "input": primary_source,
                    "output": preview_snippet,
                    "citations": [],
                    "round": executed_rounds,
                    "status": "error" if source_error else "ok",
                    "preview_type": source_payload.get("preview_type") or "text",
                    "source_kind": source_payload.get("kind") or "file",
                }
                if source_error:
                    read_step["error"] = str(source_error)
                else:
                    read_citation = {
                        "source": primary_source,
                        "page": None,
                        "snippet": source_content[:200],
                    }
                    read_step["citations"] = [read_citation]
                    gathered_citations.append(read_citation)
                    if source_content:
                        gathered_context.append(source_content[:1200])
                state["steps"].append(read_step)

        if retrieval_confident and answer and "未能检索到相关上下文" not in answer:
            fallback_answer = answer
        if (
            retrieval_confident
            and new_citations
            and "未能检索到相关上下文" not in answer
        ):
            normalized_answer = "\n".join(
                line.strip() for line in answer.splitlines() if line.strip()
            )
            if normalized_answer and normalized_answer not in seen_context_blocks:
                seen_context_blocks.add(normalized_answer)
                gathered_context.append(answer)

        if len(gathered_context) >= 2:
            break

        remaining_rounds = rounds - executed_rounds
        if remaining_rounds > 0 and len(gathered_context) < 2:
            excluded_queries = seen_queries.union(set(pending_queries))
            followups = _derive_followup_queries_from_hits(
                question=seed_question,
                top_hits=top_hits,
                limit=max(1, remaining_rounds),
            )
            allow_llm_planner = (
                index is None
                or settings.mock_mode
                or settings.agent_enable_llm_search_fallback
                or retrieval_confident
            )
            if allow_llm_planner and (
                (not followups) or (not settings.agent_prefer_deterministic_followups)
            ):
                followups.extend(
                    _plan_followup_queries(
                        question=seed_question,
                        steps=state["steps"],
                        exclude=excluded_queries,
                        max_queries=max(1, remaining_rounds),
                    )
                )
            elif (
                allow_llm_planner
                and settings.agent_prefer_deterministic_followups
                and len(followups) < max(1, remaining_rounds)
            ):
                followups.extend(
                    _plan_followup_queries(
                        question=seed_question,
                        steps=state["steps"],
                        exclude=excluded_queries,
                        max_queries=max(1, remaining_rounds) - len(followups),
                    )
                )
            for followup in followups:
                normalized_followup = " ".join(followup.split())
                if _query_is_novel(normalized_followup, excluded_queries):
                    pending_queries.append(normalized_followup)
                    excluded_queries.add(normalized_followup)

            if allow_llm_planner and not pending_queries:
                planned_followups = _plan_followup_queries(
                    question=seed_question,
                    steps=state["steps"],
                    exclude=excluded_queries,
                    max_queries=max(1, remaining_rounds),
                )
                for followup in planned_followups:
                    normalized_followup = " ".join(followup.split())
                    if _query_is_novel(normalized_followup, excluded_queries):
                        pending_queries.append(normalized_followup)
                        excluded_queries.add(normalized_followup)

    fetched = _fetch_first_url(question)
    if fetched is not None:
        url = str(fetched.get("url") or "")
        error = fetched.get("error")
        if error:
            state["steps"].append(
                {
                    "tool": "fetch_url",
                    "input": url,
                    "output": "",
                    "citations": [],
                    "round": executed_rounds,
                    "status": "error",
                    "error": str(error),
                }
            )
        else:
            text = str(fetched.get("text") or "")
            clipped = text[:2000]
            gathered_context.append(clipped)
            citation = {"source": url, "page": None, "snippet": clipped[:200]}
            state["steps"].append(
                {
                    "tool": "fetch_url",
                    "input": url,
                    "output": clipped[:500],
                    "citations": [citation],
                    "round": executed_rounds,
                    "status": "ok",
                }
            )
            gathered_citations.append(citation)

    state["citations"] = _merge_citations(gathered_citations)
    if gathered_context:
        state["context"] = "\n\n".join(gathered_context)
    elif state["citations"]:
        state["context"] = fallback_answer
    else:
        state["context"] = "未找到相关信息。"
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

    no_evidence_context = state["context"].strip() in {
        "",
        "未找到相关信息。",
        "未能检索到相关上下文。",
    }

    if (
        settings.agent_skip_llm_when_no_evidence
        and not state["citations"]
        and no_evidence_context
    ):
        answer = "未找到相关信息。"
    elif settings.mock_mode:
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
            "round": _next_round(state["steps"]),
            "status": "ok",
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

    no_evidence_context = state["context"].strip() in {
        "",
        "未找到相关信息。",
        "未能检索到相关上下文。",
    }

    metadata: dict[str, Any] = {
        "answer": "",
        "steps": list(state["steps"]),
        "citations": state["citations"],
        "research_mode": True,
    }

    if (
        settings.agent_skip_llm_when_no_evidence
        and not state["citations"]
        and no_evidence_context
    ):
        answer = "未找到相关信息。"
        state["steps"].append(
            {
                "tool": "summarize",
                "input": question,
                "output": answer,
                "citations": state["citations"],
                "round": _next_round(state["steps"]),
                "status": "ok",
            }
        )
        metadata["answer"] = answer
        metadata["steps"] = state["steps"]

        def _no_evidence_stream() -> Generator[str, None, None]:
            yield from _split_stream_text(answer)

        return _no_evidence_stream(), metadata

    if settings.mock_mode:
        answer = state["context"][:400]
        state["steps"].append(
            {
                "tool": "summarize",
                "input": question,
                "output": answer,
                "citations": state["citations"],
                "round": _next_round(state["steps"]),
                "status": "ok",
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
                "round": _next_round(state["steps"]),
                "status": "ok",
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
