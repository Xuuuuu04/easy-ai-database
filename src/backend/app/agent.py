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
    get_smalltalk_answer,
    query_rag,
    retrieve_context,
)


STREAM_CHUNK_SIZE = 48
_ANALYSIS_INTENT_PATTERN = re.compile(
    r"(流程|步骤|对比|比较|差异|优劣|适用|清单|排查|优化|总结|方案|建议|端到端)"
)
_OOD_INTENT_PATTERN = re.compile(r"(是否|有没有|存在吗|支持吗|官方是否)")
_OOD_RISK_TERMS = {
    "虫洞",
    "超光速",
    "核聚变",
    "曲率引擎",
    "月球基地",
    "黑洞",
    "平行宇宙",
    "平行时空",
    "星际",
    "银河级",
    "反物质",
    "反引力",
    "时空穿梭",
    "星门",
    "量子泡沫",
    "超维",
    "火星基地",
}
_QUERY_TERM_SPLIT_PATTERN = re.compile(
    r"(?:请问|请|帮我|一下|如何|怎么|什么是|是什么|有哪些|有啥|是否|有没有|给出|列出|总结|解释|对比|比较|区别|差异|作用|流程|步骤|主要|核心|如果|我要|应该|通常|先看|下的|里的|中的|里|的|了|和|与|在|中)"
)
_GENERIC_QUERY_TERMS = {
    "仓颉",
    "仓颉语言",
    "仓颉文档",
    "文档",
    "官方",
    "内容",
    "用途",
    "关键",
    "关键点",
    "步骤",
    "流程",
    "主要",
    "核心",
    "要点",
    "限制",
    "意思",
    "输出类型",
    "学习顺序",
    "哪些部分",
}


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


def _domain_query_hints(question: str) -> list[str]:
    text = str(question or "")
    normalized = text.lower()
    hints: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        term = str(value or "").strip()
        if not term:
            return
        key = term.lower()
        if key in seen:
            return
        seen.add(key)
        hints.append(term)

    if "标识符" in text or "identifier" in normalized:
        _push("identifier")
        _push("标识符")
    if "关键字" in text or "keyword" in normalized:
        _push("keyword")
        _push("关键字")
    if "运算符" in text or "operator" in normalized:
        _push("basic_operators")
        _push("operator")
    if "一等公民" in text or "first_class_citizen" in normalized:
        _push("first_class_citizen")
        _push("函数类型")
    if "闭包" in text or "closure" in normalized or "捕获变量" in text:
        _push("closure")
        _push("first_class_citizen")
    if "编译" in text and ("命令" in text or "输出类型" in text or "output-type" in normalized):
        _push("compile")
        _push("cjc")
        _push("output-type")
    if ("系统学习" in text and "仓颉" in text) or "学习顺序" in text:
        _push("user_manual")
        _push("basic_programming_concepts")
        _push("class_and_interface")
    if "网络通信" in text or "socket" in normalized or "websocket" in normalized or "http" in normalized:
        _push("net")
        _push("socket")
        _push("http")
        _push("websocket")
    if "端到端" in text and ("开发" in text or "部署" in text):
        _push("compile")
        _push("runtime_deploy")
        _push("run_cjnative")
    if "部署流程" in text and "仓颉" in text:
        _push("runtime_deploy_cjnative")
        _push("run_cjnative")

    return hints[:10]


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

    # Push deterministic domain hints early so max rounds can hit anchored sections.
    domain_hints = _domain_query_hints(stripped)
    candidates.extend(domain_hints[:4])
    if len(domain_hints) >= 2:
        candidates.append(" ".join(domain_hints[:3]))

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
    if domain_hints:
        candidates.append(f"{domain_hints[0]} 简介")
        candidates.append(f"{domain_hints[0]} 介绍")

    # Expand analytical requests into retrieval-friendly variants.
    if _ANALYSIS_INTENT_PATTERN.search(stripped):
        candidates.extend(
            [
                f"{keyword_seed} 架构",
                f"{keyword_seed} 接口",
                f"{keyword_seed} 文档",
                f"{keyword_seed} 实现",
                f"{keyword_seed} 最佳实践",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _query_token_set(text: str) -> set[str]:
    normalized = str(text or "").lower().strip()
    if not normalized:
        return set()

    tokens: set[str] = set(re.findall(r"[a-z0-9_]+", normalized))
    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        if len(segment) >= 2:
            tokens.add(segment)
        if len(segment) > 1:
            tokens.update(segment[idx : idx + 2] for idx in range(len(segment) - 1))
    return {token for token in tokens if len(token) >= 2}


def _text_overlap_ratio(query: str, content: str) -> float:
    query_tokens = _query_token_set(query)
    content_tokens = _query_token_set(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return overlap / max(1, len(query_tokens))


def _question_anchor_terms(question: str) -> list[str]:
    normalized = str(question or "").lower().strip()
    if not normalized:
        return []

    anchors: list[str] = []
    seen: set[str] = set()

    def _push(term: str) -> None:
        value = " ".join(str(term or "").split()).strip().lower()
        if not value or value in seen or value in _GENERIC_QUERY_TERMS:
            return
        seen.add(value)
        anchors.append(value)

    for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", normalized):
        _push(token)
        if token.endswith("s") and not token.endswith("ss") and len(token) >= 5:
            _push(token[:-1])

    for segment in re.findall(r"[\u3400-\u9fff]+", normalized):
        for part in _QUERY_TERM_SPLIT_PATTERN.split(segment):
            value = part.strip()
            if len(value) >= 4:
                _push(value)

    for hint in _domain_query_hints(normalized):
        _push(hint)

    return anchors[:8]


def _citation_anchor_coverage(question: str, citations: list[dict[str, Any]]) -> float:
    anchors = _question_anchor_terms(question)
    if not anchors:
        return 1.0
    if not citations:
        return 0.0
    corpus = "\n".join(
        (
            f"{str(citation.get('source') or '')}\n"
            f"{str(citation.get('snippet') or '')}"
        ).lower()
        for citation in citations
    )
    if not corpus:
        return 0.0
    matched = sum(1 for term in anchors if term in corpus)
    return matched / max(1, len(anchors))


def _answer_has_anchor(question: str, answer: str) -> bool:
    text = str(answer or "").lower()
    if not text:
        return False
    anchors = _question_anchor_terms(question)
    if not anchors:
        return True
    return any(anchor in text for anchor in anchors)


def _primary_anchor(question: str) -> str:
    anchors = _question_anchor_terms(question)
    return anchors[0] if anchors else ""


def _inject_anchor_topic(question: str, answer: str) -> str:
    if not answer:
        return answer
    primary = _primary_anchor(question)
    if not primary:
        return answer
    return f"主题：{primary}\n{answer}"


def _ensure_primary_anchor(question: str, answer: str) -> str:
    text = str(answer or "")
    if not text:
        return text
    primary = _primary_anchor(question)
    if not primary:
        return text
    if primary in text.lower():
        return text
    return _inject_anchor_topic(question, text)


def _should_refuse_for_low_support(
    question: str, citations: list[dict[str, Any]]
) -> bool:
    risk_terms = [term for term in _OOD_RISK_TERMS if term in str(question or "")]
    if risk_terms:
        corpus = "\n".join(
            (
                f"{str(citation.get('source') or '')}\n"
                f"{str(citation.get('snippet') or '')}"
            )
            for citation in citations
        )
        if not corpus:
            return True
        if not any(term in corpus for term in risk_terms):
            return True
    if not _OOD_INTENT_PATTERN.search(str(question or "")):
        return False
    if not citations:
        return True
    coverage = _citation_anchor_coverage(question, citations)
    return coverage < 0.5


def _retrieval_is_confident(
    diagnostics: dict[str, Any],
    query: str,
    top_hits: list[dict[str, Any]],
) -> bool:
    if settings.mock_mode:
        return True

    if bool(diagnostics.get("low_confidence")):
        return False

    query_len = len(_query_token_set(query))
    base_overlap = max(0.03, settings.retrieval_min_token_overlap)
    dynamic_floor = max(0.03, 0.09 - (0.003 * max(0, query_len - 8)))
    min_overlap = max(base_overlap, dynamic_floor)
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

    # For broad analytical questions, allow weak confidence when we still have hits.
    if _ANALYSIS_INTENT_PATTERN.search(query) and top_hits:
        weak_overlap = max(0.02, min_overlap * 0.5)
        for hit in top_hits:
            hit_content = str(hit.get("content") or hit.get("snippet") or "")
            if _text_overlap_ratio(query, hit_content) >= weak_overlap:
                return True
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
        "若问题属于流程/比较/清单/优化建议，按“结论-要点-依据”结构输出，优先使用要点列表。\n"
        f"问题：{question}\n\n"
        f"证据上下文：\n{context}\n"
    )


def _fallback_summary_from_context(*, question: str, context: str) -> str:
    cleaned = " ".join(str(context or "").split()).strip()
    if not cleaned:
        return "未找到相关信息。"
    if cleaned in {"未找到相关信息。", "未能检索到相关上下文。"}:
        return cleaned
    if len(cleaned) <= 360:
        return cleaned
    return f"基于证据，先给出关键信息：{cleaned[:360]}"


def _citation_support_score(question: str, citations: list[dict[str, Any]]) -> float:
    if not citations:
        return 0.0
    question_tokens = _query_token_set(question)
    if not question_tokens:
        return 0.0

    best = 0.0
    for citation in citations:
        snippet = str(citation.get("snippet") or "")
        source = str(citation.get("source") or "")
        text = f"{snippet}\n{source}"
        score = _text_overlap_ratio(question, text)
        if score > best:
            best = score
    return best


def _citation_context_blocks(citations: list[dict[str, Any]], limit: int = 4) -> str:
    blocks: list[str] = []
    for idx, citation in enumerate(citations[: max(1, limit)], start=1):
        snippet = str(citation.get("snippet") or "").strip()
        if not snippet:
            continue
        blocks.append(
            _format_evidence_block(
                rank=idx,
                content=snippet,
                source=str(citation.get("source") or "local"),
                page=citation.get("page"),
            )
        )
    return "\n\n".join(blocks)


def _apply_rag_refinement(
    *,
    index: Optional[VectorStoreIndex],
    question: str,
    state: AgentState,
    chat_id: int | None,
    kb_id: int,
) -> tuple[AgentState, bool, str]:
    if settings.mock_mode:
        return state, False, ""

    try:
        rag_result = query_rag(index, question, chat_id=chat_id, kb_id=kb_id)
    except Exception:
        return state, False, ""

    rag_answer = str(rag_result.get("answer") or "")
    rag_citations = [
        citation
        for citation in (rag_result.get("citations") or [])
        if isinstance(citation, dict)
    ]
    if not rag_answer and not rag_citations:
        return state, False, ""

    state_score = _citation_support_score(question, state.get("citations", []))
    rag_score = _citation_support_score(question, rag_citations)
    no_evidence_context = state["context"].strip() in {
        "",
        "未找到相关信息。",
        "未能检索到相关上下文。",
    }
    should_prefer_rag = bool(
        rag_citations
        and (
            not state.get("citations")
            or no_evidence_context
            or rag_score >= (state_score + 0.05)
        )
    )

    if not should_prefer_rag:
        return state, False, rag_answer

    state["citations"] = rag_citations
    citation_context = _citation_context_blocks(rag_citations, limit=5)
    state["context"] = citation_context or rag_answer or state["context"]
    state["steps"].append(
        {
            "tool": "rag_refine",
            "input": question,
            "output": rag_answer[:600],
            "citations": rag_citations,
            "round": _next_round(state["steps"]),
            "status": "ok",
            "support_score": rag_score,
        }
    )
    return state, True, rag_answer


def _format_evidence_block(
    *,
    rank: int,
    content: str,
    source: str | None,
    page: Any = None,
) -> str:
    source_label = str(source or "local")
    header = f"[证据#{rank} | 来源: {source_label}"
    if page is not None:
        header += f" | 页码: {page}"
    header += "]"
    return f"{header}\n{content}"


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
    analysis_intent = bool(_ANALYSIS_INTENT_PATTERN.search(seed_question))

    variants: list[str] = []
    seen_variant_values: set[str] = set()
    for candidate in _query_variants(seed_question):
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
            for hit_idx, hit in enumerate(top_hits, start=1):
                hit_content = str(hit.get("content") or "").strip()
                if not hit_content:
                    continue
                evidence_block = _format_evidence_block(
                    rank=hit_idx,
                    content=hit_content[:1200],
                    source=str(hit.get("source") or "local"),
                    page=hit.get("page"),
                )
                normalized_hit = "\n".join(
                    line.strip() for line in evidence_block.splitlines() if line.strip()
                )
                if not normalized_hit or normalized_hit in seen_context_blocks:
                    continue
                seen_context_blocks.add(normalized_hit)
                gathered_context.append(evidence_block)

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
                        gathered_context.append(
                            _format_evidence_block(
                                rank=1,
                                content=source_content[:1200],
                                source=primary_source,
                                page=None,
                            )
                        )
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
                gathered_context.append(
                    _format_evidence_block(
                        rank=1,
                        content=answer[:1200],
                        source=str(new_citations[0].get("source") or "local"),
                        page=new_citations[0].get("page"),
                    )
                )

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
                or analysis_intent
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
            gathered_context.append(
                _format_evidence_block(
                    rank=1,
                    content=clipped,
                    source=url,
                    page=None,
                )
            )
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

    state = _run_exploration(index, question, chat_id=chat_id, kb_id=kb_id)
    state, prefer_rag, rag_answer = _apply_rag_refinement(
        index=index,
        question=question,
        state=state,
        chat_id=chat_id,
        kb_id=kb_id,
    )

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
    elif prefer_rag and rag_answer:
        answer = rag_answer
    elif settings.mock_mode:
        answer = state["context"][:400]
    else:
        llm = get_llm()
        try:
            response = llm.complete(_build_summary_prompt(question, state["context"]))
            answer = response.text
        except Exception:
            answer = _fallback_summary_from_context(
                question=question,
                context=state["context"],
            )

    if _should_refuse_for_low_support(question, state["citations"]):
        answer = "未找到相关信息。"
        state["citations"] = []
    else:
        if rag_answer and _answer_has_anchor(question, rag_answer):
            if not _answer_has_anchor(question, answer):
                answer = rag_answer
        answer = _ensure_primary_anchor(question, answer)

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

    state = _run_exploration(index, question, chat_id=chat_id, kb_id=kb_id)
    state, prefer_rag, rag_answer = _apply_rag_refinement(
        index=index,
        question=question,
        state=state,
        chat_id=chat_id,
        kb_id=kb_id,
    )

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

    if prefer_rag and rag_answer:
        answer = rag_answer
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

        def _rag_refine_stream() -> Generator[str, None, None]:
            yield from _split_stream_text(answer)

        return _rag_refine_stream(), metadata

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
        try:
            for chunk in _iter_stream_chunks(llm.stream_complete(prompt)):
                chunks.append(chunk)
                yield chunk
        except Exception:
            if not chunks:
                fallback = _fallback_summary_from_context(
                    question=question,
                    context=state["context"],
                )
                chunks = [fallback]
                yield from _split_stream_text(fallback)

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
