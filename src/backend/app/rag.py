from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Generator, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .cache import semantic_cache
from .config import settings
from .db import get_chat_history, get_chat_summary, list_chunks, search_chunks
from .hybrid_search import (
    bm25_search,
    build_bm25_index,
    get_all_chunks_from_db,
    hybrid_retrieve,
)
from .indexer import get_llm
from .reranker import rerank_documents


STREAM_CHUNK_SIZE = 48

_SMALL_TALK_TEMPLATES = {
    "greeting": "你好！我是 easy-ai-database 助手。你可以直接问我知识库里的问题，我会基于本地资料回答并给出来源。",
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
_MULTI_QUERY_SPLIT_PATTERN = re.compile(
    r"(?:以及|并且|同时|还有|,|，|、|;|；|/|\band\b|\bor\b)",
    re.IGNORECASE,
)
_OOD_CUE_PATTERN = re.compile(r"(?:是否|有没有|存在吗|支持吗|官方是否)")
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
_LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_CJK_SEGMENT_PATTERN = re.compile(r"[\u3400-\u9fff]+")
_QUERY_TERM_SPLIT_PATTERN = re.compile(
    r"(?:请问|请|帮我|一下|如何|怎么|什么是|是什么|有哪些|有啥|是否|有没有|给出|总结|解释|对比|比较|区别|差异|作用|流程|步骤|主要|核心|如果|我要|应该|通常|先看|下的|里的|中的|里|的|了|和|与|在|中)"
)
_QUERY_TERM_STOPWORDS = {
    "请",
    "请问",
    "帮我",
    "一下",
    "如何",
    "怎么",
    "什么",
    "是否",
    "有没有",
    "给出",
    "总结",
    "解释",
    "对比",
    "比较",
    "区别",
    "差异",
    "作用",
    "流程",
    "步骤",
    "主要",
    "核心",
    "定义",
    "要点",
    "限制",
    "意思",
    "输出类型",
    "学习顺序",
    "哪些部分",
}
_GENERIC_FOCUS_TERMS = {
    "cangjie",
    "仓颉",
    "仓颉语言",
    "仓颉文档",
    "官方",
    "文档",
    "模块",
    "章节",
    "关键",
    "关键点",
    "信息",
    "内容",
    "用途",
    "要点",
    "限制",
    "意思",
    "输出类型",
    "学习顺序",
    "哪些部分",
    "作用",
    "区别",
    "差异",
    "步骤",
    "流程",
    "主要",
    "核心",
    "什么",
    "哪些",
    "如何",
    "怎么",
    "附录",
    "列出",
    "什么信息",
    "相关章节",
    "包含哪些关键点",
    "端到端实践清单",
    "系统学习仓颉",
    "学习顺序建议",
    "哪些部分",
    "排查网络通信问题",
}
_LOW_SIGNAL_SOURCE_PATTERNS = (
    "/claude.md",
    "/_structure.json",
    "/_failed_urls.json",
)


def _extract_focus_question(question: str) -> str:
    if not settings.retrieval_focus_on_current_question:
        return question
    marker = "当前问题:"
    if marker in question:
        focused = question.split(marker)[-1].strip()
        if focused:
            return focused
    return question


def _relevance_tokens(text: str) -> set[str]:
    normalized = str(text or "").lower().strip()
    if not normalized:
        return set()

    tokens: set[str] = set()
    for token in _LATIN_TOKEN_PATTERN.findall(normalized):
        tokens.add(token)
        for part in re.split(r"[_-]+", token):
            if len(part) >= 2:
                tokens.add(part)

    for segment in _CJK_SEGMENT_PATTERN.findall(normalized):
        if len(segment) >= 2:
            tokens.add(segment)
        if len(segment) > 1:
            tokens.update(segment[idx : idx + 2] for idx in range(len(segment) - 1))

    return {token for token in tokens if len(token) >= 2}


def _token_overlap_ratio(query: str, content: str) -> float:
    query_tokens = _relevance_tokens(query)
    content_tokens = _relevance_tokens(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return overlap / max(1, len(query_tokens))


def _read_rerank_score(meta: dict[str, Any]) -> float | None:
    score = meta.get("rerank_score")
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def _filter_nodes_by_quality(nodes: list[Any], query: str) -> list[Any]:
    if not settings.retrieval_enable_quality_gate:
        return nodes

    min_overlap = max(0.0, settings.retrieval_min_token_overlap)
    min_rerank_score = settings.retrieval_min_rerank_score
    focus_terms = _query_focus_terms(query)
    filtered: list[Any] = []

    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        content_text = str(content or "").lower()
        source_text = str(meta.get("source") or "").lower()
        overlap = _token_overlap_ratio(query, content)
        source_overlap = _token_overlap_ratio(query, source_text)
        rerank_score = _read_rerank_score(meta)
        focus_hit = any(term in source_text or term in content_text for term in focus_terms)

        if focus_terms:
            keep = focus_hit or overlap >= min_overlap or source_overlap >= max(
                0.08, min_overlap * 0.8
            )
        else:
            keep = overlap >= min_overlap or source_overlap >= max(
                0.08, min_overlap * 0.8
            )
        if not keep and rerank_score is not None and overlap > 0:
            keep = rerank_score >= min_rerank_score

        if keep:
            filtered.append(node)

    return filtered


def _average_hit_overlap(query: str, payload_hits: list[dict[str, Any]]) -> float:
    if not payload_hits:
        return 0.0
    overlaps: list[float] = []
    for hit in payload_hits:
        text = str(hit.get("content") or hit.get("snippet") or "")
        overlaps.append(_token_overlap_ratio(query, text))
    return sum(overlaps) / max(1, len(overlaps))


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _node_source_key(meta: dict[str, Any]) -> str:
    source = str(meta.get("source") or "").strip()
    if source:
        return source
    document_id = meta.get("document_id")
    if document_id is not None:
        return f"doc:{document_id}"
    return "local"


def _is_low_signal_source(meta: dict[str, Any]) -> bool:
    source = str(meta.get("source") or "").lower()
    if not source:
        return False
    return any(pattern in source for pattern in _LOW_SIGNAL_SOURCE_PATTERNS)


def _filter_low_signal_nodes(nodes: list[Any], query: str = "") -> list[Any]:
    if not nodes:
        return []
    focus_terms = _query_focus_terms(query)
    preferred = []
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        if not _is_low_signal_source(meta):
            preferred.append(node)
            continue
        if focus_terms:
            source_text = str(meta.get("source") or "").lower()
            content_text = str(content or "").lower()
            if any(term in source_text or term in content_text for term in focus_terms):
                preferred.append(node)
            continue
    return preferred if preferred else nodes


def _node_relevance_score(query: str, content: str, meta: dict[str, Any]) -> float:
    overlap = _token_overlap_ratio(query, content)
    source_text = str(meta.get("source") or "").lower()
    source_overlap = _token_overlap_ratio(query, source_text)
    rerank_score = _read_rerank_score(meta)
    rerank_component = 0.0
    if rerank_score is not None:
        rerank_component = max(0.0, min(1.0, rerank_score))
    normalized_query = _normalize_query_text(query)
    normalized_content = _normalize_query_text(content)
    exact_bonus = 0.15 if normalized_query and normalized_query in normalized_content else 0.0
    source_anchor_bonus = 0.0
    anchors = _query_anchor_terms(query)
    if anchors and source_text:
        anchor_hits = sum(1 for anchor in anchors if anchor in source_text)
        source_anchor_bonus = min(0.35, 0.35 * (anchor_hits / max(1, len(anchors))))
    focus_match = _node_focus_match_score(query, content, source_text)
    return (
        (overlap * 0.4)
        + (source_overlap * 0.25)
        + (rerank_component * 0.2)
        + (focus_match * 0.55)
        + exact_bonus
        + source_anchor_bonus
    )


def _query_focus_terms(query: str) -> list[str]:
    anchors = _query_anchor_terms(query)
    hints = _domain_query_hints(query)
    if not anchors:
        return hints[:8]
    focus_terms: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if hint and hint not in seen:
            seen.add(hint)
            focus_terms.append(hint)
    for anchor in anchors:
        term = anchor.strip().lower()
        if not term or term in seen:
            continue
        if term in _GENERIC_FOCUS_TERMS:
            continue
        if len(term) < 3 and _CJK_SEGMENT_PATTERN.fullmatch(term) is None:
            continue
        if _CJK_SEGMENT_PATTERN.fullmatch(term) is not None and len(term) > 14:
            continue
        seen.add(term)
        focus_terms.append(term)
    return focus_terms[:8]


def _domain_query_hints(query: str) -> list[str]:
    text = str(query or "")
    normalized = text.lower()
    hints: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        term = str(value or "").strip().lower()
        if not term or term in seen:
            return
        seen.add(term)
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
        _push("运算符")
    if "一等公民" in text or "first_class_citizen" in normalized:
        _push("first_class_citizen")
        _push("函数类型")
        _push("一等公民")
    if "闭包" in text or "closure" in normalized or "捕获变量" in text:
        _push("closure")
        _push("first_class_citizen")
        _push("闭包")
    if "编译" in text and ("命令" in text or "输出类型" in text or "output-type" in normalized):
        _push("compile")
        _push("cjc")
        _push("output-type")
    if ("系统学习" in text and "仓颉" in text) or "学习顺序" in text:
        _push("user_manual")
        _push("basic_programming_concepts")
        _push("function")
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
        _push("deploy_and_run")
    if "部署流程" in text and "仓颉" in text:
        _push("runtime_deploy_cjnative")
        _push("run_cjnative")
        _push("deploy_and_run")

    return hints[:10]


def _node_focus_match_score(query: str, content: str, source_text: str) -> float:
    terms = _query_focus_terms(query)
    if not terms:
        return 0.0
    content_text = str(content or "").lower()
    source_text = str(source_text or "").lower()
    matched = 0.0
    for term in terms:
        if term in source_text:
            matched += 2.2
        elif term in content_text:
            matched += 0.4
    return matched / max(1, len(terms))


def _query_anchor_terms(query: str) -> list[str]:
    normalized = str(query or "").lower().strip()
    if not normalized:
        return []

    terms: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        term = " ".join(value.split()).strip().lower()
        if not term or len(term) < 2:
            return
        if term in _QUERY_TERM_STOPWORDS:
            return
        if _CJK_SEGMENT_PATTERN.fullmatch(term) is not None and len(term) > 8:
            if any(marker in term for marker in ("如果", "应该", "哪些", "怎么", "如何", "是否", "什么")):
                return
        if term in seen:
            return
        seen.add(term)
        terms.append(term)
        for prefix in ("这个", "该", "此"):
            if term.startswith(prefix) and len(term) > len(prefix) + 1:
                trimmed = term[len(prefix) :].strip()
                if trimmed and trimmed not in seen:
                    seen.add(trimmed)
                    terms.append(trimmed)

    for latin in _LATIN_TOKEN_PATTERN.findall(normalized):
        if len(latin) >= 3:
            _push(latin)
            if latin.endswith("s") and not latin.endswith("ss") and len(latin) >= 5:
                _push(latin[:-1])

    for segment in _CJK_SEGMENT_PATTERN.findall(normalized):
        for part in _QUERY_TERM_SPLIT_PATTERN.split(segment):
            part = part.strip()
            if len(part) >= 2:
                _push(part)

    return terms[:10]


def _context_anchor_coverage(query: str, nodes: list[Any]) -> float:
    anchors = _query_anchor_terms(query)
    if not anchors:
        return 1.0

    corpus_token_sets: list[set[str]] = []
    corpus_parts: list[str] = []
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        source = str(meta.get("source") or "")
        if content:
            corpus_token_sets.append(_relevance_tokens(content))
            corpus_parts.append(content.lower())
        if source:
            corpus_token_sets.append(_relevance_tokens(source))
            corpus_parts.append(source.lower())
    if not corpus_token_sets:
        return 0.0
    corpus_text = "\n".join(corpus_parts)

    matches = 0
    for anchor in anchors:
        if len(anchor) >= 4 and _CJK_SEGMENT_PATTERN.fullmatch(anchor) is not None:
            if anchor in corpus_text:
                matches += 1
            continue
        anchor_tokens = _relevance_tokens(anchor)
        if not anchor_tokens:
            continue
        for token_set in corpus_token_sets:
            overlap = len(anchor_tokens.intersection(token_set)) / max(
                1, len(anchor_tokens)
            )
            if overlap >= 0.5:
                matches += 1
                break
    return matches / max(1, len(anchors))


def _context_focus_coverage(query: str, nodes: list[Any]) -> float:
    focus_terms = _query_focus_terms(query)
    if not focus_terms:
        return 1.0

    corpus_token_sets: list[set[str]] = []
    corpus_parts: list[str] = []
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        source = str(meta.get("source") or "")
        if content:
            corpus_parts.append(content.lower())
            corpus_token_sets.append(_relevance_tokens(content))
        if source:
            corpus_parts.append(source.lower())
            corpus_token_sets.append(_relevance_tokens(source))
    if not corpus_parts:
        return 0.0

    corpus_text = "\n".join(corpus_parts)
    matches = 0
    for term in focus_terms:
        if term in corpus_text:
            matches += 1
            continue
        if _CJK_SEGMENT_PATTERN.fullmatch(term) is not None and len(term) >= 3:
            if term in corpus_text:
                matches += 1
            continue

        term_tokens = _relevance_tokens(term)
        if not term_tokens:
            continue
        for token_set in corpus_token_sets:
            overlap = len(term_tokens.intersection(token_set)) / max(1, len(term_tokens))
            if overlap >= 0.6:
                matches += 1
                break
    return matches / max(1, len(focus_terms))


def _query_strict_terms(query: str) -> list[str]:
    strict_terms: list[str] = []
    seen: set[str] = set()

    for term in _query_focus_terms(query):
        candidate = term.strip().lower()
        if not candidate or candidate in _GENERIC_FOCUS_TERMS:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        strict_terms.append(candidate)

    for latin in _LATIN_TOKEN_PATTERN.findall(str(query or "").lower()):
        if len(latin) < 3:
            continue
        if latin in seen:
            continue
        seen.add(latin)
        strict_terms.append(latin)
        if (
            latin.endswith("s")
            and not latin.endswith("ss")
            and len(latin) >= 5
            and latin[:-1] not in seen
        ):
            seen.add(latin[:-1])
            strict_terms.append(latin[:-1])

    for segment in _CJK_SEGMENT_PATTERN.findall(str(query or "").lower()):
        for part in _QUERY_TERM_SPLIT_PATTERN.split(segment):
            candidate = part.strip()
            if len(candidate) < 4:
                continue
            if candidate in _GENERIC_FOCUS_TERMS or candidate in seen:
                continue
            seen.add(candidate)
            strict_terms.append(candidate)

    return strict_terms[:8]


def _strict_term_coverage(query: str, nodes: list[Any]) -> float:
    strict_terms = _query_strict_terms(query)
    if not strict_terms:
        return 1.0

    corpus_parts: list[str] = []
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        source = str(meta.get("source") or "")
        if content:
            corpus_parts.append(content.lower())
        if source:
            corpus_parts.append(source.lower())
    if not corpus_parts:
        return 0.0

    corpus_text = "\n".join(corpus_parts)
    matched = sum(1 for term in strict_terms if term in corpus_text)
    return matched / max(1, len(strict_terms))


def _query_ood_risk_terms(query: str) -> list[str]:
    text = str(query or "")
    return [term for term in _OOD_RISK_TERMS if term in text]


def _nodes_match_focus_terms(nodes: list[Any], query: str) -> bool:
    terms = _query_focus_terms(query)
    if not terms:
        return True
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        source_text = str(meta.get("source") or "").lower()
        content_text = str(content or "").lower()
        if any(term in source_text or term in content_text for term in terms):
            return True
    return False


def _should_abstain_for_low_coverage(query: str, nodes: list[Any]) -> bool:
    if not nodes:
        return True
    risk_terms = _query_ood_risk_terms(query)
    if risk_terms:
        corpus = []
        for node in nodes:
            content, meta = _extract_content_and_metadata(node)
            corpus.append(str(content or ""))
            corpus.append(str(meta.get("source") or ""))
        corpus_text = "\n".join(corpus)
        if not any(term in corpus_text for term in risk_terms):
            return True
    strict_coverage = _strict_term_coverage(query, nodes)
    if strict_coverage <= 0:
        return True
    if _OOD_CUE_PATTERN.search(str(query or "")) and strict_coverage < 0.67:
        return True
    focus_terms = _query_focus_terms(query)
    if focus_terms:
        focus_coverage = _context_focus_coverage(query, nodes)
        return focus_coverage < settings.retrieval_min_focus_coverage
    anchor_coverage = _context_anchor_coverage(query, nodes)
    return anchor_coverage < settings.retrieval_min_anchor_coverage


def _token_jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _relevance_tokens(left)
    right_tokens = _relevance_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    if union <= 0:
        return 0.0
    return overlap / union


def _context_limit_for_question(question: str) -> int:
    base_limit = max(1, settings.max_context_chunks)
    query_tokens = _relevance_tokens(question)
    if len(query_tokens) >= 18:
        return min(base_limit + 2, 12)
    return base_limit


def _candidate_limit_for_question(question: str, corpus_size: int = 0) -> int:
    base = max(settings.rerank_top_k, _context_limit_for_question(question))
    multiplier = max(2, settings.retrieval_candidate_multiplier)
    token_bonus = 1 if len(_relevance_tokens(question)) >= 12 else 0
    size_bonus = 0
    if corpus_size >= 100:
        size_bonus = min(3, int(math.log2(max(2, corpus_size // 100))))
    raw_limit = max(base * (multiplier + token_bonus + size_bonus), base * 2)
    return min(max(raw_limit, base * 2), max(base * 2, settings.retrieval_max_candidates))


def _filter_nodes_by_focus_terms(nodes: list[Any], query: str) -> list[Any]:
    focus_terms = _query_focus_terms(query)
    if not nodes or not focus_terms:
        return nodes

    matched_nodes: list[Any] = []
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        source = str(meta.get("source") or "").lower()
        content_text = str(content or "").lower()
        if any(term in source or term in content_text for term in focus_terms):
            matched_nodes.append(node)

    return matched_nodes if matched_nodes else nodes


def _select_diverse_nodes(
    nodes: list[Any],
    *,
    query: str,
    limit: int,
) -> list[Any]:
    if limit <= 0 or not nodes:
        return []

    source_cap = max(1, settings.retrieval_max_chunks_per_source)
    diversity_penalty = max(0.0, settings.retrieval_diversity_penalty)
    source_penalty = max(0.0, settings.retrieval_source_penalty)
    min_context_score = max(0.0, settings.retrieval_min_context_score)

    selected: list[Any] = []
    source_counts: dict[str, int] = {}
    used_indexes: set[int] = set()

    while len(selected) < limit and len(used_indexes) < len(nodes):
        best_idx: int | None = None
        best_score = float("-inf")

        for idx, node in enumerate(nodes):
            if idx in used_indexes:
                continue
            content, meta = _extract_content_and_metadata(node)
            if not content.strip():
                continue
            source_key = _node_source_key(meta)
            source_count = source_counts.get(source_key, 0)
            if source_count >= source_cap:
                continue

            relevance = _node_relevance_score(query, content, meta)
            novelty_penalty = 0.0
            if selected:
                novelty_penalty = max(
                    _token_jaccard_similarity(content, _extract_content_and_metadata(item)[0])
                    for item in selected
                )
            score = relevance - (diversity_penalty * novelty_penalty) - (source_penalty * source_count)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            # If source cap is too strict for current candidate set, relax it as fallback.
            for idx, node in enumerate(nodes):
                if idx in used_indexes:
                    continue
                content, meta = _extract_content_and_metadata(node)
                if not content.strip():
                    continue
                relevance = _node_relevance_score(query, content, meta)
                novelty_penalty = 0.0
                if selected:
                    novelty_penalty = max(
                        _token_jaccard_similarity(content, _extract_content_and_metadata(item)[0])
                        for item in selected
                    )
                score = relevance - (diversity_penalty * novelty_penalty)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is None:
            break

        if selected and best_score < min_context_score:
            break

        used_indexes.add(best_idx)
        chosen = nodes[best_idx]
        _, chosen_meta = _extract_content_and_metadata(chosen)
        chosen_source = _node_source_key(chosen_meta)
        source_counts[chosen_source] = source_counts.get(chosen_source, 0) + 1
        selected.append(chosen)

    return selected


def _format_context_chunk(content: str, meta: dict[str, Any], rank: int) -> str:
    source = str(meta.get("source") or "local")
    page = meta.get("page")
    header = f"[证据#{rank} | 来源: {source}"
    if page is not None:
        header += f" | 页码: {page}"
    header += "]"
    return f"{header}\n{content}"


def _reorder_for_long_context(nodes: list[Any]) -> list[Any]:
    if len(nodes) <= 2:
        return nodes
    # Keep top-1 at front and top-2 at the end to reduce "lost in the middle" risk.
    head = nodes[:1]
    tail = nodes[1:2]
    middle = nodes[2:]
    return head + middle + tail


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


def _normalize_content_for_fingerprint(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()[:1200]


def _content_fingerprint(content: str) -> str:
    normalized = _normalize_content_for_fingerprint(content)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _evidence_dedup_key(content: str, meta: dict[str, Any]) -> str:
    chunk_id = meta.get("chunk_id") or meta.get("id")
    if chunk_id is not None:
        return f"chunk:{chunk_id}"

    source = str(meta.get("source") or "")
    page = str(meta.get("page") or "")
    document_id = str(meta.get("document_id") or "")
    fingerprint = _content_fingerprint(content)
    if source:
        return f"source:{source}|page:{page}|fp:{fingerprint}"
    if document_id:
        return f"doc:{document_id}|page:{page}|fp:{fingerprint}"
    return f"page:{page}|fp:{fingerprint}"


def _deduplicate_nodes(nodes: list[Any], limit: int | None = None) -> list[Any]:
    deduplicated: list[Any] = []
    seen: set[str] = set()
    max_items = limit if limit is not None and limit > 0 else None

    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        key = _evidence_dedup_key(content, meta)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(node)
        if max_items is not None and len(deduplicated) >= max_items:
            break

    return deduplicated


def _deduplicate_hits(hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        content = str(hit.get("content") or "")
        key = _evidence_dedup_key(
            content,
            {
                "source": hit.get("source"),
                "page": hit.get("page"),
                "document_id": hit.get("document_id"),
            },
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(hit)
        if len(deduplicated) >= limit:
            break
    return deduplicated


def _build_citations_from_nodes(
    nodes, query: str | None = None
) -> list[dict[str, Any]]:
    focus_terms = _query_focus_terms(query or "")

    def _snippet(content: str) -> str:
        text = str(content or "")
        lowered = text.lower()
        for term in focus_terms:
            idx = lowered.find(term)
            if idx >= 0:
                start = max(0, idx - 90)
                end = min(len(text), idx + len(term) + 110)
                return text[start:end]
        return text[:200]

    citations = []
    seen: set[str] = set()
    for node in nodes:
        content, meta = _extract_content_and_metadata(node)
        if not content:
            continue
        key = _evidence_dedup_key(content, meta)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page"),
                "snippet": _snippet(content),
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

    # Add deterministic domain hints early so low-round retrieval can hit anchored docs.
    hint_terms = _domain_query_hints(core or text)
    for hint in hint_terms[:4]:
        _push(hint)
    if len(hint_terms) >= 2:
        _push(" ".join(hint_terms[:3]))

    for token in _LATIN_TOKEN_PATTERN.findall(core.lower()):
        for part in re.split(r"[_-]+", token):
            if len(part) >= 3:
                _push(part)

    for segment in _MULTI_QUERY_SPLIT_PATTERN.split(core):
        candidate = segment.strip()
        if len(candidate) >= 2 and candidate != core:
            _push(candidate)

    if core and core != text:
        if hint_terms:
            _push(f"{hint_terms[0]} 简介")
            _push(f"{hint_terms[0]} 介绍")
        _push(f"{core} 简介")
        _push(f"{core} 是什么")
        _push(f"{core} 介绍")

    max_rewrites = max(1, settings.retrieval_max_rewrites)
    return variants[:max_rewrites]


def _retrieve_candidates(
    index: VectorStoreIndex,
    query: str,
    bm25_index,
    chunks,
    candidate_limit: int,
):
    top_k = max(1, candidate_limit)
    if settings.enable_hybrid_search and bm25_index and chunks:
        return hybrid_retrieve(
            query,
            index,
            chunks,
            bm25_index,
            top_k=top_k,
            candidate_multiplier=max(1, settings.retrieval_candidate_multiplier),
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
        )

    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def _node_dedup_key(node: Any) -> str:
    content, meta = _extract_content_and_metadata(node)
    return _evidence_dedup_key(content, meta)


def _retrieve_with_rewrites(
    index: VectorStoreIndex,
    question: str,
    bm25_index,
    chunks,
) -> list[Any]:
    merged: list[Any] = []
    seen_keys: set[str] = set()
    limit = _candidate_limit_for_question(question, corpus_size=len(chunks or []))
    rewrites = _rewrite_queries(question)
    if not rewrites:
        return []
    per_rewrite_limit = max(2, limit // max(1, len(rewrites)))

    for rewritten_query in rewrites:
        for node in _retrieve_candidates(
            index,
            rewritten_query,
            bm25_index,
            chunks,
            candidate_limit=per_rewrite_limit,
        ):
            key = _node_dedup_key(node)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(node)
            if len(merged) >= limit:
                return merged

    if len(merged) < limit:
        for node in _retrieve_candidates(
            index,
            rewrites[0],
            bm25_index,
            chunks,
            candidate_limit=limit,
        ):
            key = _node_dedup_key(node)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(node)
            if len(merged) >= limit:
                break

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


def _fallback_extractive_answer(
    *,
    question: str,
    evidence_chunks: list[str],
) -> str:
    if not evidence_chunks:
        return "未能检索到相关上下文。"

    snippets: list[str] = []
    for content in evidence_chunks[:3]:
        normalized = re.sub(r"\s+", " ", str(content or "")).strip()
        if normalized:
            snippets.append(normalized[:220])

    if not snippets:
        return "未能检索到相关上下文。"

    focus_terms = _query_focus_terms(question)
    if len(snippets) == 1:
        single = snippets[0]
        if focus_terms:
            return f"主题：{focus_terms[0]}\n{single}"
        return single

    lines = ["基于已检索证据，先给出关键要点："]
    if focus_terms:
        lines.append(f"主题：{focus_terms[0]}")
    for idx, snippet in enumerate(snippets, start=1):
        lines.append(f"{idx}. {snippet}")
    return "\n".join(lines)


def _fallback_retrieve_from_chunks(
    *,
    query: str,
    kb_id: int,
    bm25_index,
    chunks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    top_k = max(1, limit)
    hits: list[dict[str, Any]] = []
    query_tokens = _relevance_tokens(query)
    focus_terms = _query_focus_terms(query)
    latin_focus_terms = [
        term
        for term in focus_terms
        if _LATIN_TOKEN_PATTERN.fullmatch(term) is not None and len(term) >= 4
    ]

    if bm25_index and chunks:
        pool_size = min(
            max(top_k, settings.retrieval_max_candidates),
            max(top_k * 2, top_k + 24),
        )
        for chunk, score in bm25_search(query, chunks, bm25_index, top_k=pool_size):
            payload = dict(chunk)
            payload["fallback_score"] = float(score)
            payload.setdefault("source", payload.get("source") or "local")
            hits.append(payload)
        if query_tokens:
            path_match_budget = min(96, max(24, top_k))
            path_matched = 0
            for chunk in chunks:
                source_text = str(chunk.get("source") or "").lower()
                if not source_text:
                    continue
                if not any(token in source_text for token in query_tokens):
                    continue
                payload = dict(chunk)
                payload["fallback_score"] = float(payload.get("fallback_score") or 0.0) + 3.0
                payload.setdefault("source", payload.get("source") or "local")
                hits.append(payload)
                path_matched += 1
                if path_matched >= path_match_budget:
                    break
        if latin_focus_terms:
            fuzzy_budget = min(128, max(32, top_k * 2))
            fuzzy_matched = 0
            for chunk in chunks:
                source_text = str(chunk.get("source") or "").lower()
                content_text = str(chunk.get("content") or "").lower()
                if not source_text and not content_text:
                    continue
                if not any(
                    term in source_text
                    or term in content_text
                    or any(term in part for part in re.split(r"[^a-z0-9_]+", source_text))
                    for term in latin_focus_terms
                ):
                    continue
                payload = dict(chunk)
                payload["fallback_score"] = float(payload.get("fallback_score") or 0.0) + 2.6
                payload.setdefault("source", payload.get("source") or "local")
                hits.append(payload)
                fuzzy_matched += 1
                if fuzzy_matched >= fuzzy_budget:
                    break

    if not hits:
        lexical_hits = search_chunks(query, limit=top_k, kb_id=kb_id)
        for hit in lexical_hits:
            hits.append(
                {
                    "content": str(hit.get("content") or ""),
                    "source": "local",
                    "page": hit.get("page"),
                    "document_id": None,
                    "fallback_score": 0.0,
                }
            )

    if not hits:
        return []

    def _fallback_rank(hit: dict[str, Any]) -> float:
        base = float(hit.get("fallback_score") or 0.0)
        source = str(hit.get("source") or "").lower()
        content = str(hit.get("content") or "").lower()
        source_hits = 0
        content_hits = 0
        for token in query_tokens:
            if token and token in source:
                source_hits += 1
            elif token and token in content:
                content_hits += 1
        return base + (source_hits * 2.5) + (content_hits * 0.25)

    ranked = sorted(hits, key=_fallback_rank, reverse=True)
    deduped = _deduplicate_hits(ranked, top_k)
    return deduped[:top_k]


def _retrieve_nodes_with_fallback(
    *,
    index: Optional[VectorStoreIndex],
    retrieval_question: str,
    enhanced_question: str,
    bm25_index,
    chunks: list[dict[str, Any]],
    kb_id: int,
) -> tuple[list[Any], bool]:
    if index is None:
        fallback_nodes = _fallback_retrieve_from_chunks(
            query=retrieval_question,
            kb_id=kb_id,
            bm25_index=bm25_index,
            chunks=chunks,
            limit=_candidate_limit_for_question(
                retrieval_question, corpus_size=len(chunks)
            ),
        )
        return fallback_nodes, True

    try:
        retrieved_nodes = _retrieve_with_rewrites(
            index,
            retrieval_question,
            bm25_index,
            chunks,
        )
        if not retrieved_nodes and retrieval_question != enhanced_question:
            retrieved_nodes = _retrieve_with_rewrites(
                index,
                enhanced_question,
                bm25_index,
                chunks,
            )
        if not retrieved_nodes:
            fallback_nodes = _fallback_retrieve_from_chunks(
                query=retrieval_question,
                kb_id=kb_id,
                bm25_index=bm25_index,
                chunks=chunks,
                limit=_candidate_limit_for_question(
                    retrieval_question, corpus_size=len(chunks)
                ),
            )
            if not fallback_nodes and retrieval_question != enhanced_question:
                fallback_nodes = _fallback_retrieve_from_chunks(
                    query=enhanced_question,
                    kb_id=kb_id,
                    bm25_index=bm25_index,
                    chunks=chunks,
                    limit=_candidate_limit_for_question(
                        enhanced_question, corpus_size=len(chunks)
                    ),
                )
            return fallback_nodes, True
        if not _nodes_match_focus_terms(retrieved_nodes, retrieval_question):
            fallback_nodes = _fallback_retrieve_from_chunks(
                query=retrieval_question,
                kb_id=kb_id,
                bm25_index=bm25_index,
                chunks=chunks,
                limit=_candidate_limit_for_question(
                    retrieval_question, corpus_size=len(chunks)
                ),
            )
            if fallback_nodes:
                return fallback_nodes, True
        return retrieved_nodes, False
    except Exception:
        fallback_nodes = _fallback_retrieve_from_chunks(
            query=retrieval_question,
            kb_id=kb_id,
            bm25_index=bm25_index,
            chunks=chunks,
            limit=_candidate_limit_for_question(
                retrieval_question, corpus_size=len(chunks)
            ),
        )
        if not fallback_nodes and retrieval_question != enhanced_question:
            fallback_nodes = _fallback_retrieve_from_chunks(
                query=enhanced_question,
                kb_id=kb_id,
                bm25_index=bm25_index,
                chunks=chunks,
                limit=_candidate_limit_for_question(
                    enhanced_question, corpus_size=len(chunks)
                ),
            )
        return fallback_nodes, True


def _stream_answer_from_context(
    question: str, context: str
) -> Generator[str, None, None]:
    llm = get_llm()
    prompt = (
        "你是严格的知识库问答助手。\n"
        "只允许使用给定上下文，不得使用外部知识。\n"
        "如果上下文不足以回答，请只输出：未找到相关信息。\n"
        "必须保留证据中的关键术语、实体名、编号、代码与专有 token 原文，不可改写。\n"
        "若问题询问定义或结论，请优先给出上下文原句，至少保留核心词原文。\n"
        "引用时优先使用证据编号格式（例如：[证据#1]）。\n"
        '请在结尾给出"引用来源"并仅引用上下文中的证据。\n'
        f"问题：{question}\n\n"
        f"上下文：\n{context}\n"
    )
    try:
        yield from _iter_stream_chunks(llm.stream_complete(prompt))
    except Exception:
        fallback_answer = _fallback_extractive_answer(
            question=question,
            evidence_chunks=[context],
        )
        yield from _split_stream_text(fallback_answer)


def _apply_reranking(
    nodes: list[Any],
    query: str,
    final_limit: int,
) -> list[Any]:
    if not nodes or not settings.rerank_model:
        return nodes

    if len(nodes) <= final_limit:
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

    reranked = rerank_documents(query, docs, top_k=final_limit)
    reranked_nodes: list[Any] = []
    for doc in reranked:
        node = doc.get("node")
        if node is None:
            continue
        score = doc.get("relevance_score")
        if score is not None:
            if isinstance(node, dict):
                node["rerank_score"] = score
            else:
                metadata = getattr(node, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["rerank_score"] = score
        reranked_nodes.append(node)
    return reranked_nodes


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
        retrieval_question = _extract_focus_question(enhanced_question)

        cached = semantic_cache.get(enhanced_question, kb_id=kb_id)
        if cached:
            return {"answer": cached["answer"], "citations": cached["citations"]}

        if settings.mock_mode or index is None:
            return self._mock_query(enhanced_question, kb_id=kb_id)

        bm25_index, chunks = self._get_bm25_index(kb_id)

        retrieved_nodes, degraded_mode = _retrieve_nodes_with_fallback(
            index=index,
            retrieval_question=retrieval_question,
            enhanced_question=enhanced_question,
            bm25_index=bm25_index,
            chunks=chunks,
            kb_id=kb_id,
        )

        context_limit = _context_limit_for_question(retrieval_question)
        rerank_limit = max(settings.rerank_top_k, context_limit * 2)
        reranked_nodes = (
            _apply_reranking(retrieved_nodes, retrieval_question, final_limit=rerank_limit)
            if not degraded_mode
            else list(retrieved_nodes)
        )
        unfiltered_nodes = list(reranked_nodes)
        reranked_nodes = _filter_nodes_by_quality(reranked_nodes, retrieval_question)
        if not reranked_nodes and unfiltered_nodes:
            reranked_nodes = unfiltered_nodes
        reranked_nodes = _deduplicate_nodes(reranked_nodes)
        reranked_nodes = _filter_nodes_by_focus_terms(reranked_nodes, retrieval_question)
        reranked_nodes = _filter_low_signal_nodes(
            reranked_nodes, retrieval_question
        )
        selected_nodes = _select_diverse_nodes(
            reranked_nodes,
            query=retrieval_question,
            limit=context_limit,
        )
        if not selected_nodes and reranked_nodes:
            selected_nodes = reranked_nodes[:context_limit]
        selected_nodes = _reorder_for_long_context(selected_nodes)
        if _should_abstain_for_low_coverage(retrieval_question, selected_nodes):
            selected_nodes = []

        citations = _build_citations_from_nodes(selected_nodes, retrieval_question)

        raw_context_chunks: list[str] = []
        context_chunks: list[str] = []
        for idx, node in enumerate(selected_nodes, start=1):
            content, meta = _extract_content_and_metadata(node)
            if not content:
                continue
            raw_context_chunks.append(content)
            context_chunks.append(_format_context_chunk(content, meta, idx))

        if _context_is_unusable(raw_context_chunks):
            context_chunks = []
            citations = []

        if not context_chunks:
            return {"answer": "未能检索到相关上下文。", "citations": []}

        context = "\n\n".join(context_chunks)
        llm = get_llm()
        prompt = (
            "你是严格的知识库问答助手。\n"
            "只允许使用给定上下文，不得使用外部知识。\n"
            "如果上下文不足以回答，请只输出：未找到相关信息。\n"
            "必须保留证据中的关键术语、实体名、编号、代码与专有 token 原文，不可改写。\n"
            "若问题询问定义或结论，请优先给出上下文原句，至少保留核心词原文。\n"
            "引用时优先使用证据编号格式（例如：[证据#1]）。\n"
            '请在结尾给出"引用来源"并仅引用上下文中的证据。\n'
            f"问题：{question}\n\n"
            f"上下文：\n{context}\n"
        )
        try:
            response = llm.complete(prompt)
            answer = response.text
        except Exception:
            answer = _fallback_extractive_answer(
                question=question,
                evidence_chunks=raw_context_chunks,
            )

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
        retrieval_question = _extract_focus_question(enhanced_question)

        cached = semantic_cache.get(enhanced_question, kb_id=kb_id)
        if cached:
            return _single_chunk_stream(cached["answer"]), cached["citations"]

        if settings.mock_mode or index is None:
            return self._mock_query_stream(enhanced_question, kb_id=kb_id)

        bm25_index, chunks = self._get_bm25_index(kb_id)

        retrieved_nodes, degraded_mode = _retrieve_nodes_with_fallback(
            index=index,
            retrieval_question=retrieval_question,
            enhanced_question=enhanced_question,
            bm25_index=bm25_index,
            chunks=chunks,
            kb_id=kb_id,
        )

        context_limit = _context_limit_for_question(retrieval_question)
        rerank_limit = max(settings.rerank_top_k, context_limit * 2)
        reranked_nodes = (
            _apply_reranking(retrieved_nodes, retrieval_question, final_limit=rerank_limit)
            if not degraded_mode
            else list(retrieved_nodes)
        )
        unfiltered_nodes = list(reranked_nodes)
        reranked_nodes = _filter_nodes_by_quality(reranked_nodes, retrieval_question)
        if not reranked_nodes and unfiltered_nodes:
            reranked_nodes = unfiltered_nodes
        reranked_nodes = _deduplicate_nodes(reranked_nodes)
        reranked_nodes = _filter_nodes_by_focus_terms(reranked_nodes, retrieval_question)
        reranked_nodes = _filter_low_signal_nodes(
            reranked_nodes, retrieval_question
        )
        selected_nodes = _select_diverse_nodes(
            reranked_nodes,
            query=retrieval_question,
            limit=context_limit,
        )
        if not selected_nodes and reranked_nodes:
            selected_nodes = reranked_nodes[:context_limit]
        selected_nodes = _reorder_for_long_context(selected_nodes)
        if _should_abstain_for_low_coverage(retrieval_question, selected_nodes):
            selected_nodes = []
        citations = _build_citations_from_nodes(selected_nodes, retrieval_question)

        raw_context_chunks: list[str] = []
        context_chunks: list[str] = []
        for idx, node in enumerate(selected_nodes, start=1):
            content, meta = _extract_content_and_metadata(node)
            if content:
                raw_context_chunks.append(content)
                context_chunks.append(_format_context_chunk(content, meta, idx))

        if _context_is_unusable(raw_context_chunks):
            context_chunks = []
            citations = []

        if not context_chunks:
            return _single_chunk_stream("未能检索到相关上下文。"), []

        context = "\n\n".join(context_chunks)

        def _stream_with_cache():
            chunks = []
            try:
                for chunk in _stream_answer_from_context(question, context):
                    chunks.append(chunk)
                    yield chunk
            except Exception:
                fallback_answer = _fallback_extractive_answer(
                    question=question,
                    evidence_chunks=raw_context_chunks,
                )
                for chunk in _split_stream_text(fallback_answer):
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
        hits = _deduplicate_hits(hits, settings.max_context_chunks)
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
        hits = _deduplicate_hits(hits, settings.max_context_chunks)
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
    retrieval_question = _extract_focus_question(enhanced_question)
    rewrites = _rewrite_queries(retrieval_question)

    if settings.mock_mode or index is None:
        hits = search_chunks(retrieval_question, limit=effective_top_k, kb_id=kb_id)
        if not hits:
            hits = list_chunks(limit=effective_top_k, kb_id=kb_id)
        hits = _filter_placeholder_hits(hits)
        hits = _deduplicate_hits(hits, effective_top_k)
        diagnostics_hits = [
            {
                "source": "local",
                "page": hit.get("page"),
                "snippet": str(hit.get("content") or "")[:220],
                "content": str(hit.get("content") or ""),
            }
            for hit in hits[:effective_top_k]
        ]
        return {
            "query": enhanced_question,
            "retrieval_query": retrieval_question,
            "rewrites": rewrites,
            "hits": diagnostics_hits,
            "diagnostics": {
                "candidate_count": len(diagnostics_hits),
                "kept_count": len(diagnostics_hits),
                "low_confidence": len(diagnostics_hits) == 0,
                "avg_token_overlap": _average_hit_overlap(
                    retrieval_question, diagnostics_hits
                ),
            },
        }

    bm25_index, chunks = rag_pipeline._get_bm25_index(kb_id)
    retrieved_nodes, degraded_mode = _retrieve_nodes_with_fallback(
        index=index,
        retrieval_question=retrieval_question,
        enhanced_question=enhanced_question,
        bm25_index=bm25_index,
        chunks=chunks,
        kb_id=kb_id,
    )
    rerank_limit = max(settings.rerank_top_k, effective_top_k * 2)
    reranked_nodes = (
        _apply_reranking(retrieved_nodes, retrieval_question, final_limit=rerank_limit)
        if not degraded_mode
        else list(retrieved_nodes)
    )
    candidate_count = len(reranked_nodes)
    unfiltered_nodes = list(reranked_nodes)
    reranked_nodes = _filter_nodes_by_quality(reranked_nodes, retrieval_question)
    if not reranked_nodes and unfiltered_nodes:
        reranked_nodes = unfiltered_nodes
    reranked_nodes = _deduplicate_nodes(reranked_nodes)
    reranked_nodes = _filter_nodes_by_focus_terms(reranked_nodes, retrieval_question)
    reranked_nodes = _filter_low_signal_nodes(reranked_nodes, retrieval_question)
    selected_nodes = _select_diverse_nodes(
        reranked_nodes,
        query=retrieval_question,
        limit=effective_top_k,
    )
    if not selected_nodes and reranked_nodes:
        selected_nodes = reranked_nodes[:effective_top_k]
    selected_nodes = _reorder_for_long_context(selected_nodes)
    if _should_abstain_for_low_coverage(retrieval_question, selected_nodes):
        selected_nodes = []

    payload_hits: list[dict[str, Any]] = []
    for node in selected_nodes:
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
        "retrieval_query": retrieval_question,
        "rewrites": rewrites,
        "hits": payload_hits,
        "diagnostics": {
            "candidate_count": candidate_count,
            "kept_count": len(payload_hits),
            "low_confidence": len(payload_hits) == 0,
            "avg_token_overlap": _average_hit_overlap(retrieval_question, payload_hits),
        },
    }
