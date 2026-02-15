from __future__ import annotations

import json
import hashlib
import math
import random
import re
from itertools import product
from pathlib import Path
from typing import Any, Optional

from llama_index.core import VectorStoreIndex

from .config import settings
from .db import search_chunks
from .hybrid_search import build_bm25_index, get_all_chunks_from_db, hybrid_retrieve
from .indexer import get_llm
from .reranker import rerank_documents


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "retrieval_benchmark.json"
)

DEFAULT_TUNING_PARAMETER_GRID: dict[str, list[Any]] = {
    "candidate_top_k": [6, 8, 10, 12],
    "final_top_k": [4, 6],
    "candidate_multiplier": [2],
    "enable_hybrid_search": [True],
    "rrf_k": [30, 60],
    "vector_weight": [0.6, 0.7, 0.8],
    "bm25_weight": [0.4, 0.3, 0.2],
    "use_reranker": [False, True],
    "rerank_top_k": [6, 8],
}


def build_default_tuning_parameter_grid() -> dict[str, list[Any]]:
    return {key: list(values) for key, values in DEFAULT_TUNING_PARAMETER_GRID.items()}


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_unit_score(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if numeric > 1.0:
        numeric = numeric / 5.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def load_default_benchmark_dataset(
    path: Path = DEFAULT_BENCHMARK_PATH,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Default benchmark dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_parameter_grid(
    parameter_grid: Optional[dict[str, list[Any]]],
) -> list[dict[str, Any]]:
    if not parameter_grid:
        return [{}]

    keys: list[str] = []
    values: list[list[Any]] = []
    for key, raw_values in parameter_grid.items():
        keys.append(key)
        if isinstance(raw_values, list):
            values.append(raw_values)
        else:
            values.append([raw_values])

    return [dict(zip(keys, combo)) for combo in product(*values)]


def _normalize_node(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "id": item.get("id"),
            "document_id": item.get("document_id"),
            "source": item.get("source"),
            "page": item.get("page"),
            "content": item.get("content", ""),
        }

    node = getattr(item, "node", item)
    metadata = getattr(node, "metadata", {}) or {}
    get_content = getattr(node, "get_content", None)
    content = get_content() if callable(get_content) else ""
    return {
        "id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "content": content,
    }


def _normalize_content_for_fingerprint(content: Any) -> str:
    raw = str(content or "")
    return re.sub(r"\s+", " ", raw).strip()[:1200]


def _item_dedup_key(item: dict[str, Any]) -> str:
    item_id = item.get("id")
    if item_id is not None:
        return f"id:{item_id}"

    doc_id = item.get("document_id")
    source = str(item.get("source") or "")
    page = str(item.get("page") or "")
    normalized_content = _normalize_content_for_fingerprint(item.get("content"))
    fingerprint = (
        hashlib.sha1(normalized_content.encode("utf-8")).hexdigest()
        if normalized_content
        else ""
    )
    if source:
        return f"source:{source}|page:{page}|fp:{fingerprint}"
    if doc_id is not None:
        return f"doc:{doc_id}|page:{page}|fp:{fingerprint}"
    return f"page:{page}|fp:{fingerprint}"


def _deduplicate_ranked_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = _item_dedup_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def _retrieve_ranked(
    *,
    index: Optional[VectorStoreIndex],
    query: str,
    config: dict[str, Any],
    kb_id: int,
    chunks: list[dict[str, Any]],
    bm25_index: Any,
) -> list[dict[str, Any]]:
    k = max(1, int(config.get("candidate_top_k", settings.rerank_top_k)))
    final_top_k = max(1, int(config.get("final_top_k", k)))
    candidate_multiplier = max(1, int(config.get("candidate_multiplier", 2)))
    use_hybrid = bool(config.get("enable_hybrid_search", settings.enable_hybrid_search))
    rrf_k = max(1, int(config.get("rrf_k", 60)))
    vector_weight = float(config.get("vector_weight", settings.vector_weight))
    bm25_weight = float(config.get("bm25_weight", settings.bm25_weight))

    if index is not None:
        if use_hybrid and chunks and bm25_index is not None:
            ranked = hybrid_retrieve(
                query,
                index,
                chunks,
                bm25_index,
                top_k=k,
                rrf_k=rrf_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                candidate_multiplier=candidate_multiplier,
            )
        else:
            retriever = index.as_retriever(similarity_top_k=k * candidate_multiplier)
            ranked = retriever.retrieve(query)
        normalized = _deduplicate_ranked_items(
            [_normalize_node(item) for item in ranked]
        )
    else:
        normalized = _deduplicate_ranked_items(
            [
                _normalize_node(item)
                for item in search_chunks(query, limit=k, kb_id=kb_id)
            ]
        )

    use_reranker = bool(config.get("use_reranker", False))
    rerank_top_k = max(1, int(config.get("rerank_top_k", k)))
    if use_reranker and normalized:
        docs = [
            {"content": item.get("content", ""), "item": item} for item in normalized
        ]
        reranked = rerank_documents(query, docs, top_k=rerank_top_k)
        normalized = _deduplicate_ranked_items(
            [doc["item"] for doc in reranked if "item" in doc]
        )

    return normalized[:final_top_k]


def _count_relevant_targets(case: dict[str, Any]) -> int:
    if case.get("relevant_ids"):
        return len(set(case["relevant_ids"]))
    if case.get("relevant_snippets"):
        return len(case["relevant_snippets"])
    if case.get("relevant_pages"):
        return len(set(case["relevant_pages"]))
    if case.get("relevant_sources"):
        return len(set(case["relevant_sources"]))
    return 0


def _contains_any_snippet(content: str, snippets: list[Any]) -> bool:
    lowered = content.lower()
    for snippet in snippets:
        token = str(snippet or "").strip().lower()
        if token and token in lowered:
            return True
    return False


def _to_snippet_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item or "") for item in value if str(item or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _tokenize_for_similarity(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) if token}


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize_for_similarity(left)
    right_tokens = _tokenize_for_similarity(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens.intersection(right_tokens)
    union = left_tokens.union(right_tokens)
    return len(intersection) / len(union)


def _is_relevant(item: dict[str, Any], case: dict[str, Any]) -> bool:
    if case.get("relevant_ids"):
        return item.get("id") in set(case["relevant_ids"])

    if case.get("relevant_snippets"):
        content = str(item.get("content") or "").lower()
        return _contains_any_snippet(content, case.get("relevant_snippets", []))

    if case.get("relevant_pages"):
        return item.get("page") in set(case["relevant_pages"])

    if case.get("relevant_sources"):
        return item.get("source") in set(case["relevant_sources"])

    return False


def _is_hard_negative(item: dict[str, Any], case: dict[str, Any]) -> bool:
    hard_negative_ids = set(case.get("hard_negative_ids") or [])
    if hard_negative_ids and item.get("id") in hard_negative_ids:
        return True

    hard_negative_pages = set(case.get("hard_negative_pages") or [])
    if hard_negative_pages and item.get("page") in hard_negative_pages:
        return True

    hard_negative_sources = set(case.get("hard_negative_sources") or [])
    if hard_negative_sources and item.get("source") in hard_negative_sources:
        return True

    hard_negative_snippets = list(case.get("hard_negative_snippets") or [])
    if hard_negative_snippets:
        content = str(item.get("content") or "")
        if _contains_any_snippet(content, hard_negative_snippets):
            return True

    return False


def _compute_metrics(
    relevance_flags: list[int],
    hard_negative_flags: list[int],
    relevant_total: int,
    k: int,
) -> dict[str, float]:
    top = relevance_flags[:k]
    hard_negative_top = hard_negative_flags[:k]
    top_count = len(top)
    hits = sum(top)

    precision = hits / k if k > 0 else 0.0
    recall = hits / relevant_total if relevant_total > 0 else 0.0
    hit_rate = 1.0 if hits > 0 else 0.0

    reciprocal_rank = 0.0
    for idx, rel in enumerate(top, start=1):
        if rel:
            reciprocal_rank = 1.0 / idx
            break

    dcg = 0.0
    for idx, rel in enumerate(top, start=1):
        if rel:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(relevant_total, k)
    idcg = 0.0
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    hard_negative_hits = sum(hard_negative_top)
    hard_negative_hit_rate = 1.0 if hard_negative_hits > 0 else 0.0
    hard_negative_ratio = hard_negative_hits / k if k > 0 else 0.0

    false_positive_ratio = (top_count - hits) / top_count if top_count > 0 else 0.0
    empty_result_rate = 1.0 if top_count == 0 else 0.0
    abstain_success = 1.0 if relevant_total <= 0 and top_count == 0 else 0.0

    return {
        "precision@k": precision,
        "recall@k": recall,
        "hit_rate@k": hit_rate,
        "mrr@k": reciprocal_rank,
        "ndcg@k": ndcg,
        "hard_negative_hit_rate@k": hard_negative_hit_rate,
        "hard_negative_ratio@k": hard_negative_ratio,
        "false_positive_ratio@k": false_positive_ratio,
        "empty_result_rate@k": empty_result_rate,
        "abstain_success@k": abstain_success,
    }


def _best_config_sort_key(result: dict[str, Any]) -> tuple[float, ...]:
    metrics = result.get("metrics") or {}
    judge = result.get("llm_judge") or {}
    judge_scores = judge.get("scores") if isinstance(judge, dict) else {}
    judge_scores = judge_scores if isinstance(judge_scores, dict) else {}

    hard_negative_ratio = _normalize_unit_score(
        metrics.get("hard_negative_ratio@k", 0.0)
    )
    hard_negative_hit_rate = _normalize_unit_score(
        metrics.get("hard_negative_hit_rate@k", 0.0)
    )

    has_judge = 1.0 if judge_scores else 0.0
    judge_overall = _normalize_unit_score(judge_scores.get("overall", 0.0))
    judge_faithfulness = _normalize_unit_score(judge_scores.get("faithfulness", 0.0))
    judge_citation_precision = _normalize_unit_score(
        judge_scores.get("citation_precision", 0.0)
    )
    judge_citation_recall = _normalize_unit_score(
        judge_scores.get("citation_recall", 0.0)
    )

    return (
        has_judge,
        judge_overall,
        judge_faithfulness,
        judge_citation_precision,
        judge_citation_recall,
        1.0 - float(metrics.get("false_positive_ratio@k", 0.0)),
        float(metrics.get("abstain_success@k", 0.0)),
        1.0 - hard_negative_ratio,
        1.0 - hard_negative_hit_rate,
        float(metrics.get("ndcg@k", 0.0)),
        float(metrics.get("mrr@k", 0.0)),
        float(metrics.get("precision@k", 0.0)),
        float(metrics.get("recall@k", 0.0)),
        float(metrics.get("hit_rate@k", 0.0)),
    )


def _judge_single_case(
    query: str,
    hits: list[dict[str, Any]],
    expected: dict[str, Any],
) -> dict[str, Any]:
    llm = get_llm()
    compact_hits = [
        {
            "source": hit.get("source"),
            "page": hit.get("page"),
            "snippet": str(hit.get("content") or "")[:320],
        }
        for hit in hits
    ]
    expected_hints = {
        "relevant_ids": expected.get("relevant_ids", []),
        "relevant_sources": expected.get("relevant_sources", []),
        "relevant_pages": expected.get("relevant_pages", []),
        "relevant_snippets": expected.get("relevant_snippets", []),
    }

    prompt = (
        "You are a strict RAG evaluator. Score retrieval quality for one query.\n"
        "Return JSON only with keys: relevance, coverage, groundedness, citation_precision, citation_recall, faithfulness, reason.\n"
        "- relevance: how well retrieved chunks match user intent (0-1).\n"
        "- coverage: whether key evidence likely exists in retrieved chunks (0-1).\n"
        "- groundedness: internal consistency/non-noise quality of retrieved evidence (0-1).\n"
        "- citation_precision: proportion of retrieved snippets that are truly useful evidence (0-1).\n"
        "- citation_recall: proportion of expected evidence covered by retrieved snippets (0-1).\n"
        "- faithfulness: confidence that answer built from these snippets would stay grounded (0-1).\n"
        "Use conservative scoring.\n\n"
        f"Query: {query}\n"
        f"Expected clues: {json.dumps(expected_hints, ensure_ascii=False)}\n"
        f"Retrieved chunks: {json.dumps(compact_hits, ensure_ascii=False)}"
    )
    raw = llm.complete(prompt).text
    parsed = _extract_json_object(raw)
    if parsed is None:
        return {
            "relevance": 0.0,
            "coverage": 0.0,
            "groundedness": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "faithfulness": 0.0,
            "reason": "judge_parse_failed",
        }
    return {
        "relevance": _normalize_unit_score(parsed.get("relevance")),
        "coverage": _normalize_unit_score(parsed.get("coverage")),
        "groundedness": _normalize_unit_score(parsed.get("groundedness")),
        "citation_precision": _normalize_unit_score(parsed.get("citation_precision")),
        "citation_recall": _normalize_unit_score(parsed.get("citation_recall")),
        "faithfulness": _normalize_unit_score(parsed.get("faithfulness")),
        "reason": str(parsed.get("reason") or ""),
    }


def _evaluate_llm_judge(
    judge_cases: list[dict[str, Any]],
    sample_size: int,
) -> dict[str, Any]:
    if not judge_cases:
        return {
            "enabled": True,
            "sample_size": 0,
            "scores": {
                "relevance": 0.0,
                "coverage": 0.0,
                "groundedness": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "faithfulness": 0.0,
                "overall": 0.0,
            },
            "cases": [],
        }

    rng = random.Random(42)
    if sample_size > 0 and len(judge_cases) > sample_size:
        selected_cases = rng.sample(judge_cases, sample_size)
    else:
        selected_cases = list(judge_cases)

    judged_cases: list[dict[str, Any]] = []
    sum_relevance = 0.0
    sum_coverage = 0.0
    sum_groundedness = 0.0
    sum_citation_precision = 0.0
    sum_citation_recall = 0.0
    sum_faithfulness = 0.0

    for case in selected_cases:
        judged = _judge_single_case(
            query=str(case.get("query") or ""),
            hits=case.get("hits", []),
            expected=case.get("expected", {}),
        )
        sum_relevance += judged["relevance"]
        sum_coverage += judged["coverage"]
        sum_groundedness += judged["groundedness"]
        sum_citation_precision += judged["citation_precision"]
        sum_citation_recall += judged["citation_recall"]
        sum_faithfulness += judged["faithfulness"]
        judged_cases.append(
            {
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "scores": {
                    "relevance": judged["relevance"],
                    "coverage": judged["coverage"],
                    "groundedness": judged["groundedness"],
                    "citation_precision": judged["citation_precision"],
                    "citation_recall": judged["citation_recall"],
                    "faithfulness": judged["faithfulness"],
                    "overall": (
                        judged["relevance"]
                        + judged["coverage"]
                        + judged["groundedness"]
                        + judged["citation_precision"]
                        + judged["citation_recall"]
                        + judged["faithfulness"]
                    )
                    / 6.0,
                },
                "reason": judged["reason"],
            }
        )

    count = len(judged_cases)
    if count == 0:
        avg_relevance = 0.0
        avg_coverage = 0.0
        avg_groundedness = 0.0
        avg_citation_precision = 0.0
        avg_citation_recall = 0.0
        avg_faithfulness = 0.0
    else:
        avg_relevance = sum_relevance / count
        avg_coverage = sum_coverage / count
        avg_groundedness = sum_groundedness / count
        avg_citation_precision = sum_citation_precision / count
        avg_citation_recall = sum_citation_recall / count
        avg_faithfulness = sum_faithfulness / count

    return {
        "enabled": True,
        "sample_size": count,
        "scores": {
            "relevance": avg_relevance,
            "coverage": avg_coverage,
            "groundedness": avg_groundedness,
            "citation_precision": avg_citation_precision,
            "citation_recall": avg_citation_recall,
            "faithfulness": avg_faithfulness,
            "overall": (
                avg_relevance
                + avg_coverage
                + avg_groundedness
                + avg_citation_precision
                + avg_citation_recall
                + avg_faithfulness
            )
            / 6.0,
        },
        "cases": judged_cases,
    }


def _derive_query_from_chunk(content: str) -> str:
    cleaned = re.sub(r"\s+", " ", content).strip()
    if not cleaned:
        return "这段内容的核心是什么？"
    short = cleaned[:80]
    short = re.sub(r"[\.;。！？!?,，]+$", "", short)
    return f"请解释：{short}"


def _generate_case_with_llm(content: str) -> dict[str, Any] | None:
    llm = get_llm()
    prompt = (
        "Generate one evaluation case for RAG retrieval from the chunk below.\n"
        "Return JSON only with keys: query, relevant_snippet.\n"
        "query must be a concise user question.\n"
        "relevant_snippet must be an exact short phrase copied from chunk (<= 40 chars).\n"
        f"Chunk: {content[:1200]}"
    )
    parsed = _extract_json_object(llm.complete(prompt).text)
    if parsed is None:
        return None
    query = str(parsed.get("query") or "").strip()
    snippet = str(parsed.get("relevant_snippet") or "").strip()
    if not query or not snippet:
        return None
    return {"query": query, "relevant_snippets": [snippet]}


def generate_retrieval_benchmark_dataset(
    kb_id: int,
    count: int = 20,
    use_llm: bool = True,
) -> dict[str, Any]:
    chunks = get_all_chunks_from_db(kb_id=kb_id)
    qualified = [
        chunk for chunk in chunks if len(str(chunk.get("content") or "")) >= 20
    ]
    if not qualified:
        return {
            "kb_id": kb_id,
            "k": 5,
            "parameter_grid": build_default_tuning_parameter_grid(),
            "cases": [],
            "generated": 0,
        }

    rng = random.Random(42)
    if len(qualified) > count:
        sampled = rng.sample(qualified, count)
    else:
        sampled = qualified

    cases: list[dict[str, Any]] = []
    all_contents = [str(item.get("content") or "") for item in qualified]
    for idx, chunk in enumerate(sampled):
        content = str(chunk.get("content") or "")
        generated = _generate_case_with_llm(content) if use_llm else None
        if generated is None:
            generated = {
                "query": _derive_query_from_chunk(content),
                "relevant_snippets": [content[:36]],
            }

        relevant_snippets = _to_snippet_list(generated.get("relevant_snippets"))
        if not relevant_snippets:
            relevant_snippets = [content[:36]]

        negative_pool = [
            text
            for text in all_contents
            if text
            and text != content
            and not _contains_any_snippet(text, relevant_snippets)
        ]
        hard_negative_snippets: list[str] = []
        if negative_pool:
            ranked_negatives = sorted(
                negative_pool,
                key=lambda candidate: (
                    _jaccard_similarity(content, candidate),
                    len(candidate),
                ),
                reverse=True,
            )
            top_bucket = ranked_negatives[: min(3, len(ranked_negatives))]
            chosen_negative = rng.choice(top_bucket)
            hard_negative_snippets = [chosen_negative[:36]]

        cases.append(
            {
                "id": f"kb{kb_id}-case-{idx + 1}",
                "query": generated["query"],
                "relevant_snippets": relevant_snippets,
                "hard_negative_snippets": hard_negative_snippets,
            }
        )

    return {
        "kb_id": kb_id,
        "k": 5,
        "parameter_grid": build_default_tuning_parameter_grid(),
        "cases": cases,
        "generated": len(cases),
    }


def run_retrieval_evaluation(
    *,
    index: Optional[VectorStoreIndex],
    cases: list[dict[str, Any]],
    parameter_grid: Optional[dict[str, list[Any]]] = None,
    kb_id: int = 1,
    k: int = 5,
    include_case_results: bool = False,
    include_llm_judge: bool = False,
    llm_judge_sample_size: int = 10,
    llm_judge_on_all_configs: bool = False,
) -> dict[str, Any]:
    if not cases:
        return {
            "k": k,
            "case_count": 0,
            "config_count": 0,
            "results": [],
            "best_config": None,
        }

    chunks = get_all_chunks_from_db(kb_id=kb_id)
    bm25_index = build_bm25_index(chunks)
    configs = _expand_parameter_grid(parameter_grid)

    results: list[dict[str, Any]] = []

    for idx, config in enumerate(configs):
        metrics_by_case: list[dict[str, float]] = []
        case_results: list[dict[str, Any]] = []
        judge_cases: list[dict[str, Any]] = []

        for case in cases:
            query = str(case.get("query", "")).strip()
            if not query:
                continue

            ranked = _retrieve_ranked(
                index=index,
                query=query,
                config=config,
                kb_id=kb_id,
                chunks=chunks,
                bm25_index=bm25_index,
            )

            judge_cases.append(
                {
                    "case_id": case.get("id"),
                    "query": query,
                    "expected": case,
                    "hits": ranked[:k],
                }
            )

            relevance_flags = [1 if _is_relevant(item, case) else 0 for item in ranked]
            hard_negative_flags = [
                1 if _is_hard_negative(item, case) else 0 for item in ranked
            ]
            metrics = _compute_metrics(
                relevance_flags=relevance_flags,
                hard_negative_flags=hard_negative_flags,
                relevant_total=_count_relevant_targets(case),
                k=k,
            )
            metrics_by_case.append(metrics)

            if include_case_results:
                case_results.append(
                    {
                        "case_id": case.get("id"),
                        "query": query,
                        "metrics": metrics,
                        "top_hits": ranked[:k],
                    }
                )

        valid_cases = len(metrics_by_case)
        if valid_cases == 0:
            averaged = {
                "precision@k": 0.0,
                "recall@k": 0.0,
                "hit_rate@k": 0.0,
                "mrr@k": 0.0,
                "ndcg@k": 0.0,
                "hard_negative_hit_rate@k": 0.0,
                "hard_negative_ratio@k": 0.0,
                "false_positive_ratio@k": 0.0,
                "empty_result_rate@k": 0.0,
                "abstain_success@k": 0.0,
            }
        else:
            averaged = {
                "precision@k": sum(m["precision@k"] for m in metrics_by_case)
                / valid_cases,
                "recall@k": sum(m["recall@k"] for m in metrics_by_case) / valid_cases,
                "hit_rate@k": sum(m["hit_rate@k"] for m in metrics_by_case)
                / valid_cases,
                "mrr@k": sum(m["mrr@k"] for m in metrics_by_case) / valid_cases,
                "ndcg@k": sum(m["ndcg@k"] for m in metrics_by_case) / valid_cases,
                "hard_negative_hit_rate@k": sum(
                    m["hard_negative_hit_rate@k"] for m in metrics_by_case
                )
                / valid_cases,
                "hard_negative_ratio@k": sum(
                    m["hard_negative_ratio@k"] for m in metrics_by_case
                )
                / valid_cases,
                "false_positive_ratio@k": sum(
                    m["false_positive_ratio@k"] for m in metrics_by_case
                )
                / valid_cases,
                "empty_result_rate@k": sum(
                    m["empty_result_rate@k"] for m in metrics_by_case
                )
                / valid_cases,
                "abstain_success@k": sum(
                    m["abstain_success@k"] for m in metrics_by_case
                )
                / valid_cases,
            }

        payload: dict[str, Any] = {
            "config_id": idx,
            "config": config,
            "metrics": averaged,
            "evaluated_cases": valid_cases,
            "_judge_cases": judge_cases,
        }
        if include_case_results:
            payload["cases"] = case_results
        results.append(payload)

    best = max(results, key=_best_config_sort_key) if results else None

    if include_llm_judge and results:
        if llm_judge_on_all_configs:
            for result in results:
                result["llm_judge"] = _evaluate_llm_judge(
                    result.get("_judge_cases", []),
                    sample_size=max(1, llm_judge_sample_size),
                )
            best = max(results, key=_best_config_sort_key)
        elif best is not None:
            best["llm_judge"] = _evaluate_llm_judge(
                best.get("_judge_cases", []),
                sample_size=max(1, llm_judge_sample_size),
            )

    for result in results:
        result.pop("_judge_cases", None)

    best_summary = None
    if best is not None:
        best_summary = {
            "config_id": best["config_id"],
            "config": best["config"],
            "metrics": best["metrics"],
        }
        if "llm_judge" in best:
            best_summary["llm_judge"] = best["llm_judge"]

    return {
        "kb_id": kb_id,
        "k": k,
        "case_count": len(cases),
        "config_count": len(results),
        "results": results,
        "best_config": best_summary,
    }
