from __future__ import annotations

import json
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
        normalized = [_normalize_node(item) for item in ranked]
    else:
        normalized = [
            _normalize_node(item) for item in search_chunks(query, limit=k, kb_id=kb_id)
        ]

    use_reranker = bool(config.get("use_reranker", False))
    rerank_top_k = max(1, int(config.get("rerank_top_k", k)))
    if use_reranker and normalized:
        docs = [
            {"content": item.get("content", ""), "item": item} for item in normalized
        ]
        reranked = rerank_documents(query, docs, top_k=rerank_top_k)
        normalized = [doc["item"] for doc in reranked if "item" in doc]

    return normalized[:final_top_k]


def _count_relevant_targets(case: dict[str, Any]) -> int:
    if case.get("relevant_ids"):
        return len(set(case["relevant_ids"]))
    if case.get("relevant_snippets"):
        return len(case["relevant_snippets"])
    if case.get("relevant_sources"):
        return len(set(case["relevant_sources"]))
    if case.get("relevant_pages"):
        return len(set(case["relevant_pages"]))
    return 0


def _is_relevant(item: dict[str, Any], case: dict[str, Any]) -> bool:
    if case.get("relevant_ids") and item.get("id") in set(case["relevant_ids"]):
        return True
    if case.get("relevant_sources") and item.get("source") in set(
        case["relevant_sources"]
    ):
        return True
    if case.get("relevant_pages") and item.get("page") in set(case["relevant_pages"]):
        return True

    content = str(item.get("content") or "").lower()
    snippets = [str(s).lower() for s in case.get("relevant_snippets", [])]
    return any(snippet and snippet in content for snippet in snippets)


def _compute_metrics(
    relevance_flags: list[int], relevant_total: int, k: int
) -> dict[str, float]:
    top = relevance_flags[:k]
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

    return {
        "precision@k": precision,
        "recall@k": recall,
        "hit_rate@k": hit_rate,
        "mrr@k": reciprocal_rank,
        "ndcg@k": ndcg,
    }


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
        "Return JSON only with keys: relevance, coverage, groundedness, reason.\n"
        "- relevance: how well retrieved chunks match user intent (0-1).\n"
        "- coverage: whether key evidence likely exists in retrieved chunks (0-1).\n"
        "- groundedness: internal consistency/non-noise quality of retrieved evidence (0-1).\n"
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
            "reason": "judge_parse_failed",
        }
    return {
        "relevance": _normalize_unit_score(parsed.get("relevance")),
        "coverage": _normalize_unit_score(parsed.get("coverage")),
        "groundedness": _normalize_unit_score(parsed.get("groundedness")),
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

    for case in selected_cases:
        judged = _judge_single_case(
            query=str(case.get("query") or ""),
            hits=case.get("hits", []),
            expected=case.get("expected", {}),
        )
        sum_relevance += judged["relevance"]
        sum_coverage += judged["coverage"]
        sum_groundedness += judged["groundedness"]
        judged_cases.append(
            {
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "scores": {
                    "relevance": judged["relevance"],
                    "coverage": judged["coverage"],
                    "groundedness": judged["groundedness"],
                    "overall": (
                        judged["relevance"]
                        + judged["coverage"]
                        + judged["groundedness"]
                    )
                    / 3.0,
                },
                "reason": judged["reason"],
            }
        )

    count = len(judged_cases)
    if count == 0:
        avg_relevance = 0.0
        avg_coverage = 0.0
        avg_groundedness = 0.0
    else:
        avg_relevance = sum_relevance / count
        avg_coverage = sum_coverage / count
        avg_groundedness = sum_groundedness / count

    return {
        "enabled": True,
        "sample_size": count,
        "scores": {
            "relevance": avg_relevance,
            "coverage": avg_coverage,
            "groundedness": avg_groundedness,
            "overall": (avg_relevance + avg_coverage + avg_groundedness) / 3.0,
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
    for idx, chunk in enumerate(sampled):
        content = str(chunk.get("content") or "")
        generated = _generate_case_with_llm(content) if use_llm else None
        if generated is None:
            generated = {
                "query": _derive_query_from_chunk(content),
                "relevant_snippets": [content[:36]],
            }
        cases.append(
            {
                "id": f"kb{kb_id}-case-{idx + 1}",
                "query": generated["query"],
                "relevant_snippets": generated["relevant_snippets"],
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
            metrics = _compute_metrics(
                relevance_flags=relevance_flags,
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

    best = None
    if results:
        best = max(
            results,
            key=lambda r: (
                r["metrics"]["ndcg@k"],
                r["metrics"]["mrr@k"],
                r["metrics"]["hit_rate@k"],
            ),
        )

    if include_llm_judge and results:
        if llm_judge_on_all_configs:
            for result in results:
                result["llm_judge"] = _evaluate_llm_judge(
                    result.get("_judge_cases", []),
                    sample_size=max(1, llm_judge_sample_size),
                )
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
