#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.app.main import app
from src.backend.app.db import get_conn


OOD_QUESTIONS = [
    "仓颉语言是否定义了虫洞加密协议v9？",
    "仓颉官方是否有海底核聚变引擎调优指南？",
    "仓颉是否支持月球基地生命维持系统驱动？",
    "请列出仓颉超光速通信栈的标准接口。",
    "仓颉有没有黑洞存储后端的官方SDK？",
    "请给出仓颉在平行宇宙同步中的错误码。",
    "请给出仓颉在曲率引擎中的部署流程。",
    "仓颉是否提供反重力推进器控制库？",
    "仓颉文档是否包含虫洞路由算法手册？",
    "请给出仓颉的星际舰队通信协议章节。",
    "仓颉是否支持时间旅行事务回滚？",
    "请给出仓颉暗物质数据库驱动配置。",
    "仓颉有没有量子瞬移 SDK 下载地址？",
    "请列出仓颉在平行宇宙分布式锁规范。",
    "仓颉官方是否发布超光速消息队列指南？",
    "仓颉是否提供火星基地容灾方案模板？",
    "请给出仓颉黑洞容器编排命令。",
    "仓颉有没有反物质能量监控 API？",
    "请列出仓颉跨星系服务发现标准。",
    "仓颉是否支持银河级实时渲染引擎内核？",
    "请给出仓颉反引力数据库主从切换策略。",
    "仓颉是否有时空穿梭日志采集插件？",
    "请列出仓颉星门认证协议字段定义。",
    "仓颉有没有虫洞链路压缩算法文档？",
    "请给出仓颉反物质编译器安装步骤。",
    "仓颉官方是否发布宇宙射线防护 SDK？",
    "请列出仓颉平行时空消息幂等规范。",
    "仓颉是否支持行星级边缘计算调度器？",
    "请给出仓颉量子泡沫网络诊断命令。",
    "仓颉有没有超维对象存储一致性协议？",
]

_GENERIC_TOKENS = {
    "index",
    "package",
    "class",
    "classes",
    "struct",
    "structs",
    "interface",
    "interfaces",
    "function",
    "functions",
    "sample",
    "samples",
    "overview",
    "appendix",
    "readme",
    "claude",
    "manual",
    "docs",
    "doc",
}


def _title_tokens(text: str) -> list[str]:
    normalized = str(text or "").lower()
    tokens = re.findall(r"[a-z][a-z0-9_]{2,}", normalized)
    picked: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) > 20:
            continue
        if re.fullmatch(r"[a-f0-9]{16,}", token):
            continue
        if len(token) >= 16 and sum(ch.isdigit() for ch in token) >= 4:
            continue
        if token in _GENERIC_TOKENS or token in seen:
            continue
        seen.add(token)
        picked.append(token)
    return picked


def _fetch_document_rows(kb_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, source_ref FROM documents WHERE kb_id = ? ORDER BY id",
            (kb_id,),
        ).fetchall()
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "id": int(row["id"]),
                "title": str(row["title"] or ""),
                "source_ref": str(row["source_ref"] or ""),
            }
        )
    return payload


def _build_positive_cases(kb_id: int, max_cases: int) -> list[dict[str, Any]]:
    rows = _fetch_document_rows(kb_id)
    token_to_docs: dict[str, set[int]] = {}
    for row in rows:
        tokens = set(_title_tokens(f"{row['title']} {row['source_ref']}"))
        if not tokens:
            continue
        doc_id = int(row["id"])
        for token in tokens:
            token_to_docs.setdefault(token, set()).add(doc_id)

    candidates: list[str] = []
    for token, doc_ids in token_to_docs.items():
        freq = len(doc_ids)
        if freq > 80:
            continue
        candidates.append(token)
    candidates.sort(
        key=lambda item: (len(token_to_docs.get(item, set())), -len(item), item)
    )

    max_topics = max(0, max_cases // 2)
    selected_topics = candidates[:max_topics]

    cases: list[dict[str, Any]] = []
    index = 1
    for topic in selected_topics:
        expect = [topic]

        cases.append(
            {
                "id": f"p{index:03d}a",
                "question": f"请解释 {topic} 相关章节的核心内容和用途。",
                "allow_nohit": False,
                "expect": expect,
            }
        )
        cases.append(
            {
                "id": f"p{index:03d}b",
                "question": f"{topic} 在仓颉文档中通常包含哪些关键点？",
                "allow_nohit": False,
                "expect": expect,
            }
        )
        index += 1
        if len(cases) >= max_cases:
            break
    return cases[:max_cases]


def _build_negative_cases(max_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, question in enumerate(OOD_QUESTIONS[:max_cases], start=1):
        cases.append(
            {
                "id": f"n{index:03d}",
                "question": question,
                "allow_nohit": True,
                "expect": [],
            }
        )
    return cases


def _evaluate_endpoint(
    *,
    endpoint: str,
    cases: list[dict[str, Any]],
    kb_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []

    with TestClient(app) as client:
        for case in cases:
            started = time.perf_counter()
            response = client.post(endpoint, json={"question": case["question"], "kb_id": kb_id})
            latency = round(time.perf_counter() - started, 3)
            payload = response.json() if response.status_code == 200 else {}
            answer = str(payload.get("answer") or "")
            citations = payload.get("citations") or []
            snippets = " ".join(
                str((item or {}).get("snippet") or "")
                for item in citations
                if isinstance(item, dict)
            )
            sources = " ".join(
                str((item or {}).get("source") or "")
                for item in citations
                if isinstance(item, dict)
            )
            haystack = f"{answer}\n{snippets}\n{sources}".lower()

            if case["allow_nohit"]:
                passed = (
                    len(citations) == 0
                    or "未能检索到相关上下文" in answer
                    or "未找到相关信息" in answer
                )
            else:
                expect_tokens = [str(item).lower() for item in case["expect"]]
                passed = len(citations) > 0 and any(
                    token and token in haystack for token in expect_tokens
                )

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "allow_nohit": case["allow_nohit"],
                    "expect": case["expect"],
                    "status": response.status_code,
                    "pass": bool(passed),
                    "latency_sec": latency,
                    "answer_chars": len(answer),
                    "citations": len(citations),
                    "support_overlap": 0.0,
                    "answer_preview": answer[:280],
                }
            )

    total = len(results)
    failed = sum(1 for item in results if not item["pass"])
    summary = {
        "cases": total,
        "pass_rate": round((total - failed) / total, 4) if total else 0.0,
        "failed": failed,
        "avg_latency_sec": round(
            sum(float(item["latency_sec"]) for item in results) / max(1, total), 3
        ),
        "avg_citations": round(
            sum(int(item["citations"]) for item in results) / max(1, total), 3
        ),
    }
    return summary, results


def _render_markdown(report: dict[str, Any]) -> str:
    rag = report["summary"]["rag"]
    agent = report["summary"]["agent"]
    lines = [
        "# Expanded RAG/Agent Evaluation",
        "",
        f"- Positive cases: {report['meta']['positive_cases']}",
        f"- Negative (OOD) cases: {report['meta']['negative_cases']}",
        f"- Total cases: {report['meta']['total_cases']}",
        "",
        "## Summary",
        "",
        (
            f"- RAG: pass_rate={rag['pass_rate']}, failed={rag['failed']}, "
            f"avg_latency_sec={rag['avg_latency_sec']}, avg_citations={rag['avg_citations']}"
        ),
        (
            f"- AGENT: pass_rate={agent['pass_rate']}, failed={agent['failed']}, "
            f"avg_latency_sec={agent['avg_latency_sec']}, avg_citations={agent['avg_citations']}"
        ),
        "",
    ]

    for mode in ("rag", "agent"):
        failures = [item for item in report["results"][mode] if not item["pass"]]
        lines.append(f"## {mode.upper()} Failures ({len(failures)})")
        lines.append("")
        for item in failures:
            preview = str(item.get("answer_preview") or "").replace("\n", " ")[:220]
            lines.append(
                (
                    f"- {item['id']} | q={item['question']} | cit={item['citations']} | "
                    f"preview={preview}"
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded RAG/Agent evaluation runner")
    parser.add_argument("--kb-id", type=int, default=1)
    parser.add_argument("--positive", type=int, default=120)
    parser.add_argument("--negative", type=int, default=20)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("output/rag_agent_expanded_eval.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("output/rag_agent_expanded_eval.md"),
    )
    args = parser.parse_args()

    positives = _build_positive_cases(args.kb_id, max_cases=max(0, args.positive))
    negatives = _build_negative_cases(max_cases=max(0, args.negative))
    cases = positives + negatives

    rag_summary, rag_results = _evaluate_endpoint(
        endpoint="/chat/rag",
        cases=cases,
        kb_id=args.kb_id,
    )
    agent_summary, agent_results = _evaluate_endpoint(
        endpoint="/chat/agent",
        cases=cases,
        kb_id=args.kb_id,
    )

    report = {
        "meta": {
            "positive_cases": len(positives),
            "negative_cases": len(negatives),
            "total_cases": len(cases),
        },
        "summary": {"rag": rag_summary, "agent": agent_summary},
        "results": {"rag": rag_results, "agent": agent_results},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
