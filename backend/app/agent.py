"""Agent 编排与工具执行流程。"""

from __future__ import annotations

from typing import Any, TypedDict, Optional

from langgraph.graph import StateGraph, END

from .config import settings
from .indexer import get_llm
from .rag import query_rag
from .ingest import extract_text_from_url
from llama_index.core import VectorStoreIndex


class AgentState(TypedDict):
    """在 Agent 图节点之间传递的状态容器。"""

    question: str
    context: str
    steps: list[dict[str, Any]]
    citations: list[dict[str, Any]]


def _tool_search_kb(state: AgentState, index: Optional[VectorStoreIndex]) -> AgentState:
    """搜索知识库并记录一次工具调用步骤。

    Args:
        state: 当前 Agent 状态。
        index: 用于检索的向量索引。

    Returns:
        更新后的状态，包含上下文、引用与步骤记录。
    """
    result = query_rag(index, state["question"])
    state["context"] = result["answer"]
    state["citations"] = result["citations"]
    state["steps"].append(
        {
            "tool": "search_kb",
            "input": state["question"],
            "output": result["answer"],
            "citations": result["citations"],
        }
    )
    return state


def _tool_fetch_url(state: AgentState) -> AgentState:
    """抓取问题中第一个 URL 的内容并追加到上下文。

    Args:
        state: 当前 Agent 状态。

    Returns:
        更新后的状态，包含抓取内容与步骤信息。
    """
    urls = [token for token in state["question"].split() if token.startswith("http")]
    if not urls:
        return state
    url = urls[0]
    text = extract_text_from_url(url)
    state["context"] += "\n\n" + text[:2000]
    state["steps"].append(
        {
            "tool": "fetch_url",
            "input": url,
            "output": text[:500],
            "citations": [{"source": url, "page": None, "snippet": text[:200]}],
        }
    )
    return state


def _tool_summarize(state: AgentState) -> AgentState:
    """将当前上下文汇总为最终回答步骤。

    Args:
        state: 当前 Agent 状态。

    Returns:
        更新后的状态，包含摘要输出。
    """
    if settings.mock_mode:
        answer = state["context"][:400]
        state["steps"].append(
            {
                "tool": "summarize",
                "input": state["question"],
                "output": answer,
                "citations": state.get("citations", []),
            }
        )
        return state

    llm = get_llm()
    prompt = (
        "请基于以下上下文回答问题，必须引用来源：\n"
        f"问题：{state['question']}\n\n"
        f"上下文：\n{state['context']}\n"
    )
    response = llm.complete(prompt)
    state["steps"].append(
        {
            "tool": "summarize",
            "input": state["question"],
            "output": response.text,
            "citations": state.get("citations", []),
        }
    )
    return state


def build_agent_graph(index: Optional[VectorStoreIndex]):
    """为当前索引构建并编译 Agent 图。

    Args:
        index: 用于检索的向量索引。

    Returns:
        可直接调用的 LangGraph 应用。
    """
    graph = StateGraph(AgentState)

    graph.add_node("search_kb", lambda state: _tool_search_kb(state, index))
    graph.add_node("fetch_url", _tool_fetch_url)
    graph.add_node("summarize", _tool_summarize)

    graph.set_entry_point("search_kb")
    graph.add_edge("search_kb", "fetch_url")
    graph.add_edge("fetch_url", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def run_agent(index: Optional[VectorStoreIndex], question: str) -> dict[str, Any]:
    """运行 Agent 流程并返回最终响应。

    Args:
        index: 用于检索的向量索引。
        question: 用户问题。

    Returns:
        包含回答、步骤与引用的响应负载。
    """
    app = build_agent_graph(index)
    state: AgentState = {"question": question, "context": "", "steps": [], "citations": []}
    result = app.invoke(state)
    final_answer = ""
    if result["steps"]:
        final_answer = result["steps"][-1]["output"]
    return {
        "answer": final_answer,
        "steps": result["steps"],
        "citations": result.get("citations", []),
    }
