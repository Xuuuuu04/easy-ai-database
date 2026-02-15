from __future__ import annotations

import json
from typing import Any, Generator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..agent import run_agent, run_agent_stream_with_metadata
from ..db import (
    add_agent_step,
    add_message,
    create_chat,
    ensure_knowledge_base_exists,
    get_chat,
    list_chats,
)
from ..rag import query_rag, query_rag_stream_with_citations, retrieve_context
from ..schemas import ChatRequest, RetrievalRequest
from ..state import get_or_create_kb_index

router = APIRouter()


@router.post("/chat/rag")
def chat_rag(req: ChatRequest) -> Any:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if req.chat_id is not None:
        existing_chat = get_chat(req.chat_id, kb_id=req.kb_id)
        if not existing_chat:
            raise HTTPException(
                status_code=404, detail="Chat not found for knowledge base"
            )
    chat_id = req.chat_id or create_chat(kb_id=req.kb_id)
    add_message(chat_id, "user", req.question)

    kb_index = get_or_create_kb_index(req.kb_id)

    if req.stream:
        stream, citations = query_rag_stream_with_citations(
            kb_index,
            req.question,
            chat_id=chat_id,
            kb_id=req.kb_id,
        )

        def generate() -> Generator[str, None, None]:
            full_answer = ""
            for chunk in stream:
                if not chunk:
                    continue
                full_answer += chunk
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"

            add_message(chat_id, "assistant", full_answer)
            done_payload = {
                "done": True,
                "chat_id": chat_id,
                "citations": citations,
                "answer": full_answer,
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = query_rag(kb_index, req.question, chat_id=chat_id, kb_id=req.kb_id)
    add_message(chat_id, "assistant", result["answer"])

    return {"chat_id": chat_id, "kb_id": req.kb_id, **result}


@router.post("/retrieve")
def retrieve(req: RetrievalRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb_index = get_or_create_kb_index(req.kb_id)
    payload = retrieve_context(
        kb_index,
        req.question,
        kb_id=req.kb_id,
        chat_id=req.chat_id,
        top_k=max(1, req.top_k),
    )
    return {"kb_id": req.kb_id, **payload}


@router.post("/chat/agent")
def chat_agent(req: ChatRequest) -> Any:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if req.chat_id is not None:
        existing_chat = get_chat(req.chat_id, kb_id=req.kb_id)
        if not existing_chat:
            raise HTTPException(
                status_code=404, detail="Chat not found for knowledge base"
            )
    chat_id = req.chat_id or create_chat(kb_id=req.kb_id)
    add_message(chat_id, "user", req.question)

    kb_index = get_or_create_kb_index(req.kb_id)

    if req.stream:
        stream, metadata = run_agent_stream_with_metadata(
            kb_index,
            req.question,
            chat_id=chat_id,
            kb_id=req.kb_id,
        )

        def generate() -> Generator[str, None, None]:
            full_answer = ""
            research_steps = metadata.get("steps", [])
            research_citations = metadata.get("citations", [])

            start_payload = {
                "event": "agent_research_start",
                "mode": "agent",
                "research_mode": bool(metadata.get("research_mode", True)),
                "steps_total": len(research_steps),
                "citations": research_citations,
            }
            yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\n\n"

            streamed_steps: list[dict[str, Any]] = []
            for step_index, step in enumerate(research_steps):
                streamed_steps.append(step)
                step_payload = {
                    "event": "agent_step",
                    "mode": "agent",
                    "step_index": step_index,
                    "step": step,
                    "steps": list(streamed_steps),
                    "citations": research_citations,
                }
                yield f"data: {json.dumps(step_payload, ensure_ascii=False)}\n\n"

            summary_start_payload = {
                "event": "agent_summary_start",
                "mode": "agent",
                "steps": list(streamed_steps),
                "citations": research_citations,
            }
            yield f"data: {json.dumps(summary_start_payload, ensure_ascii=False)}\n\n"

            for chunk in stream:
                if not chunk:
                    continue
                full_answer += chunk
                yield f"data: {json.dumps({'event': 'answer_chunk', 'mode': 'agent', 'chunk': chunk}, ensure_ascii=False)}\n\n"

            answer = metadata.get("answer") or full_answer
            steps = metadata.get("steps", [])
            citations = metadata.get("citations", [])

            add_message(chat_id, "assistant", answer)

            for i, step in enumerate(steps):
                add_agent_step(
                    chat_id,
                    i,
                    step["tool"],
                    str(step["input"]),
                    str(step["output"]),
                    str(step.get("citations")),
                )

            done_payload = {
                "event": "agent_done",
                "mode": "agent",
                "done": True,
                "chat_id": chat_id,
                "citations": citations,
                "steps": steps,
                "answer": answer,
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = run_agent(kb_index, req.question, chat_id=chat_id, kb_id=req.kb_id)
    add_message(chat_id, "assistant", result["answer"])

    for i, step in enumerate(result["steps"]):
        add_agent_step(
            chat_id,
            i,
            step["tool"],
            str(step["input"]),
            str(step["output"]),
            str(step.get("citations")),
        )

    return {"chat_id": chat_id, "kb_id": req.kb_id, **result}


@router.get("/chat/history")
def chat_history(kb_id: int = Query(1, ge=1)) -> list[dict[str, Any]]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return list_chats(kb_id=kb_id)


@router.get("/chat/{chat_id}")
def chat_detail(chat_id: int, kb_id: int = Query(1, ge=1)) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    chat = get_chat(chat_id, kb_id=kb_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
