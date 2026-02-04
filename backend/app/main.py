"""FastAPI 应用入口与路由定义。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .db import (
    init_db,
    insert_document,
    insert_chunks,
    list_documents,
    delete_document,
    create_chat,
    add_message,
    list_chats,
    get_chat,
    add_agent_step,
)
from .ingest import build_documents_from_file, build_documents_from_url, split_into_chunks
from .indexer import load_or_create_index, insert_nodes
from .rag import query_rag
from .agent import run_agent

index: Optional[object] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化本地存储、数据库与向量索引。"""
    global index
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    if settings.mock_mode:
        index = None
    else:
        index = load_or_create_index()
    yield


app = FastAPI(title="Local Knowledge AI Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlIngestRequest(BaseModel):
    """URL 入库请求负载。"""

    url: str


class ChatRequest(BaseModel):
    """对话请求负载。"""

    question: str
    chat_id: Optional[int] = None


@app.get("/health")
def health() -> dict[str, Any]:
    """健康检查接口。"""
    return {"status": "ok", "mock_mode": settings.mock_mode}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """将上传的本地文件入库。

    Args:
        file: 上传的文档文件。

    Returns:
        文档 id 与分块数量。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    upload_dir = Path(settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    doc_id = insert_document(file.filename, "file", str(dest))
    docs = build_documents_from_file(dest, str(dest))
    nodes = split_into_chunks(docs)

    if not settings.mock_mode and index is not None:
        insert_nodes(index, nodes)

    insert_chunks(
        doc_id,
        [
            {
                "content": node.get_content(),
                "page": node.metadata.get("page"),
                "start_offset": None,
                "end_offset": None,
            }
            for node in nodes
        ],
    )

    return {"document_id": doc_id, "chunks": len(nodes)}


@app.post("/ingest/url")
def ingest_url(req: UrlIngestRequest) -> dict[str, Any]:
    """将 URL 内容入库。

    Args:
        req: URL 入库请求。

    Returns:
        文档 id 与分块数量。
    """
    doc_id = insert_document(req.url, "url", req.url)
    docs = build_documents_from_url(req.url)
    nodes = split_into_chunks(docs)

    if not settings.mock_mode and index is not None:
        insert_nodes(index, nodes)

    insert_chunks(
        doc_id,
        [
            {
                "content": node.get_content(),
                "page": None,
                "start_offset": None,
                "end_offset": None,
            }
            for node in nodes
        ],
    )
    return {"document_id": doc_id, "chunks": len(nodes)}


@app.get("/kb/documents")
def get_documents() -> list[dict[str, Any]]:
    """列出所有已索引的文档。"""
    return list_documents()


@app.delete("/kb/documents/{doc_id}")
def remove_document(doc_id: int) -> dict[str, Any]:
    """删除文档及其分块。

    Args:
        doc_id: 要删除的文档 id。

    Returns:
        删除确认负载。
    """
    delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/chat/rag")
def chat_rag(req: ChatRequest) -> dict[str, Any]:
    """执行 RAG 对话并持久化记录。

    Args:
        req: 对话请求负载。

    Returns:
        含引用的对话响应。
    """
    chat_id = req.chat_id or create_chat()
    add_message(chat_id, "user", req.question)

    result = query_rag(index, req.question)
    add_message(chat_id, "assistant", result["answer"])

    return {"chat_id": chat_id, **result}


@app.post("/chat/agent")
def chat_agent(req: ChatRequest) -> dict[str, Any]:
    """执行 Agent 流程并持久化记录。

    Args:
        req: 对话请求负载。

    Returns:
        含步骤与引用的对话响应。
    """
    chat_id = req.chat_id or create_chat()
    add_message(chat_id, "user", req.question)

    result = run_agent(index, req.question)
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

    return {"chat_id": chat_id, **result}


@app.get("/chat/history")
def chat_history() -> list[dict[str, Any]]:
    """列出对话会话列表。"""
    return list_chats()


@app.get("/chat/{chat_id}")
def chat_detail(chat_id: int) -> dict[str, Any]:
    """获取包含消息与 Agent 步骤的对话详情。

    Args:
        chat_id: 要获取的对话 id。

    Returns:
        对话详情负载。
    """
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
