"""FastAPI 应用入口与路由定义。"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil
from typing import Any, Generator, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field

FastApiMCP = None
try:
    _fastapi_mcp_mod = __import__("fastapi_mcp", fromlist=["FastApiMCP"])
    FastApiMCP = getattr(_fastapi_mcp_mod, "FastApiMCP", None)
except Exception:
    FastApiMCP = None

from .config import settings
from .db import (
    init_db,
    insert_document,
    insert_chunks,
    replace_document_chunks,
    list_documents,
    get_documents_by_ids,
    delete_document,
    batch_delete_documents,
    create_chat,
    add_message,
    list_chats,
    get_chat,
    add_agent_step,
    document_title_exists,
    list_knowledge_bases,
    create_knowledge_base,
    delete_knowledge_base,
    ensure_knowledge_base_exists,
)
from .ingest import (
    build_documents_from_file,
    build_documents_from_url,
    split_into_chunks,
)
from .indexer import load_or_create_index, insert_nodes
from .rag import (
    query_rag,
    query_rag_stream_with_citations,
    retrieve_context,
    invalidate_rag_cache,
)
from .agent import run_agent, run_agent_stream_with_metadata
from .retrieval_eval import (
    build_default_tuning_parameter_grid,
    generate_retrieval_benchmark_dataset,
    load_default_benchmark_dataset,
    run_retrieval_evaluation,
)

indexes: dict[int, VectorStoreIndex] = {}


def _kb_index_dir(kb_id: int) -> Path:
    return Path(settings.index_dir) / f"kb_{kb_id}"


def _get_or_create_kb_index(kb_id: int) -> Optional[VectorStoreIndex]:
    if settings.mock_mode:
        return None
    if kb_id in indexes:
        return indexes[kb_id]
    index = load_or_create_index(_kb_index_dir(kb_id))
    indexes[kb_id] = index
    return index


def _drop_kb_index(kb_id: int) -> None:
    indexes.pop(kb_id, None)
    index_dir = _kb_index_dir(kb_id)
    if index_dir.exists():
        shutil.rmtree(index_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化本地存储、数据库与向量索引。"""
    indexes.clear()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    if not settings.mock_mode:
        _get_or_create_kb_index(1)
    yield


app = FastAPI(title="Local Knowledge AI Assistant", lifespan=lifespan)

if FastApiMCP is not None:
    mcp_server = FastApiMCP(app, name="ima-simple-mcp")
    mcp_server.mount_http()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1024)


class UrlIngestRequest(BaseModel):
    """URL 入库请求负载。"""

    url: str
    kb_id: int = 1


class ChatRequest(BaseModel):
    """对话请求负载。"""

    question: str
    chat_id: Optional[int] = None
    kb_id: int = 1
    stream: bool = False


class KnowledgeBaseCreateRequest(BaseModel):
    name: str
    description: str = ""


class BatchDocumentDeleteRequest(BaseModel):
    kb_id: int = 1
    document_ids: list[int] = Field(default_factory=list)


class BatchDocumentReindexRequest(BaseModel):
    kb_id: int = 1
    document_ids: list[int] = Field(default_factory=list)


class RetrievalEvalCase(BaseModel):
    id: Optional[str] = None
    query: str
    relevant_ids: list[int] = Field(default_factory=list)
    relevant_sources: list[str] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    relevant_snippets: list[str] = Field(default_factory=list)


class RetrievalEvalRequest(BaseModel):
    cases: Optional[list[RetrievalEvalCase]] = None
    parameter_grid: Optional[dict[str, list[Any]]] = None
    kb_id: int = 1
    k: int = 5
    auto_tune: bool = False
    include_case_results: bool = False
    include_llm_judge: bool = False
    llm_judge_sample_size: int = 10
    llm_judge_on_all_configs: bool = False


class RetrievalDatasetGenerateRequest(BaseModel):
    kb_id: int = 1
    case_count: int = 20
    use_llm: bool = True


class RetrievalRequest(BaseModel):
    question: str
    kb_id: int = 1
    chat_id: Optional[int] = None
    top_k: int = 6


@app.get("/health")
def health() -> dict[str, Any]:
    """健康检查接口。"""
    return {"status": "ok", "mock_mode": settings.mock_mode}


def _build_nodes_for_document_record(document: dict[str, Any]):
    source_type = str(document.get("source_type") or "")
    source_ref = str(document.get("source_ref") or "")

    if source_type == "url":
        docs = build_documents_from_url(source_ref)
    else:
        source_path = Path(source_ref)
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        docs = build_documents_from_file(source_path, str(source_path))

    nodes = split_into_chunks(docs)
    payload = [
        {
            "content": node.get_content(),
            "page": node.metadata.get("page"),
            "start_offset": None,
            "end_offset": None,
        }
        for node in nodes
    ]
    return nodes, payload


def _reindex_documents(kb_id: int, documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        return {"reindexed": 0, "failed": []}

    _drop_kb_index(kb_id)
    kb_index = _get_or_create_kb_index(kb_id)

    reindexed = 0
    failed: list[dict[str, Any]] = []
    for document in documents:
        doc_id = int(document["id"])
        try:
            nodes, payload = _build_nodes_for_document_record(document)
            replace_document_chunks(doc_id, payload)
            if not settings.mock_mode and kb_index is not None:
                insert_nodes(kb_index, nodes, index_dir=_kb_index_dir(kb_id))
            reindexed += 1
        except Exception as exc:
            failed.append({"document_id": doc_id, "error": str(exc)})

    invalidate_rag_cache(kb_id=kb_id)
    return {"reindexed": reindexed, "failed": failed}


@app.get("/kb")
def get_knowledge_base_list() -> list[dict[str, Any]]:
    return list_knowledge_bases()


@app.post("/kb")
def create_kb(req: KnowledgeBaseCreateRequest) -> dict[str, Any]:
    try:
        created = create_knowledge_base(req.name, req.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return created


@app.delete("/kb/{kb_id}")
def remove_knowledge_base(kb_id: int) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        delete_knowledge_base(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _drop_kb_index(kb_id)
    invalidate_rag_cache(kb_id=kb_id)
    return {"deleted": kb_id}


@app.post("/kb/{kb_id}/reindex")
def reindex_knowledge_base(kb_id: int) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = list_documents(kb_id=kb_id)
    result = _reindex_documents(kb_id, documents)
    return {"kb_id": kb_id, **result}


@app.post("/kb/documents/batch-delete")
def remove_documents_batch(req: BatchDocumentDeleteRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    deleted = batch_delete_documents(req.document_ids, req.kb_id)
    _drop_kb_index(req.kb_id)
    invalidate_rag_cache(kb_id=req.kb_id)
    return {"kb_id": req.kb_id, "deleted": deleted}


@app.post("/kb/documents/reindex-batch")
def reindex_documents_batch(req: BatchDocumentReindexRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = get_documents_by_ids(req.document_ids, req.kb_id)
    result = _reindex_documents(req.kb_id, documents)
    return {"kb_id": req.kb_id, **result}


@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    allow_duplicate: bool = Query(False),
    kb_id: int = Query(1, ge=1),
) -> dict[str, Any]:
    """将上传的本地文件入库。

    Args:
        file: 上传的文档文件。

    Returns:
        文档 id 与分块数量。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    normalized_name = file.filename.replace("\\", "/").lstrip("/")
    if not normalized_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not allow_duplicate and document_title_exists(normalized_name, kb_id=kb_id):
        raise HTTPException(
            status_code=409,
            detail="Duplicate filename detected. Choose keep mode to upload anyway.",
        )

    ext = Path(normalized_name).suffix.lower()
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".xlsx",
        ".xls",
        ".csv",
        ".pptx",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".json",
        ".xml",
        ".rtf",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".sh",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
    }

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    upload_dir = Path(settings.data_dir) / "uploads" / f"kb_{kb_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / normalized_name
    try:
        dest.resolve().relative_to(upload_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid filename path") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)

    doc_id = insert_document(normalized_name, "file", str(dest), kb_id=kb_id)
    docs = build_documents_from_file(dest, str(dest))
    nodes = split_into_chunks(docs)

    kb_index = _get_or_create_kb_index(kb_id)
    if not settings.mock_mode and kb_index is not None:
        insert_nodes(kb_index, nodes, index_dir=_kb_index_dir(kb_id))

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

    invalidate_rag_cache(kb_id=kb_id)

    return {"document_id": doc_id, "chunks": len(nodes), "kb_id": kb_id}


@app.post("/ingest/url")
def ingest_url(req: UrlIngestRequest) -> dict[str, Any]:
    """将 URL 内容入库。

    Args:
        req: URL 入库请求。

    Returns:
        文档 id 与分块数量。
    """
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    doc_id = insert_document(req.url, "url", req.url, kb_id=req.kb_id)
    docs = build_documents_from_url(req.url)
    nodes = split_into_chunks(docs)

    kb_index = _get_or_create_kb_index(req.kb_id)
    if not settings.mock_mode and kb_index is not None:
        insert_nodes(kb_index, nodes, index_dir=_kb_index_dir(req.kb_id))

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

    invalidate_rag_cache(kb_id=req.kb_id)

    return {"document_id": doc_id, "chunks": len(nodes), "kb_id": req.kb_id}


@app.get("/kb/documents")
def get_documents(kb_id: int = Query(1, ge=1)) -> list[dict[str, Any]]:
    """列出所有已索引的文档。"""
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return list_documents(kb_id=kb_id)


@app.get("/kb/preview")
def get_document_preview(
    source: str = Query(..., min_length=1), kb_id: int = Query(1, ge=1)
) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    normalized_source = source.strip()
    if not normalized_source:
        raise HTTPException(status_code=400, detail="Missing source")

    if normalized_source == "local":
        raise HTTPException(
            status_code=400,
            detail="Source 'local' is a derived citation and has no direct file preview.",
        )

    parsed = urlparse(normalized_source)
    if parsed.scheme in {"http", "https"}:
        docs = build_documents_from_url(normalized_source)
        content = "\n\n".join(doc.text for doc in docs if doc.text)
        return {
            "source": normalized_source,
            "kind": "url",
            "preview_type": "text",
            "content": content,
        }

    upload_root = (Path(settings.data_dir) / "uploads" / f"kb_{kb_id}").resolve()
    source_path = Path(normalized_source.replace("\\", "/"))
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    else:
        source_path = source_path.resolve()

    try:
        source_path.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403, detail="Preview path is outside upload root"
        ) from exc

    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="Preview source file not found")

    docs = build_documents_from_file(source_path, str(source_path))
    content = "\n\n".join(doc.text for doc in docs if doc.text)
    ext = source_path.suffix.lower().lstrip(".")
    preview_type = "markdown" if ext in {"md", "markdown"} else "text"
    if ext in {"json", "xml", "yaml", "yml", "toml", "ini", "cfg"}:
        preview_type = "code"

    return {
        "source": normalized_source,
        "kind": "file",
        "preview_type": preview_type,
        "content": content,
    }


@app.delete("/kb/documents/{doc_id}")
def remove_document(doc_id: int, kb_id: int = Query(1, ge=1)) -> dict[str, Any]:
    """删除文档及其分块。

    Args:
        doc_id: 要删除的文档 id。

    Returns:
        删除确认负载。
    """
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    delete_document(doc_id, kb_id=kb_id)
    _drop_kb_index(kb_id)
    invalidate_rag_cache(kb_id=kb_id)
    return {"deleted": doc_id, "kb_id": kb_id}


@app.post("/kb/documents/{doc_id}/reindex")
def reindex_document(doc_id: int, kb_id: int = Query(1, ge=1)) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    docs = get_documents_by_ids([doc_id], kb_id)
    if not docs:
        raise HTTPException(status_code=404, detail="Document not found")
    result = _reindex_documents(kb_id, docs)
    return {"kb_id": kb_id, **result}


@app.post("/chat/rag")
def chat_rag(req: ChatRequest) -> Any:
    """执行 RAG 对话并持久化记录。

    Args:
        req: 对话请求负载。

    Returns:
        含引用的对话响应。
    """
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

    kb_index = _get_or_create_kb_index(req.kb_id)

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


@app.post("/retrieve")
def retrieve(req: RetrievalRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb_index = _get_or_create_kb_index(req.kb_id)
    payload = retrieve_context(
        kb_index,
        req.question,
        kb_id=req.kb_id,
        chat_id=req.chat_id,
        top_k=max(1, req.top_k),
    )
    return {"kb_id": req.kb_id, **payload}


@app.post("/chat/agent")
def chat_agent(req: ChatRequest) -> Any:
    """执行 Agent 流程并持久化记录。

    Args:
        req: 对话请求负载。

    Returns:
        含步骤与引用的对话响应。
    """
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

    kb_index = _get_or_create_kb_index(req.kb_id)

    if req.stream:
        stream, metadata = run_agent_stream_with_metadata(
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


@app.get("/chat/history")
def chat_history(kb_id: int = Query(1, ge=1)) -> list[dict[str, Any]]:
    """列出对话会话列表。"""
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return list_chats(kb_id=kb_id)


@app.get("/chat/{chat_id}")
def chat_detail(chat_id: int, kb_id: int = Query(1, ge=1)) -> dict[str, Any]:
    """获取包含消息与 Agent 步骤的对话详情。

    Args:
        chat_id: 要获取的对话 id。

    Returns:
        对话详情负载。
    """
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    chat = get_chat(chat_id, kb_id=kb_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.post("/eval/retrieval")
def evaluate_retrieval(req: RetrievalEvalRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    dataset_source = "request"
    cases: list[dict[str, Any]]
    parameter_grid = req.parameter_grid
    k = req.k

    if req.cases is None:
        dataset_source = "default"
        dataset = load_default_benchmark_dataset()
        raw_cases = dataset.get("cases", [])
        if not isinstance(raw_cases, list) or not raw_cases:
            raise HTTPException(
                status_code=400, detail="Default benchmark dataset is empty"
            )
        cases = [dict(case) for case in raw_cases]
        if parameter_grid is None and isinstance(dataset.get("parameter_grid"), dict):
            parameter_grid = dataset["parameter_grid"]
        if parameter_grid is None and req.auto_tune:
            parameter_grid = build_default_tuning_parameter_grid()
        k = int(dataset.get("k", req.k))
    else:
        if not req.cases:
            raise HTTPException(status_code=400, detail="At least one case is required")
        cases = [case.model_dump() for case in req.cases]
        if parameter_grid is None and req.auto_tune:
            parameter_grid = build_default_tuning_parameter_grid()

    kb_index = _get_or_create_kb_index(req.kb_id)
    result = run_retrieval_evaluation(
        index=kb_index,
        cases=cases,
        parameter_grid=parameter_grid,
        kb_id=req.kb_id,
        k=max(1, k),
        include_case_results=req.include_case_results,
        include_llm_judge=req.include_llm_judge,
        llm_judge_sample_size=max(1, req.llm_judge_sample_size),
        llm_judge_on_all_configs=req.llm_judge_on_all_configs,
    )
    result["dataset"] = dataset_source
    return result


@app.post("/eval/retrieval/generate-dataset")
def generate_retrieval_dataset(req: RetrievalDatasetGenerateRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return generate_retrieval_benchmark_dataset(
        kb_id=req.kb_id,
        count=max(1, req.case_count),
        use_llm=req.use_llm,
    )
