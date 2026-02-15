from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from .. import config
from ..db import (
    batch_delete_documents,
    create_knowledge_base,
    delete_document,
    delete_knowledge_base,
    document_title_exists,
    ensure_knowledge_base_exists,
    get_documents_by_ids,
    insert_chunks,
    insert_document,
    list_documents,
    list_knowledge_bases,
)
from ..indexer import insert_nodes
from ..ingest import (
    build_documents_from_file,
    build_documents_from_url,
    split_into_chunks,
)
from ..rag import invalidate_rag_cache
from ..schemas import (
    BatchDocumentDeleteRequest,
    BatchDocumentReindexRequest,
    KnowledgeBaseCreateRequest,
    UrlIngestRequest,
)
from ..source_access import resolve_allowed_source_path
from ..state import (
    drop_kb_index,
    get_or_create_kb_index,
    kb_index_dir,
    reindex_documents,
)

router = APIRouter()


@router.get("/kb")
def get_knowledge_base_list() -> list[dict[str, Any]]:
    return list_knowledge_bases()


@router.post("/kb")
def create_kb(req: KnowledgeBaseCreateRequest) -> dict[str, Any]:
    try:
        created = create_knowledge_base(req.name, req.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return created


@router.delete("/kb/{kb_id}")
def remove_knowledge_base(kb_id: int) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        delete_knowledge_base(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    drop_kb_index(kb_id)
    invalidate_rag_cache(kb_id=kb_id)
    return {"deleted": kb_id}


@router.post("/kb/{kb_id}/reindex")
def reindex_knowledge_base(kb_id: int) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = list_documents(kb_id=kb_id)
    result = reindex_documents(kb_id, documents)
    return {"kb_id": kb_id, **result}


@router.post("/kb/documents/batch-delete")
def remove_documents_batch(req: BatchDocumentDeleteRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    deleted = batch_delete_documents(req.document_ids, req.kb_id)
    drop_kb_index(req.kb_id)
    invalidate_rag_cache(kb_id=req.kb_id)
    return {"kb_id": req.kb_id, "deleted": deleted}


@router.post("/kb/documents/reindex-batch")
def reindex_documents_batch(req: BatchDocumentReindexRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    documents = get_documents_by_ids(req.document_ids, req.kb_id)
    result = reindex_documents(req.kb_id, documents)
    return {"kb_id": req.kb_id, **result}


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    allow_duplicate: bool = Query(False),
    kb_id: int = Query(1, ge=1),
) -> dict[str, Any]:
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
    supported_extensions = {
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

    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported types: {', '.join(sorted(supported_extensions))}",
        )

    upload_dir = Path(config.settings.data_dir) / "uploads" / f"kb_{kb_id}"
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

    kb_index = get_or_create_kb_index(kb_id)
    if not config.settings.mock_mode and kb_index is not None:
        insert_nodes(kb_index, nodes, index_dir=kb_index_dir(kb_id))

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


@router.post("/ingest/url")
def ingest_url(req: UrlIngestRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    doc_id = insert_document(req.url, "url", req.url, kb_id=req.kb_id)
    docs = build_documents_from_url(req.url)
    nodes = split_into_chunks(docs)

    kb_index = get_or_create_kb_index(req.kb_id)
    if not config.settings.mock_mode and kb_index is not None:
        insert_nodes(kb_index, nodes, index_dir=kb_index_dir(req.kb_id))

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


@router.get("/kb/documents")
def get_documents(kb_id: int = Query(1, ge=1)) -> list[dict[str, Any]]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return list_documents(kb_id=kb_id)


@router.get("/kb/preview")
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

    try:
        source_path = resolve_allowed_source_path(normalized_source, kb_id)
    except PermissionError as exc:
        raise HTTPException(
            status_code=403, detail="Preview path is outside upload root"
        ) from exc
    except FileNotFoundError:
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


@router.delete("/kb/documents/{doc_id}")
def remove_document(doc_id: int, kb_id: int = Query(1, ge=1)) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    delete_document(doc_id, kb_id=kb_id)
    drop_kb_index(kb_id)
    invalidate_rag_cache(kb_id=kb_id)
    return {"deleted": doc_id, "kb_id": kb_id}


@router.post("/kb/documents/{doc_id}/reindex")
def reindex_document(doc_id: int, kb_id: int = Query(1, ge=1)) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    docs = get_documents_by_ids([doc_id], kb_id)
    if not docs:
        raise HTTPException(status_code=404, detail="Document not found")
    result = reindex_documents(kb_id, docs)
    return {"kb_id": kb_id, **result}
