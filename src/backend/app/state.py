from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Optional

from fastapi import Request
from llama_index.core import VectorStoreIndex

from . import config
from .db import replace_document_chunks
from .indexer import insert_nodes, load_or_create_index
from .ingest import (
    build_documents_from_file,
    build_documents_from_url,
    split_into_chunks,
)
from .mcp_server import KnowledgeBaseMCPServer
from .rag import invalidate_rag_cache

indexes: dict[int, VectorStoreIndex] = {}


def kb_index_dir(kb_id: int) -> Path:
    return Path(config.settings.index_dir) / f"kb_{kb_id}"


def get_or_create_kb_index(kb_id: int) -> Optional[VectorStoreIndex]:
    if config.settings.mock_mode:
        return None
    if kb_id in indexes:
        return indexes[kb_id]
    index = load_or_create_index(kb_index_dir(kb_id))
    indexes[kb_id] = index
    return index


def drop_kb_index(kb_id: int) -> None:
    indexes.pop(kb_id, None)
    index_dir = kb_index_dir(kb_id)
    if index_dir.exists():
        shutil.rmtree(index_dir)


def mcp_response_headers() -> dict[str, str]:
    return {
        "MCP-Protocol-Version": config.settings.mcp_protocol_version,
        "Cache-Control": "no-cache",
    }


def mcp_authorized(request: Request) -> bool:
    token = config.settings.mcp_auth_token.strip()
    if not token:
        return True

    auth_value = request.headers.get("authorization", "")
    if auth_value.startswith("Bearer "):
        return auth_value[7:].strip() == token
    return False


mcp_protocol_server = KnowledgeBaseMCPServer(
    index_getter=get_or_create_kb_index,
    index_dropper=drop_kb_index,
)


def build_nodes_for_document_record(document: dict[str, Any]):
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


def reindex_documents(kb_id: int, documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        return {"reindexed": 0, "failed": []}

    drop_kb_index(kb_id)
    kb_index = get_or_create_kb_index(kb_id)

    reindexed = 0
    failed: list[dict[str, Any]] = []
    for document in documents:
        doc_id = int(document["id"])
        try:
            nodes, payload = build_nodes_for_document_record(document)
            replace_document_chunks(doc_id, payload)
            if not config.settings.mock_mode and kb_index is not None:
                insert_nodes(kb_index, nodes, index_dir=kb_index_dir(kb_id))
            reindexed += 1
        except Exception as exc:
            failed.append({"document_id": doc_id, "error": str(exc)})

    invalidate_rag_cache(kb_id=kb_id)
    return {"reindexed": reindexed, "failed": failed}
