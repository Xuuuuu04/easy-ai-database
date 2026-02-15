from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .agent import run_agent
from .config import settings
from .db import (
    add_agent_step,
    add_message,
    batch_delete_documents,
    create_chat,
    create_knowledge_base,
    delete_document,
    delete_knowledge_base,
    document_title_exists,
    ensure_knowledge_base_exists,
    get_chat,
    get_documents_by_ids,
    insert_chunks,
    insert_document,
    list_chats,
    list_documents,
    list_knowledge_bases,
    replace_document_chunks,
)
from .indexer import insert_nodes
from .ingest import (
    build_documents_from_file,
    build_documents_from_url,
    split_into_chunks,
)
from .rag import invalidate_rag_cache, query_rag, retrieve_context
from .source_access import resolve_allowed_source_path

JsonObj = dict[str, Any]
IndexGetter = Callable[[int], Any]
IndexDropper = Callable[[int], None]

MCP_PROTOCOL_VERSION = "2025-11-25"


class MCPProtocolError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _dump_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _to_int(value: Any, field: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise MCPProtocolError(-32602, f"Missing required field: {field}")
        return default
    try:
        return int(value)
    except Exception as exc:
        raise MCPProtocolError(-32602, f"Invalid integer for {field}") from exc


def _to_text(value: Any, field: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise MCPProtocolError(-32602, f"Missing required field: {field}")
    return text


def _to_obj(value: Any, field: str) -> JsonObj:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MCPProtocolError(-32602, f"Field {field} must be an object")
    return value


class KnowledgeBaseMCPServer:
    def __init__(self, index_getter: IndexGetter, index_dropper: IndexDropper):
        self._index_getter = index_getter
        self._index_dropper = index_dropper

        self._tool_handlers: dict[str, Callable[[JsonObj], Any]] = {
            "kb.list": self._tool_kb_list,
            "kb.create": self._tool_kb_create,
            "kb.delete": self._tool_kb_delete,
            "kb.reindex": self._tool_kb_reindex,
            "kb.documents.list": self._tool_kb_documents_list,
            "kb.documents.delete": self._tool_kb_documents_delete,
            "kb.documents.batch_delete": self._tool_kb_documents_batch_delete,
            "kb.documents.reindex": self._tool_kb_documents_reindex,
            "kb.ingest.url": self._tool_kb_ingest_url,
            "kb.ingest.file_path": self._tool_kb_ingest_file_path,
            "kb.retrieve": self._tool_kb_retrieve,
            "kb.answer.rag": self._tool_kb_answer_rag,
            "kb.answer.agent": self._tool_kb_answer_agent,
            "kb.preview": self._tool_kb_preview,
            "chat.history.list": self._tool_chat_history_list,
            "chat.get": self._tool_chat_get,
        }

    def protocol_version(self) -> str:
        return str(
            getattr(settings, "mcp_protocol_version", MCP_PROTOCOL_VERSION)
            or MCP_PROTOCOL_VERSION
        )

    def server_name(self) -> str:
        return str(
            getattr(settings, "mcp_server_name", "easy-ai-database-mcp")
            or "easy-ai-database-mcp"
        )

    def handle_message(self, payload: Any) -> JsonObj | None:
        if not isinstance(payload, dict):
            return self._error_response(
                None, -32600, "Invalid Request: expected object payload"
            )

        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params")

        if payload.get("jsonrpc") != "2.0":
            return self._error_response(
                request_id, -32600, "Invalid Request: jsonrpc must be '2.0'"
            )
        if not isinstance(method, str) or not method:
            return self._error_response(
                request_id, -32600, "Invalid Request: method is required"
            )

        try:
            result = self._dispatch(method, params)
        except MCPProtocolError as exc:
            return self._error_response(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:
            return self._error_response(
                request_id, -32603, "Internal error", {"detail": str(exc)}
            )

        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _dispatch(self, method: str, params: Any) -> JsonObj:
        if method in {
            "notifications/initialized",
            "initialized",
            "notifications/cancelled",
        }:
            return {}
        if method == "initialize":
            return self._handle_initialize(_to_obj(params, "params"))
        if method == "ping":
            return {}
        if method == "tools/list":
            return self._handle_tools_list(_to_obj(params, "params"))
        if method == "tools/call":
            return self._handle_tools_call(_to_obj(params, "params"))
        if method == "resources/list":
            return self._handle_resources_list(_to_obj(params, "params"))
        if method == "resources/read":
            return self._handle_resources_read(_to_obj(params, "params"))
        if method == "resources/templates/list":
            return self._handle_resource_templates_list(_to_obj(params, "params"))
        if method == "prompts/list":
            return self._handle_prompts_list(_to_obj(params, "params"))
        if method == "prompts/get":
            return self._handle_prompts_get(_to_obj(params, "params"))
        raise MCPProtocolError(-32601, f"Method not found: {method}")

    def _handle_initialize(self, params: JsonObj) -> JsonObj:
        requested = str(params.get("protocolVersion") or self.protocol_version())
        supported = self.protocol_version()
        negotiated = requested if requested == supported else supported

        return {
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": self.server_name(),
                "version": "1.0.0",
            },
            "instructions": (
                "Use kb.retrieve for side-effect-free retrieval. "
                "Use kb.answer.rag or kb.answer.agent for grounded answering."
            ),
        }

    def _handle_tools_list(self, params: JsonObj) -> JsonObj:
        if not settings.mcp_tools_enabled:
            return {"tools": []}

        tools = [
            {
                "name": "kb.list",
                "description": "List knowledge bases.",
                "inputSchema": {"type": "object", "additionalProperties": False},
            },
            {
                "name": "kb.create",
                "description": "Create a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.delete",
                "description": "Delete a knowledge base by id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"kb_id": {"type": "integer", "minimum": 1}},
                    "required": ["kb_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.reindex",
                "description": "Reindex all documents in a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"kb_id": {"type": "integer", "minimum": 1}},
                    "required": ["kb_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.documents.list",
                "description": "List documents in a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"kb_id": {"type": "integer", "minimum": 1}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.documents.delete",
                "description": "Delete one document from a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "document_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["document_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.documents.batch_delete",
                "description": "Delete multiple documents from a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                    },
                    "required": ["document_ids"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.documents.reindex",
                "description": "Reindex a single document.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "document_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["document_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.ingest.url",
                "description": "Ingest URL content into a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.ingest.file_path",
                "description": "Ingest a local file path into a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "file_path": {"type": "string"},
                        "title": {"type": "string"},
                        "allow_duplicate": {"type": "boolean"},
                    },
                    "required": ["file_path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.retrieve",
                "description": "Retrieve grounded context without final answer generation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "kb_id": {"type": "integer", "minimum": 1},
                        "chat_id": {"type": "integer", "minimum": 1},
                        "top_k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.answer.rag",
                "description": "Generate answer from RAG pipeline.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "kb_id": {"type": "integer", "minimum": 1},
                        "chat_id": {"type": "integer", "minimum": 1},
                        "persist_chat": {"type": "boolean"},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.answer.agent",
                "description": "Run multi-step agent workflow for research-style answering.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "kb_id": {"type": "integer", "minimum": 1},
                        "chat_id": {"type": "integer", "minimum": 1},
                        "persist_chat": {"type": "boolean"},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kb.preview",
                "description": "Read preview content for a citation source path or URL.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "source": {"type": "string"},
                    },
                    "required": ["source"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "chat.history.list",
                "description": "List chats under a knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"kb_id": {"type": "integer", "minimum": 1}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "chat.get",
                "description": "Get chat detail including messages and agent steps.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kb_id": {"type": "integer", "minimum": 1},
                        "chat_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["chat_id"],
                    "additionalProperties": False,
                },
            },
        ]
        page, next_cursor = self._paginate(tools, params)
        result: JsonObj = {"tools": page}
        if next_cursor is not None:
            result["nextCursor"] = next_cursor
        return result

    def _handle_tools_call(self, params: JsonObj) -> JsonObj:
        if not settings.mcp_tools_enabled:
            raise MCPProtocolError(-32603, "MCP tools are disabled by server settings")

        name = _to_text(params.get("name"), "name")
        arguments = _to_obj(params.get("arguments"), "arguments")

        handler = self._tool_handlers.get(name)
        if handler is None:
            raise MCPProtocolError(-32602, f"Unknown tool: {name}")

        try:
            payload = handler(arguments)
            return self._tool_success(payload)
        except MCPProtocolError as exc:
            return self._tool_error(exc.message, {"code": exc.code, "data": exc.data})
        except Exception as exc:
            return self._tool_error(str(exc), None)

    def _handle_resources_list(self, params: JsonObj) -> JsonObj:
        resources: list[JsonObj] = []
        for kb in list_knowledge_bases():
            kb_id = int(kb["id"])
            resources.append(
                {
                    "uri": f"kb://{kb_id}",
                    "name": f"knowledge-base-{kb_id}",
                    "title": str(kb.get("name") or f"KB {kb_id}"),
                    "description": "Knowledge base metadata",
                    "mimeType": "application/json",
                }
            )
            resources.append(
                {
                    "uri": f"kb://{kb_id}/documents",
                    "name": f"knowledge-base-{kb_id}-documents",
                    "description": "Document list",
                    "mimeType": "application/json",
                }
            )
            resources.append(
                {
                    "uri": f"kb://{kb_id}/chats",
                    "name": f"knowledge-base-{kb_id}-chats",
                    "description": "Chat list",
                    "mimeType": "application/json",
                }
            )

        page, next_cursor = self._paginate(resources, params)
        result: JsonObj = {"resources": page}
        if next_cursor is not None:
            result["nextCursor"] = next_cursor
        return result

    def _handle_resource_templates_list(self, params: JsonObj) -> JsonObj:
        templates = [
            {
                "uriTemplate": "kb://{kb_id}/documents",
                "name": "kb-documents",
                "description": "List documents from specified knowledge base",
                "mimeType": "application/json",
            },
            {
                "uriTemplate": "kb://{kb_id}/chat/{chat_id}",
                "name": "kb-chat-detail",
                "description": "Get one chat detail",
                "mimeType": "application/json",
            },
        ]
        page, next_cursor = self._paginate(templates, params)
        result: JsonObj = {"resourceTemplates": page}
        if next_cursor is not None:
            result["nextCursor"] = next_cursor
        return result

    def _handle_resources_read(self, params: JsonObj) -> JsonObj:
        uri = _to_text(params.get("uri"), "uri")
        parsed = urlparse(uri)
        if parsed.scheme != "kb":
            raise MCPProtocolError(-32602, f"Unsupported resource uri: {uri}")

        kb_id = _to_int(parsed.netloc, "kb_id")
        self._ensure_kb_exists(kb_id)

        path_parts = [item for item in parsed.path.split("/") if item]
        if not path_parts:
            payload = {"knowledge_base": self._kb_by_id(kb_id)}
        elif path_parts == ["documents"]:
            payload = {"documents": list_documents(kb_id=kb_id)}
        elif path_parts == ["chats"]:
            payload = {"chats": list_chats(kb_id=kb_id)}
        elif len(path_parts) == 2 and path_parts[0] == "chat":
            chat_id = _to_int(path_parts[1], "chat_id")
            chat = get_chat(chat_id, kb_id=kb_id)
            if not chat:
                raise MCPProtocolError(-32602, "Chat not found")
            payload = {"chat": chat}
        elif len(path_parts) == 2 and path_parts[0] == "document":
            doc_id = _to_int(path_parts[1], "document_id")
            docs = get_documents_by_ids([doc_id], kb_id)
            if not docs:
                raise MCPProtocolError(-32602, "Document not found")
            payload = {"document": docs[0]}
        else:
            raise MCPProtocolError(
                -32602, f"Unsupported resource uri path: {parsed.path}"
            )

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": _dump_text(payload),
                }
            ]
        }

    def _handle_prompts_list(self, params: JsonObj) -> JsonObj:
        prompts = [
            {
                "name": "kb-grounded-answer",
                "description": "Generate grounded answer using retrieval evidence.",
                "arguments": [
                    {"name": "question", "required": True},
                    {"name": "kb_id", "required": False},
                    {"name": "mode", "required": False},
                ],
            },
            {
                "name": "kb-research-workflow",
                "description": "Research workflow prompt for agent-style investigation.",
                "arguments": [
                    {"name": "question", "required": True},
                    {"name": "kb_id", "required": False},
                    {"name": "top_k", "required": False},
                ],
            },
        ]
        page, next_cursor = self._paginate(prompts, params)
        result: JsonObj = {"prompts": page}
        if next_cursor is not None:
            result["nextCursor"] = next_cursor
        return result

    def _handle_prompts_get(self, params: JsonObj) -> JsonObj:
        name = _to_text(params.get("name"), "name")
        args = _to_obj(params.get("arguments"), "arguments")

        question = (
            _to_text(args.get("question"), "question") if args.get("question") else ""
        )
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)

        if name == "kb-grounded-answer":
            mode = str(args.get("mode") or "rag").strip().lower()
            prompt = (
                "You are working with a local MCP knowledge base service. "
                f"Knowledge base id is {kb_id}. "
                f"Question: {question}. "
                "First call tool 'kb.retrieve'. Then call "
                f"'kb.answer.{mode if mode in {'rag', 'agent'} else 'rag'}' and produce a cited answer."
            )
            return {
                "description": "Grounded KB answering workflow",
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": prompt}}
                ],
            }

        if name == "kb-research-workflow":
            top_k = _to_int(args.get("top_k"), "top_k", 6)
            prompt = (
                "Run an explicit research workflow: retrieve evidence, inspect sources, and summarize findings. "
                f"Use kb_id={kb_id}, top_k={top_k}, question={question}. "
                "Use tools in this order: kb.retrieve -> kb.preview (for key sources) -> kb.answer.agent."
            )
            return {
                "description": "Agent-style research workflow",
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": prompt}}
                ],
            }

        raise MCPProtocolError(-32602, f"Unknown prompt: {name}")

    def _tool_kb_list(self, _: JsonObj) -> JsonObj:
        return {"knowledge_bases": list_knowledge_bases()}

    def _tool_kb_create(self, args: JsonObj) -> JsonObj:
        name = _to_text(args.get("name"), "name")
        description = _to_text(args.get("description"), "description", required=False)
        return {"knowledge_base": create_knowledge_base(name, description)}

    def _tool_kb_delete(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id")
        self._ensure_kb_exists(kb_id)
        delete_knowledge_base(kb_id)
        self._index_dropper(kb_id)
        invalidate_rag_cache(kb_id=kb_id)
        return {"deleted": kb_id}

    def _tool_kb_reindex(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id")
        self._ensure_kb_exists(kb_id)
        result = self._reindex_documents(kb_id, list_documents(kb_id=kb_id))
        return {"kb_id": kb_id, **result}

    def _tool_kb_documents_list(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        self._ensure_kb_exists(kb_id)
        return {"kb_id": kb_id, "documents": list_documents(kb_id=kb_id)}

    def _tool_kb_documents_delete(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        doc_id = _to_int(args.get("document_id"), "document_id")
        self._ensure_kb_exists(kb_id)
        delete_document(doc_id, kb_id=kb_id)
        self._index_dropper(kb_id)
        invalidate_rag_cache(kb_id=kb_id)
        return {"kb_id": kb_id, "deleted": doc_id}

    def _tool_kb_documents_batch_delete(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        self._ensure_kb_exists(kb_id)
        document_ids = args.get("document_ids")
        if not isinstance(document_ids, list) or not document_ids:
            raise MCPProtocolError(-32602, "document_ids must be a non-empty array")
        normalized_ids = [_to_int(item, "document_ids[]") for item in document_ids]
        deleted = batch_delete_documents(normalized_ids, kb_id=kb_id)
        self._index_dropper(kb_id)
        invalidate_rag_cache(kb_id=kb_id)
        return {"kb_id": kb_id, "deleted": deleted}

    def _tool_kb_documents_reindex(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        doc_id = _to_int(args.get("document_id"), "document_id")
        self._ensure_kb_exists(kb_id)
        docs = get_documents_by_ids([doc_id], kb_id)
        if not docs:
            raise MCPProtocolError(-32602, "Document not found")
        result = self._reindex_documents(kb_id, docs)
        return {"kb_id": kb_id, **result}

    def _tool_kb_ingest_url(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        url = _to_text(args.get("url"), "url")
        self._ensure_kb_exists(kb_id)

        doc_id = insert_document(url, "url", url, kb_id=kb_id)
        docs = build_documents_from_url(url)
        nodes = split_into_chunks(docs)

        kb_index = self._index_getter(kb_id)
        if not settings.mock_mode and kb_index is not None:
            insert_nodes(
                kb_index, nodes, index_dir=Path(settings.index_dir) / f"kb_{kb_id}"
            )

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
        invalidate_rag_cache(kb_id=kb_id)
        return {"kb_id": kb_id, "document_id": doc_id, "chunks": len(nodes)}

    def _tool_kb_ingest_file_path(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        file_path = _to_text(args.get("file_path"), "file_path")
        title = _to_text(args.get("title"), "title", required=False)
        allow_duplicate = bool(args.get("allow_duplicate", False))
        self._ensure_kb_exists(kb_id)

        source_path = Path(file_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise MCPProtocolError(-32602, "Local file path not found")

        final_title = title or source_path.name
        if not allow_duplicate and document_title_exists(final_title, kb_id=kb_id):
            raise MCPProtocolError(
                -32602,
                "Duplicate filename detected. Set allow_duplicate=true to ingest anyway.",
            )

        doc_id = insert_document(final_title, "file", str(source_path), kb_id=kb_id)
        docs = build_documents_from_file(source_path, str(source_path))
        nodes = split_into_chunks(docs)

        kb_index = self._index_getter(kb_id)
        if not settings.mock_mode and kb_index is not None:
            insert_nodes(
                kb_index,
                nodes,
                index_dir=Path(settings.index_dir) / f"kb_{kb_id}",
            )

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

        return {
            "kb_id": kb_id,
            "document_id": doc_id,
            "chunks": len(nodes),
            "source_ref": str(source_path),
            "title": final_title,
        }

    def _tool_kb_retrieve(self, args: JsonObj) -> JsonObj:
        question = _to_text(args.get("question"), "question")
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        chat_id = args.get("chat_id")
        chat_id_value = _to_int(chat_id, "chat_id") if chat_id is not None else None
        top_k = _to_int(args.get("top_k"), "top_k", 6)
        self._ensure_kb_exists(kb_id)
        index = self._index_getter(kb_id)
        payload = retrieve_context(
            index,
            question,
            kb_id=kb_id,
            chat_id=chat_id_value,
            top_k=max(1, top_k),
        )
        return {"kb_id": kb_id, **payload}

    def _tool_kb_answer_rag(self, args: JsonObj) -> JsonObj:
        question = _to_text(args.get("question"), "question")
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        self._ensure_kb_exists(kb_id)

        chat_id_arg = args.get("chat_id")
        chat_id = _to_int(chat_id_arg, "chat_id") if chat_id_arg is not None else None
        persist_chat = bool(args.get("persist_chat", False))

        if persist_chat:
            effective_chat_id = chat_id or create_chat(kb_id=kb_id)
            add_message(effective_chat_id, "user", question)
            result = query_rag(
                self._index_getter(kb_id),
                question,
                chat_id=effective_chat_id,
                kb_id=kb_id,
            )
            add_message(effective_chat_id, "assistant", str(result.get("answer") or ""))
            return {"kb_id": kb_id, "chat_id": effective_chat_id, **result}

        result = query_rag(
            self._index_getter(kb_id), question, chat_id=chat_id, kb_id=kb_id
        )
        return {"kb_id": kb_id, **result}

    def _tool_kb_answer_agent(self, args: JsonObj) -> JsonObj:
        question = _to_text(args.get("question"), "question")
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        self._ensure_kb_exists(kb_id)

        chat_id_arg = args.get("chat_id")
        chat_id = _to_int(chat_id_arg, "chat_id") if chat_id_arg is not None else None
        persist_chat = bool(args.get("persist_chat", False))

        if persist_chat:
            effective_chat_id = chat_id or create_chat(kb_id=kb_id)
            add_message(effective_chat_id, "user", question)
            result = run_agent(
                self._index_getter(kb_id),
                question,
                chat_id=effective_chat_id,
                kb_id=kb_id,
            )
            add_message(effective_chat_id, "assistant", str(result.get("answer") or ""))
            for step_index, step in enumerate(result.get("steps", [])):
                add_agent_step(
                    effective_chat_id,
                    step_index,
                    str(step.get("tool") or ""),
                    str(step.get("input") or ""),
                    str(step.get("output") or ""),
                    str(step.get("citations") or "[]"),
                )
            return {"kb_id": kb_id, "chat_id": effective_chat_id, **result}

        result = run_agent(
            self._index_getter(kb_id), question, chat_id=chat_id, kb_id=kb_id
        )
        return {"kb_id": kb_id, **result}

    def _tool_kb_preview(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        source = _to_text(args.get("source"), "source")
        self._ensure_kb_exists(kb_id)
        return self._preview_source(source=source, kb_id=kb_id)

    def _tool_chat_history_list(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        self._ensure_kb_exists(kb_id)
        return {"kb_id": kb_id, "chats": list_chats(kb_id=kb_id)}

    def _tool_chat_get(self, args: JsonObj) -> JsonObj:
        kb_id = _to_int(args.get("kb_id"), "kb_id", 1)
        chat_id = _to_int(args.get("chat_id"), "chat_id")
        self._ensure_kb_exists(kb_id)
        chat = get_chat(chat_id, kb_id=kb_id)
        if not chat:
            raise MCPProtocolError(-32602, "Chat not found")
        return {"kb_id": kb_id, "chat": chat}

    def _preview_source(self, source: str, kb_id: int) -> JsonObj:
        normalized_source = source.strip()
        if not normalized_source or normalized_source == "local":
            raise MCPProtocolError(-32602, "Source is not previewable")

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
            raise MCPProtocolError(
                -32602, "Preview path is outside upload root"
            ) from exc
        except FileNotFoundError:
            raise MCPProtocolError(-32602, "Preview source file not found")

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

    def _build_nodes_for_document_record(
        self, document: JsonObj
    ) -> tuple[list[Any], list[JsonObj]]:
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

    def _reindex_documents(self, kb_id: int, documents: list[JsonObj]) -> JsonObj:
        if not documents:
            return {"reindexed": 0, "failed": []}

        self._index_dropper(kb_id)
        kb_index = self._index_getter(kb_id)
        reindexed = 0
        failed: list[JsonObj] = []

        for document in documents:
            doc_id = int(document["id"])
            try:
                nodes, payload = self._build_nodes_for_document_record(document)
                replace_document_chunks(doc_id, payload)
                if not settings.mock_mode and kb_index is not None:
                    insert_nodes(
                        kb_index,
                        nodes,
                        index_dir=Path(settings.index_dir) / f"kb_{kb_id}",
                    )
                reindexed += 1
            except Exception as exc:
                failed.append({"document_id": doc_id, "error": str(exc)})

        invalidate_rag_cache(kb_id=kb_id)
        return {"reindexed": reindexed, "failed": failed}

    def _ensure_kb_exists(self, kb_id: int) -> None:
        if not ensure_knowledge_base_exists(kb_id):
            raise MCPProtocolError(-32602, "Knowledge base not found")

    def _kb_by_id(self, kb_id: int) -> JsonObj:
        for kb in list_knowledge_bases():
            if int(kb.get("id", -1)) == kb_id:
                return kb
        raise MCPProtocolError(-32602, "Knowledge base not found")

    def _paginate(
        self, items: list[JsonObj], params: JsonObj
    ) -> tuple[list[JsonObj], str | None]:
        raw_cursor = params.get("cursor")
        raw_limit = params.get("limit")

        start = 0
        if raw_cursor is not None:
            try:
                start = int(str(raw_cursor))
            except Exception as exc:
                raise MCPProtocolError(-32602, "Invalid cursor") from exc
            if start < 0:
                raise MCPProtocolError(-32602, "Invalid cursor")

        limit = 50
        if raw_limit is not None:
            try:
                limit = int(raw_limit)
            except Exception as exc:
                raise MCPProtocolError(-32602, "Invalid limit") from exc
            limit = max(1, min(limit, 200))

        end = start + limit
        page = items[start:end]
        next_cursor = str(end) if end < len(items) else None
        return page, next_cursor

    def _tool_success(self, payload: Any) -> JsonObj:
        return {
            "content": [{"type": "text", "text": _dump_text(payload)}],
            "structuredContent": payload,
            "isError": False,
        }

    def _tool_error(self, message: str, details: Any | None) -> JsonObj:
        content = message.strip() or "Tool execution failed"
        structured: JsonObj = {"error": content}
        if details is not None:
            structured["details"] = details
        return {
            "content": [{"type": "text", "text": _dump_text(structured)}],
            "structuredContent": structured,
            "isError": True,
        }

    def _error_response(
        self, request_id: Any, code: int, message: str, data: Any | None = None
    ) -> JsonObj:
        payload: JsonObj = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            payload["error"]["data"] = data
        return payload
