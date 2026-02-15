# API 参考

## 导入
### POST /ingest/file
- query: `kb_id`(默认 1), `allow_duplicate`(默认 false)
- form-data: `file`
- 返回：`{ document_id, chunks, kb_id }`

### POST /ingest/url
```json
{ "url": "https://example.com", "kb_id": 1 }
```

## 知识库
### GET /kb
- 返回知识库列表

### POST /kb
```json
{ "name": "仓颉文档", "description": "可选" }
```

### DELETE /kb/{kb_id}
- 删除知识库（默认知识库 `id=1` 不允许删除）

### GET /kb/documents
- query: `kb_id`
- 返回指定知识库文档列表

### DELETE /kb/documents/{id}
- query: `kb_id`
- 删除指定知识库文档

### POST /kb/{kb_id}/reindex
- 重建整个知识库索引

### POST /kb/documents/reindex-batch
```json
{ "kb_id": 1, "document_ids": [1,2,3] }
```

### POST /kb/documents/batch-delete
```json
{ "kb_id": 1, "document_ids": [1,2,3] }
```

### POST /kb/documents/{doc_id}/reindex
- query: `kb_id`
- 重建单文档索引

## 问答
### POST /chat/rag
```json
{ "question": "...", "chat_id": 1, "kb_id": 1, "stream": true }
```

### POST /chat/agent
```json
{ "question": "...", "chat_id": 1, "kb_id": 1, "stream": true }
```

### POST /retrieve
- 纯检索接口（不做最终回答生成），适合外部工具/MCP调用
```json
{ "question": "Array 是什么", "kb_id": 1, "chat_id": null, "top_k": 6 }
```

## 历史
### GET /chat/history
- query: `kb_id`

### GET /chat/{id}
- query: `kb_id`

## 评测与调优
### POST /eval/retrieval/generate-dataset
- 从知识库自动生成检索评测集
```json
{ "kb_id": 1, "case_count": 20, "use_llm": true }
```

### POST /eval/retrieval
- 运行检索评测（支持自动参数网格调优与 LLM-as-Judge）
```json
{
  "kb_id": 1,
  "auto_tune": true,
  "k": 5,
  "include_case_results": false,
  "include_llm_judge": true,
  "llm_judge_sample_size": 10,
  "llm_judge_on_all_configs": false,
  "cases": [
    {"id": "c1", "query": "Array 定义", "relevant_snippets": ["Array<T>"]}
  ]
}
```

## MCP
- 标准 MCP 端点：`POST /mcp/v1`（JSON-RPC 2.0），`GET /mcp/v1`（SSE keep-alive）。
- 协议版本：默认 `2025-11-25`，响应头包含 `MCP-Protocol-Version`。
- 支持方法：`initialize`、`ping`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`resources/templates/list`、`prompts/list`、`prompts/get`。
- 工具集（核心）：
  - `kb.list` / `kb.create` / `kb.delete` / `kb.reindex`
  - `kb.documents.list` / `kb.documents.delete` / `kb.documents.batch_delete` / `kb.documents.reindex`
  - `kb.ingest.url` / `kb.ingest.file_path`
  - `kb.retrieve` / `kb.answer.rag` / `kb.answer.agent` / `kb.preview`
  - `chat.history.list` / `chat.get`
- 资源 URI：
  - `kb://{kb_id}`
  - `kb://{kb_id}/documents`
  - `kb://{kb_id}/chats`
  - `kb://{kb_id}/chat/{chat_id}`
  - `kb://{kb_id}/document/{doc_id}`
- 提示模板：`kb-grounded-answer`、`kb-research-workflow`。

### MCP 鉴权
- 环境变量 `MCP_AUTH_TOKEN` 为空时不鉴权。
- 设置后需携带 `Authorization: Bearer <token>` 调用 `/mcp/v1`。

### Claude Code 连接（stdio）
- 启动脚本：`scripts/mcp_stdio.py`。
- Claude Code 示例配置：
```json
{
  "mcpServers": {
    "easy-ai-database": {
      "command": "python",
      "args": ["./scripts/mcp_stdio.py"],
      "env": {
        "DATA_DIR": "./data",
        "DB_PATH": "./data/app.db",
        "INDEX_DIR": "./data/index",
        "MCP_PROTOCOL_VERSION": "2025-11-25"
      }
    }
  }
}
```
