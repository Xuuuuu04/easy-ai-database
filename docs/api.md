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
- 安装 `fastapi-mcp` 后，服务会自动挂载 `/mcp`。
- 通过 MCP 客户端可直接调用现有 FastAPI 路由（含 `/retrieve`、`/chat/rag`、`/chat/agent` 等）。
