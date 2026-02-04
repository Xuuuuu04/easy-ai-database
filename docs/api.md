# API 参考

## 导入
### POST /ingest/file
- form-data: `file`
- 返回：`{ document_id, chunks }`

### POST /ingest/url
```json
{ "url": "https://example.com" }
```

## 知识库
### GET /kb/documents
- 返回文档列表

### DELETE /kb/documents/{id}
- 删除文档

## 问答
### POST /chat/rag
```json
{ "question": "...", "chat_id": 1 }
```

### POST /chat/agent
```json
{ "question": "...", "chat_id": 1 }
```

## 历史
### GET /chat/history
### GET /chat/{id}
