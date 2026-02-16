# easy-ai-database

轻量级本地 AI RAG 知识库系统，支持文档/网页导入、RAG 问答、Agent 多步检索与 MCP 工具访问。

## 特性

- 本地优先：SQLite + FAISS，默认数据目录 `./data`
- 导入能力：PDF / DOCX / TXT / URL
- 问答模式：`/chat/rag`、`/chat/agent`、`/retrieve`
- MCP 协议：`/mcp/v1`（JSON-RPC + SSE）
- 前端设置：可直接管理 `.env` 关键配置

## 快速开始

### Docker

```bash
cp .env.example .env
docker compose up --build
```

### 本地脚本

```bash
cp .env.example .env
./scripts/run.sh
```

- 前端默认：`http://localhost:5173`
- 后端默认：`http://localhost:8000`

## 开发命令

```bash
python3 -m pytest -q
cd src/frontend && npm run test
cd src/frontend && npm run build
```

## 项目结构

- 后端：`src/backend/`
- 前端：`src/frontend/`
- 文档：`docs/`
- 脚本：`scripts/`

## 文档导航

- 边界约束：`docs/architecture-boundaries.md`
- API 参考：`docs/api.md`

## 开源协作

- 贡献指南：`CONTRIBUTING.md`
- 行为准则：`CODE_OF_CONDUCT.md`
- 安全策略：`SECURITY.md`
- 变更日志：`CHANGELOG.md`

## Language

- 中文：`README.md`
- English：`README_EN.md`
