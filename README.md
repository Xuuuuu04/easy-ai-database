# easy-ai-database

轻量级本地 AI RAG 知识库系统，支持文档/网页导入、RAG 问答、Agent 多步检索、MCP 工具调用。

## 核心特性
- 本地数据存储：SQLite + FAISS，默认 `./data`
- 文档导入：PDF / DOCX / TXT + URL 抓取
- 检索问答：`/chat/rag`、`/chat/agent`、`/retrieve`
- MCP 协议：`/mcp/v1`，支持外部 AI 工具调用
- 设置面板：前端直接编辑根目录 `.env`，支持 MCP 开关、部署地址、命令生成

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

前端默认：`http://localhost:5173`  
后端默认：`http://localhost:8000`

## 常用命令
```bash
python3 -m pytest -q
cd src/frontend && npm run test
cd src/frontend && npm run build
```

## 目录结构
- `src/backend/`: FastAPI 后端
- `src/frontend/`: React 前端
- `data/`: 本地数据库与索引
- `docs/`: 架构与运维文档
- `scripts/`: 开发脚本

## MCP 安装
在 Web 设置面板中填写 `DEPLOYMENT_URL` 并保存后，会自动生成 Claude Code / CodeX 的 MCP 安装命令。

## Language
- 中文: [`README.md`](./README.md)
- English: [`README_EN.md`](./README_EN.md)
