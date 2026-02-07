# 本机知识库 AI 助手

单用户、本地离线的知识库 AI 助手，支持文档/网页导入、RAG 问答、Agent 多步检索与总结（带引用溯源与历史）。

## 功能
- 本地文档导入：PDF/DOCX/TXT
- URL 导入：单页抓取
- RAG 问答：引用溯源
- Agent 模式：多步检索/总结 + 步骤记录
- 全程离线存储：`./data`

## 快速开始

### 方式一：Docker Compose
```bash
cp .env.example .env
# 修改 .env 中的模型端点

docker compose up --build
```
前端：`http://localhost:5173`，后端：`http://localhost:8000`

### 方式二：本地脚本
```bash
cp .env.example .env
./scripts/run.sh
```

## 后端 API（摘要）
- `POST /ingest/file` 上传文件
- `POST /ingest/url` 导入 URL
- `GET /kb/documents` 文档列表
- `DELETE /kb/documents/{id}` 删除文档
- `POST /chat/rag` RAG 问答
- `POST /chat/agent` Agent 模式
- `GET /chat/history` 历史
- `GET /chat/{id}` 对话详情

## 项目结构
```
backend/   FastAPI 服务
frontend/  React UI
docs/      架构与文档
data/      本地数据
scripts/   脚本
```

## 模型说明
- 使用 OpenAI 兼容 API 的本地端点（如 LM Studio）
- 在 `.env` 中配置 `LLM_*` 与 `EMBED_*`

## 测试
```bash
cd backend
MOCK_MODE=1 pytest
```

## 运行与排错
详见 `docs/runbook.md`。


## 开发进度（截至 2026-02-07）
- 已完成可公开仓库基线整理：补齐许可证、清理敏感与内部说明文件。
- 当前版本可构建/可运行，后续迭代以 issue 与提交记录持续公开追踪。

## Language
- 中文：[`README.md`](./README.md)
- English：[`README_EN.md`](./README_EN.md)

## 统一源码目录
- 源码入口：[`src/`](./src)

## 目录结构
- 结构说明：[`docs/PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md)

## 迁移说明
- 核心目录已迁移到 `src/` 下。
- 根目录保留兼容软链接，历史命令与路径可继续使用。
