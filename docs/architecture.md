# 架构说明（easy-ai-database）

> 模块边界与依赖方向约束请见：`docs/architecture-boundaries.md`

## 目标与边界
- **目标**：轻量级本地 AI RAG 知识库，支持文档/网页导入、RAG 问答、Agent 多步检索与总结，带引用溯源与历史。
- **边界**：单用户、单机、无云依赖；模型通过本地 OpenAI 兼容端点调用（如 LM Studio）。

## 关键技术选择
- **后端**：FastAPI（Python）
- **前端**：React + Vite
- **检索框架**：LlamaIndex（数据摄取、索引、Query Engine）
- **编排框架**：LangGraph（多步 Agent 流程与状态管理）
- **向量存储**：FAISS + SQLite（本地离线）

## 组件与职责
- **Ingestion 服务**
  - 文档解析：PDF/DOCX/TXT
  - URL 抓取：单页 HTML → 纯文本抽取
  - 分块（chunking）与元数据标注（页码、偏移）
  - Embedding：调用本地模型端点
  - 写入向量索引 + SQLite 元数据表

- **RAG 服务**
  - Query → 向量检索（LlamaIndex Query Engine）
  - 召回内容拼接上下文 → 生成答案
  - 引用溯源：chunk → 文档 → 页码/偏移

- **Agent 服务（LangGraph）**
  - Planner：任务分解（可选）
  - Tools：`search_kb`、`fetch_url`、`summarize`
  - 步骤输出：每一步展示输入/输出/引用
  - 最终摘要：整合并给出可追溯引用

- **History 服务**
  - 对话记录、检索结果、Agent 步骤持久化
  - 支持查询与删除

## 数据与存储
- 默认路径：`./data/`
- SQLite 表（建议）：
  - `documents`: 文档元信息
  - `chunks`: 分块内容 + 位置
  - `chats`/`messages`: 对话与消息
  - `agent_steps`: Agent 过程记录
- 向量索引：FAISS 文件（与 SQLite 映射）

## API 设计（概要）
- `POST /ingest/file` 上传文件
- `POST /ingest/url` 导入单页 URL
- `GET /kb/documents` 文档列表
- `DELETE /kb/documents/{id}` 删除文档
- `POST /chat/rag` RAG 问答
- `POST /chat/agent` Agent 模式
- `GET /chat/history` 历史列表
- `GET /chat/{id}` 对话详情

## 前端页面
- **Chat**：模式切换（RAG/Agent）、引用展示、步骤面板
- **Knowledge Base**：文档列表、导入、删除
- **Settings**：模型端点配置、Embedding 配置、索引路径

## 可靠性与可控性策略
- **引用不足拒答**：无足够引用时提示并拒答
- **工具超时**：对外部抓取设置超时与失败重试
- **可解释性**：Agent 步骤可查看、可追溯
- **数据隔离**：所有数据本地存储，无外发

## 扩展点
- 新工具节点：结构化抽取、表格分析、代码解释
- 多索引策略：按主题/项目分库
- 混合检索：BM25 + 向量融合（可在 LlamaIndex 扩展）

## 测试策略
- 单元测试：chunking、引用生成、SQLite CRUD
- API 测试：导入/检索/问答
- Smoke：导入 → 问答 → 引用 → 历史

## 里程碑（建议）
1. MVP：导入 + RAG + 引用 + 历史
2. Agent：多步流程 + 步骤展示
3. 体验：UI 打磨 + 一键启动
