# Architecture Overview

See also: `docs/architecture-boundaries.md`.

## Goals

- Local-first RAG knowledge base with source-grounded answers.
- Multi-step agent workflow with transparent evidence and steps.
- MCP-compatible access for external clients/tools.

## Stack

- Backend: FastAPI (`src/backend/app`)
- Frontend: React + Vite (`src/frontend/src`)
- Retrieval/indexing: LlamaIndex + FAISS + SQLite
- Persistence: SQLite for documents/chunks/chats/agent steps

## Backend Runtime Flow

1. Ingestion
   - `POST /ingest/file` and `POST /ingest/url`
   - parse -> chunk -> persist metadata -> write vectors
2. Retrieval
   - hybrid retrieval (vector + BM25)
   - rerank + quality/focus gates + citation assembly
3. Answering
   - RAG mode: grounded answer from retrieved context
   - Agent mode: multi-step retrieval/exploration + final summarize
4. MCP
   - `/mcp/v1` JSON-RPC endpoint exposing KB/chat tools and resources

## Core Modules

- `src/backend/app/rag.py`: retrieval pipeline, quality gating, grounded generation
- `src/backend/app/agent.py`: exploration rounds, citation merge, answer refinement
- `src/backend/app/db.py`: schema + CRUD
- `src/backend/app/state.py`: index cache lifecycle + MCP server instance
- `src/backend/app/routes/`: transport-only route handlers

## Data Layout

- Runtime defaults (configurable by env):
  - `DATA_DIR`: `./data`
  - `DB_PATH`: `./data/app.db`
  - `INDEX_DIR`: `./data/index`
- Main DB entities:
  - `knowledge_bases`, `documents`, `chunks`
  - `chats`, `messages`, `agent_steps`
  - `semantic_cache`

## Quality Controls

- Citation-required mode (`REQUIRE_CITATIONS`, `MIN_CITATIONS`)
- Retrieval quality/focus/anchor coverage gates
- OOD-risk rejection for unsupported domain claims
- Deterministic fallbacks when LLM call is unavailable
