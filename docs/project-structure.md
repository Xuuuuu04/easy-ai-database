# Project Structure

Last updated: 2026-02-16

## Repository Layout

```text
.
├── .github/                     # CI workflow
├── docs/                        # Architecture, API, runbooks
├── e2e/                         # Playwright E2E test entry
├── scripts/                     # Dev utilities (bootstrap, MCP, eval)
├── src/
│   ├── backend/                 # FastAPI backend
│   │   ├── app/                 # Core services/routes/state
│   │   └── tests/               # Backend tests
│   └── frontend/                # Vite + React UI
├── docker-compose.yml
├── README.md
└── README_EN.md
```

## Backend Modules

- `src/backend/app/main.py`: app composition only (lifespan, middleware, router mount)
- `src/backend/app/routes/`: HTTP transport layer
- `src/backend/app/rag.py`: RAG retrieval and answer pipeline
- `src/backend/app/agent.py`: multi-step agent orchestration
- `src/backend/app/db.py`: SQLite persistence
- `src/backend/app/state.py`: index lifecycle and MCP server state

## Frontend Modules

- `src/frontend/src/components/ChatPanel.tsx`: chat shell
- `src/frontend/src/components/KnowledgeBasePanel.tsx`: KB shell
- `src/frontend/src/components/chat/`: chat domain components/hooks
- `src/frontend/src/components/kb/`: knowledge base components/hooks

## Conventions

- Keep source code under `src/` only.
- Keep architecture boundary rules aligned with `docs/architecture-boundaries.md`.
- Keep runtime data and generated artifacts out of git (`data/`, `output/`, local caches).
