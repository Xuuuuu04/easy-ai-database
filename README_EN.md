# easy-ai-database

A lightweight local AI RAG knowledge base with document/web ingestion, grounded QA, multi-step agent retrieval, and MCP integration.

## Features
- Local-first storage: SQLite + FAISS in `./data`
- Ingestion: PDF / DOCX / TXT and URL import
- Retrieval APIs: `/chat/rag`, `/chat/agent`, `/retrieve`
- MCP endpoint: `/mcp/v1` for external AI clients
- Settings UI: edit root `.env`, toggle MCP tools, and generate install commands

## Quick Start

### Docker
```bash
cp .env.example .env
docker compose up --build
```

### Local script
```bash
cp .env.example .env
./scripts/run.sh
```

Default frontend: `http://localhost:5173`  
Default backend: `http://localhost:8000`

## Development Commands
```bash
python3 -m pytest -q
cd src/frontend && npm run test
cd src/frontend && npm run build
```

## Project Layout
- `src/backend/`: FastAPI backend
- `src/frontend/`: React frontend
- `data/`: local database/index data
- `docs/`: architecture and operations docs
- `scripts/`: developer scripts

## MCP Installation
Set `DEPLOYMENT_URL` in the web settings panel and save. The UI will generate ready-to-run MCP install commands for Claude Code and CodeX.
