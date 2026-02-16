# easy-ai-database

A local-first AI RAG knowledge base with file/web ingestion, grounded QA,
multi-step agent retrieval, and MCP integration.

## Features

- Local storage with SQLite + FAISS (`./data` by default)
- Ingestion for PDF / DOCX / TXT / URL
- Retrieval APIs: `/chat/rag`, `/chat/agent`, `/retrieve`
- MCP endpoint: `/mcp/v1` (JSON-RPC + SSE)
- Settings UI to manage key `.env` values

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

- Frontend default: `http://localhost:5173`
- Backend default: `http://localhost:8000`

## Development Commands

```bash
python3 -m pytest -q
cd src/frontend && npm run test
cd src/frontend && npm run build
```

## Project Layout

- Backend: `src/backend/`
- Frontend: `src/frontend/`
- Docs: `docs/`
- Scripts: `scripts/`

## Documentation

- Architecture boundaries: `docs/architecture-boundaries.md`
- API reference: `docs/api.md`

## Open Source Governance

- Contributing guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Changelog: `CHANGELOG.md`
