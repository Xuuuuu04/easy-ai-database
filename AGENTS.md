# Repository Guidelines

## Project Structure & Module Organization
- `src/backend/`: FastAPI service for ingestion, RAG, agent orchestration, and storage APIs.
- `src/frontend/`: React (Vite) web UI for chat, knowledge base, and settings.
- `data/`: Local-only persisted data (SQLite, FAISS indexes, uploads, chat history).
- `docs/`: Architecture notes, API references, and operational runbooks.
- `scripts/`: Developer utilities (e.g., bootstrap, maintenance, export/import).

## Build, Test, and Development Commands
- `docker compose up --build`: One-click local stack (preferred for demos).
- `./scripts/run.sh`: Local bootstrap and dev run without Docker.
- `cd src/backend && uvicorn app.main:app --reload`: Backend dev server.
- `cd src/frontend && npm run dev`: Frontend dev server.
- `python3 -m pytest -q`: Backend unit/API tests.
- `cd src/frontend && npm run test`: Frontend tests.

## Coding Style & Naming Conventions
- Python: 4 spaces, type hints preferred, format with `black`, lint with `ruff`.
- TypeScript/React: 2 spaces, lint with `eslint`, format with `prettier`.
- Filenames: `snake_case` for Python modules, `kebab-case` for React routes/pages.
- HTTP routes are mounted at root paths (e.g., `/chat/agent`, `/kb/documents`) in the backend.

## Testing Guidelines
- Backend tests live in `src/backend/tests/` and follow `test_*.py` naming.
- Frontend tests live in `src/frontend/src/` with `*.test.tsx` naming.
- Aim for coverage on ingestion, retrieval, and agent flows; add smoke tests for
  end-to-end import → query → cite → history persistence.

## Architecture Boundaries
- Boundary rules and dependency direction live in `docs/architecture-boundaries.md`.
- Keep `src/backend/app/main.py` as composition-only; put transport logic in `src/backend/app/routes/`.
- Keep `src/frontend/src/components/ChatPanel.tsx` and `src/frontend/src/components/KnowledgeBasePanel.tsx` as orchestration shells.

## Commit & Pull Request Guidelines
- Commit messages: `type(scope): summary` (e.g., `feat(ingest): add url loader`).
- PRs should include: description, linked issue (if any), test commands run, and
  screenshots/GIFs for UI changes.

## Security & Configuration Notes
- All data is local; do not log document content or API keys.
- Store model endpoint config in `.env` (never commit secrets).
- Default data path is `./data/`; avoid hardcoding absolute paths.
