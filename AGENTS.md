# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI service for ingestion, RAG, agent orchestration, and storage APIs.
- `frontend/`: React (Vite) web UI for chat, knowledge base, and settings.
- `data/`: Local-only persisted data (SQLite, FAISS indexes, uploads, chat history).
- `docs/`: Architecture notes, API references, and operational runbooks.
- `scripts/`: Developer utilities (e.g., bootstrap, maintenance, export/import).

## Build, Test, and Development Commands
- `docker compose up --build`: One-click local stack (preferred for demos).
- `./run.sh`: Local bootstrap and dev run without Docker.
- `cd backend && uvicorn app.main:app --reload`: Backend dev server.
- `cd frontend && npm run dev`: Frontend dev server.
- `cd backend && pytest`: Backend unit/API tests.
- `cd frontend && npm run test`: Frontend tests.

## Coding Style & Naming Conventions
- Python: 4 spaces, type hints preferred, format with `black`, lint with `ruff`.
- TypeScript/React: 2 spaces, lint with `eslint`, format with `prettier`.
- Filenames: `snake_case` for Python modules, `kebab-case` for React routes/pages.
- API routes are versioned under `/api/v1` in the backend.

## Testing Guidelines
- Backend tests live in `backend/tests/` and follow `test_*.py` naming.
- Frontend tests live in `frontend/src/__tests__/` or `*.test.tsx`.
- Aim for coverage on ingestion, retrieval, and agent flows; add smoke tests for
  end-to-end import → query → cite → history persistence.

## Commit & Pull Request Guidelines
- Commit messages: `type(scope): summary` (e.g., `feat(ingest): add url loader`).
- PRs should include: description, linked issue (if any), test commands run, and
  screenshots/GIFs for UI changes.

## Security & Configuration Notes
- All data is local; do not log document content or API keys.
- Store model endpoint config in `.env` (never commit secrets).
- Default data path is `./data/`; avoid hardcoding absolute paths.
