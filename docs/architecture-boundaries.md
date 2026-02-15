# Architecture Boundaries

## Backend Layering
- `src/backend/app/main.py` is composition only: app lifecycle, middleware, router registration, health endpoint.
- `src/backend/app/routes/*.py` owns HTTP transport concerns only and delegates domain work to app modules.
- `src/backend/app/state.py` owns process state and index lifecycle; routes call state APIs instead of duplicating state logic.
- Domain modules (`agent.py`, `rag.py`, `ingest.py`, `retrieval_eval.py`, `mcp_server.py`, `db.py`) must not import from `routes` or `main`.

## Backend Import Direction
- Allowed direction: `main -> routes -> domain/state/config/db`.
- Forbidden direction: `domain/state -> routes/main`, `routes -> main`, `route -> route`.
- Shared models for transport live in `src/backend/app/schemas.py`.

## Frontend Layering
- `src/frontend/src/components/ChatPanel.tsx` and `src/frontend/src/components/KnowledgeBasePanel.tsx` are orchestration shells.
- Chat domain modules live under `src/frontend/src/components/chat/`.
- Knowledge-base domain modules live under `src/frontend/src/components/kb/`.
- Feature logic goes to hooks/modules first; panel files stay small and focused.

## Enforced Rules
- Backend boundary checks are enforced by `src/backend/tests/test_architecture_boundaries.py`.
- Frontend shell-size guardrails are enforced by the same test to prevent panel regression into giant files.

## Change Policy
- Any boundary-rule change must update this document and the architecture boundary tests in the same pull request.
