# Runbook

## Startup Checks

- Backend health: `GET /health`
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

## Common Issues

1. Model endpoint unavailable
   - Verify `.env` values for `LLM_BASE_URL` / `EMBED_BASE_URL`.
   - Confirm local model service is running.
2. Answer rejected due to missing citations
   - Citation guard is enabled by default.
   - For debugging only, set `REQUIRE_CITATIONS=0`.
3. Retrieval/index corruption
   - Stop services.
   - Remove local index directory (`./data/index` or configured `INDEX_DIR`).
   - Re-ingest documents.

## Test/Debug Modes

- `MOCK_MODE=1`: skip vector/model calls and use lexical fallback.
- Retrieval diagnostics endpoint: `POST /retrieve`.

## Local Cleanup (safe to regenerate)

- Frontend build/cache: `src/frontend/dist`, `src/frontend/node_modules`
- Python cache: `.pytest_cache`, `.ruff_cache`, `__pycache__`
- Evaluation artifacts: `output/`
