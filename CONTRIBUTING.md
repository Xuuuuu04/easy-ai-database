# Contributing

Thanks for contributing to `easy-ai-database`.

## Development Setup

1. Copy env template:

```bash
cp .env.example .env
```

2. Run locally (choose one):

```bash
docker compose up --build
```

or

```bash
./scripts/run.sh
```

## Project Rules

- Follow module boundaries in `docs/architecture-boundaries.md`.
- Keep `src/backend/app/main.py` composition-only.
- Keep `src/frontend/src/components/ChatPanel.tsx` and
  `src/frontend/src/components/KnowledgeBasePanel.tsx` as orchestration shells.
- Do not commit secrets or local runtime data.

## Code Style

- Python: `black`, `ruff`, type hints preferred.
- Frontend: `eslint`, `prettier` compatible formatting.

## Tests

Run before opening a PR:

```bash
python3 -m pytest -q
cd src/frontend && npm run test && npm run build
```

## Commit & PR

- Use Conventional Commits style:
  - `feat(scope): ...`
  - `fix(scope): ...`
  - `docs(scope): ...`
- PR should include:
  - what changed
  - why
  - verification commands/results
  - screenshots for UI changes
