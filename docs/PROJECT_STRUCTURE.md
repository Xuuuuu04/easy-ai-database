# Project Structure

Updated: 2026-02-15

## Top-level Layout
```text
.
- .env.example
- .gitignore
- AGENTS.md
- LICENSE
- README.md
- README_EN.md
- docker-compose.yml
- docs
- pytest.ini
- scripts
- src
```

## Conventions
- Keep executable/business code under src/.
- Keep docs under docs/ (or doc/ for Cangjie projects).
- Keep local runtime artifacts and secrets out of version control.
