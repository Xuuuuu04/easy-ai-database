# Security Policy

## Supported Versions

This project follows a rolling support model on the default branch.
Security fixes are applied to the latest code first.

## Reporting a Vulnerability

Please do **not** open public issues for sensitive vulnerabilities.

Report privately with:

- affected component/file
- reproduction steps / proof of concept
- impact assessment
- suggested fix (if available)

Maintainers will acknowledge receipt and coordinate disclosure/fix timing.

## Security Notes

- Never commit API keys, tokens, or private data.
- `.env` is local-only and ignored by git.
- Runtime data under `data/`, `src/data/`, and `src/backend/data/` is local-only.
