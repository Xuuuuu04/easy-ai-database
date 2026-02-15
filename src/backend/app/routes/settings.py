from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..config import get_env_path, reload_settings, settings
from ..schemas import SettingsUpdateRequest

router = APIRouter()


def _parse_env_lines(env_path: Path) -> list[dict[str, str]]:
    if not env_path.exists():
        return []

    rows: list[dict[str, str]] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            rows.append({"type": "blank", "raw": ""})
            continue
        if stripped.startswith("#"):
            rows.append({"type": "comment", "raw": line})
            continue
        if "=" not in line:
            rows.append({"type": "comment", "raw": line})
            continue

        key, value = line.split("=", 1)
        rows.append(
            {
                "type": "pair",
                "key": key.strip(),
                "value": value.strip(),
            }
        )
    return rows


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    if not base:
        return ""
    if base.startswith("http://") or base.startswith("https://"):
        return base
    return f"https://{base}"


def _build_commands(base_url: str, mcp_token: str) -> dict[str, str]:
    endpoint = f"{base_url}/mcp/v1" if base_url else ""
    token_arg = ""
    if mcp_token.strip():
        token_arg = ' --header "Authorization: Bearer <MCP_AUTH_TOKEN>"'

    if not endpoint:
        return {
            "claude_code": "",
            "codex": "",
        }

    return {
        "claude_code": (
            f"claude mcp add easy-ai-database --transport http {endpoint}{token_arg}"
        ),
        "codex": (
            f"codex mcp add easy-ai-database --transport http {endpoint}{token_arg}"
        ),
    }


def _build_settings_payload(request: Request) -> dict[str, Any]:
    env_path = get_env_path()
    rows = _parse_env_lines(env_path)
    variables: dict[str, str] = {}
    for row in rows:
        if row.get("type") == "pair":
            variables[row["key"]] = row["value"]

    fallback_origin = str(request.base_url).rstrip("/")
    deployment_url = _normalize_base_url(
        variables.get("DEPLOYMENT_URL") or settings.deployment_url or fallback_origin
    )
    mcp_token = variables.get("MCP_AUTH_TOKEN", settings.mcp_auth_token)
    mcp_enabled = _is_true(variables.get("MCP_TOOLS_ENABLED", "1"))

    commands = _build_commands(deployment_url, mcp_token)
    return {
        "env_file": str(env_path),
        "variables": variables,
        "mcp_tools_enabled": mcp_enabled,
        "deployment_url": deployment_url,
        "mcp_endpoint": f"{deployment_url}/mcp/v1" if deployment_url else "",
        "mcp_commands": commands,
    }


def _write_env_variables(env_path: Path, updates: dict[str, str]) -> None:
    rows = _parse_env_lines(env_path)
    pending = {k: str(v) for k, v in updates.items()}

    next_lines: list[str] = []
    for row in rows:
        kind = row.get("type")
        if kind == "pair":
            key = row["key"]
            value = pending.pop(key, row["value"])
            next_lines.append(f"{key}={value}")
        elif kind == "comment":
            next_lines.append(row.get("raw", ""))
        else:
            next_lines.append("")

    if pending:
        if next_lines and next_lines[-1].strip():
            next_lines.append("")
        for key in sorted(pending.keys()):
            next_lines.append(f"{key}={pending[key]}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


@router.get("/settings/env")
def get_settings_env(request: Request) -> dict[str, Any]:
    return _build_settings_payload(request)


@router.put("/settings/env")
def update_settings_env(request: Request, req: SettingsUpdateRequest) -> dict[str, Any]:
    if not req.variables:
        raise HTTPException(status_code=400, detail="No variables provided")

    env_path = get_env_path()
    _write_env_variables(env_path, req.variables)
    reload_settings()
    return _build_settings_payload(request)
