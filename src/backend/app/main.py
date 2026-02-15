from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

FastApiMCP = None
try:
    _fastapi_mcp_mod = __import__("fastapi_mcp", fromlist=["FastApiMCP"])
    FastApiMCP = getattr(_fastapi_mcp_mod, "FastApiMCP", None)
except Exception:
    FastApiMCP = None

from .config import settings
from .db import init_db
from .routes import chat, eval, kb, mcp, settings as settings_routes
from .state import drop_kb_index, get_or_create_kb_index, indexes


@asynccontextmanager
async def lifespan(_: FastAPI):
    indexes.clear()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    if not settings.mock_mode:
        get_or_create_kb_index(1)
    yield


app = FastAPI(title="easy-ai-database", lifespan=lifespan)

if FastApiMCP is not None and settings.mcp_enable_legacy_fastapi_mount:
    mcp_server = FastApiMCP(app, name=settings.mcp_server_name)
    mcp_server.mount_http()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(kb.router)
app.include_router(chat.router)
app.include_router(eval.router)
app.include_router(mcp.router)
app.include_router(settings_routes.router)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "mock_mode": settings.mock_mode}


def _get_or_create_kb_index(kb_id: int):
    return get_or_create_kb_index(kb_id)


def _drop_kb_index(kb_id: int) -> None:
    drop_kb_index(kb_id)
