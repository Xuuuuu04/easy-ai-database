from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _emit(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    app_config = importlib.import_module("backend.app.config")
    app_db = importlib.import_module("backend.app.db")
    app_main = importlib.import_module("backend.app.main")
    app_mcp = importlib.import_module("backend.app.mcp_server")

    settings = app_config.settings
    init_db = app_db.init_db
    get_or_create_kb_index = app_main._get_or_create_kb_index
    drop_kb_index = app_main._drop_kb_index
    knowledge_mcp_server = app_mcp.KnowledgeBaseMCPServer

    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    if not settings.mock_mode:
        get_or_create_kb_index(1)

    server = knowledge_mcp_server(
        index_getter=get_or_create_kb_index,
        index_dropper=drop_kb_index,
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception:
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
            )
            continue

        if isinstance(request, list):
            responses = []
            for item in request:
                response = server.handle_message(item)
                if response is not None:
                    responses.append(response)
            if responses:
                _emit(responses)
            continue

        response = server.handle_message(request)
        if response is not None:
            _emit(response)


if __name__ == "__main__":
    main()
