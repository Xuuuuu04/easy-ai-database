import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi.testclient import TestClient


def create_app(tmpdir: str):
    os.environ["DATA_DIR"] = tmpdir
    os.environ["DB_PATH"] = str(Path(tmpdir) / "app.db")
    os.environ["INDEX_DIR"] = str(Path(tmpdir) / "index")
    os.environ["MOCK_MODE"] = "1"

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    app_config = importlib.import_module("backend.app.config")
    app_db = importlib.import_module("backend.app.db")
    app_main = importlib.import_module("backend.app.main")

    importlib.reload(app_config)
    importlib.reload(app_db)
    importlib.reload(app_main)
    return app_main.app


def _mcp_request(
    client: TestClient, payload: dict[str, Any], token: Optional[str] = None
):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/mcp/v1", json=payload, headers=headers)


def test_mcp_initialize_and_tools_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            init_resp = _mcp_request(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "1.0"},
                    },
                },
            )
            assert init_resp.status_code == 200
            payload = init_resp.json()["result"]
            assert payload["protocolVersion"] == "2025-11-25"
            assert payload["capabilities"]["tools"]

            tools_resp = _mcp_request(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                },
            )
            assert tools_resp.status_code == 200
            tools = tools_resp.json()["result"]["tools"]
            names = {item["name"] for item in tools}
            assert "kb.retrieve" in names
            assert "kb.ingest.file_path" in names


def test_mcp_tools_call_retrieve_and_resources():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            doc_path = Path(tmpdir) / "mcp-source.txt"
            doc_path.write_text(
                "Python supports generators and async programming.", encoding="utf-8"
            )
            with open(doc_path, "rb") as file_handle:
                ingest = client.post(
                    "/ingest/file",
                    files={"file": ("mcp-source.txt", file_handle, "text/plain")},
                )
            assert ingest.status_code == 200

            retrieve_resp = _mcp_request(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "kb.retrieve",
                        "arguments": {"question": "python", "kb_id": 1, "top_k": 3},
                    },
                },
            )
            assert retrieve_resp.status_code == 200
            retrieve_payload = retrieve_resp.json()["result"]
            assert retrieve_payload["isError"] is False
            structured = retrieve_payload["structuredContent"]
            assert structured["kb_id"] == 1
            assert isinstance(structured.get("hits"), list)

            resources_resp = _mcp_request(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "resources/read",
                    "params": {"uri": "kb://1/documents"},
                },
            )
            assert resources_resp.status_code == 200
            text_blob = resources_resp.json()["result"]["contents"][0]["text"]
            parsed = json.loads(text_blob)
            assert parsed["documents"]


def test_mcp_tool_ingest_file_path_and_prompts():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        source_path = Path(tmpdir) / "local-doc.md"
        source_path.write_text(
            "# MCP\n\nThis is a local markdown document.", encoding="utf-8"
        )

        with TestClient(app) as client:
            ingest_resp = _mcp_request(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "kb.ingest.file_path",
                        "arguments": {"kb_id": 1, "file_path": str(source_path)},
                    },
                },
            )
            assert ingest_resp.status_code == 200
            ingest_payload = ingest_resp.json()["result"]
            assert ingest_payload["isError"] is False
            assert ingest_payload["structuredContent"]["document_id"] > 0

            prompt_resp = _mcp_request(
                client,
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "prompts/get",
                    "params": {
                        "name": "kb-grounded-answer",
                        "arguments": {
                            "question": "What is MCP?",
                            "kb_id": 1,
                            "mode": "rag",
                        },
                    },
                },
            )
            assert prompt_resp.status_code == 200
            prompt_payload = prompt_resp.json()["result"]
            assert prompt_payload["messages"]


def test_mcp_auth_token_requirement():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MCP_AUTH_TOKEN"] = "test-token"
        app = create_app(tmpdir)
        with TestClient(app) as client:
            unauthorized = client.post(
                "/mcp/v1",
                json={"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
            )
            assert unauthorized.status_code == 401

            authorized = client.post(
                "/mcp/v1",
                headers={"Authorization": "Bearer test-token"},
                json={"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
            )
            assert authorized.status_code == 200
            assert authorized.json()["result"] == {}

        os.environ.pop("MCP_AUTH_TOKEN", None)


def test_mcp_stdio_launcher_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(__file__).resolve().parents[3]
        script_path = project_root / "scripts" / "mcp_stdio.py"

        env = os.environ.copy()
        env["DATA_DIR"] = tmpdir
        env["DB_PATH"] = str(Path(tmpdir) / "app.db")
        env["INDEX_DIR"] = str(Path(tmpdir) / "index")
        env["MOCK_MODE"] = "1"
        env.pop("MCP_AUTH_TOKEN", None)

        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root),
            env=env,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        }

        stdout, _ = process.communicate(
            input=json.dumps(request, ensure_ascii=False) + "\n", timeout=20
        )
        assert process.returncode == 0
        lines = [line for line in stdout.splitlines() if line.strip()]
        assert lines
        payload = json.loads(lines[0])
        assert payload["result"]["protocolVersion"] == "2025-11-25"
