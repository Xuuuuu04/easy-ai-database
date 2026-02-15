import os
import tempfile
import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def create_app(tmpdir: str):
    project_root = Path(__file__).resolve().parents[2]
    _load_env_file(project_root / ".env")

    os.environ["DATA_DIR"] = tmpdir
    os.environ["DB_PATH"] = str(Path(tmpdir) / "app.db")
    os.environ["INDEX_DIR"] = str(Path(tmpdir) / "index")
    os.environ["MOCK_MODE"] = "0"
    os.environ.setdefault("USE_ENV_PROXY", "1")

    sys.path.insert(0, str(project_root))

    from backend.app import config as app_config
    from backend.app import main

    importlib.reload(app_config)
    importlib.reload(main)
    return main.app


def test_real_rag_and_agent():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            file_path = Path(tmpdir) / "sample.txt"
            file_path.write_text(
                "项目名称是 easy-ai-database，用于离线问答。", encoding="utf-8"
            )
            with open(file_path, "rb") as f:
                resp = client.post(
                    "/ingest/file", files={"file": ("sample.txt", f, "text/plain")}
                )
            assert resp.status_code == 200

            rag = client.post("/chat/rag", json={"question": "这个项目名称是什么？"})
            assert rag.status_code == 200
            rag_data = rag.json()
            assert rag_data.get("answer")
            assert rag_data.get("citations")

            agent = client.post(
                "/chat/agent", json={"question": "总结这个项目并给出引用。"}
            )
            assert agent.status_code == 200
            agent_data = agent.json()
            assert agent_data.get("answer")
            assert agent_data.get("steps")
            assert agent_data.get("citations") is not None
