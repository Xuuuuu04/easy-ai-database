"""Mock 模式入库与对话的 API 冒烟测试。"""

import os
import tempfile
import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def create_app(tmpdir: str):
    """创建启用 mock 模式的 FastAPI 应用。

    Args:
        tmpdir: 临时数据目录。

    Returns:
        配置完成的 FastAPI 应用实例。
    """
    os.environ["DATA_DIR"] = tmpdir
    os.environ["DB_PATH"] = str(Path(tmpdir) / "app.db")
    os.environ["INDEX_DIR"] = str(Path(tmpdir) / "index")
    os.environ["MOCK_MODE"] = "1"

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from backend.app import config as app_config
    from backend.app import main

    importlib.reload(app_config)
    importlib.reload(main)
    return main.app


def test_ingest_and_rag():
    """入库文件并验证 RAG 返回结构。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            file_path = Path(tmpdir) / "sample.txt"
            file_path.write_text("这是一段测试文本，用于检索。", encoding="utf-8")
            with open(file_path, "rb") as f:
                resp = client.post("/ingest/file", files={"file": ("sample.txt", f, "text/plain")})
            assert resp.status_code == 200

            resp = client.post("/chat/rag", json={"question": "测试文本"})
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "citations" in data


def test_history():
    """验证历史对话接口可正常响应。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            resp = client.get("/chat/history")
            assert resp.status_code == 200
