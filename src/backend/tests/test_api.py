"""Mock 模式入库与对话的 API 冒烟测试。"""

import os
import tempfile
import importlib
import sys
import json
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

    app_config = importlib.import_module("backend.app.config")
    app_db = importlib.import_module("backend.app.db")
    main = importlib.import_module("backend.app.main")

    importlib.reload(app_config)
    importlib.reload(app_db)
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
                resp = client.post(
                    "/ingest/file", files={"file": ("sample.txt", f, "text/plain")}
                )
            assert resp.status_code == 200

            resp = client.post("/chat/rag", json={"question": "测试文本"})
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "citations" in data


def test_ingest_file_duplicate_name_requires_keep_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        file_path = Path(tmpdir) / "dup.txt"
        file_path.write_text("重复文件测试", encoding="utf-8")

        with TestClient(app) as client:
            with open(file_path, "rb") as f:
                first = client.post(
                    "/ingest/file", files={"file": ("dup.txt", f, "text/plain")}
                )
            assert first.status_code == 200

            with open(file_path, "rb") as f:
                duplicate = client.post(
                    "/ingest/file", files={"file": ("dup.txt", f, "text/plain")}
                )
            assert duplicate.status_code == 409

            with open(file_path, "rb") as f:
                keep = client.post(
                    "/ingest/file?allow_duplicate=1",
                    files={"file": ("dup.txt", f, "text/plain")},
                )
            assert keep.status_code == 200


def test_ingest_file_accepts_folder_relative_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        file_path = Path(tmpdir) / "nested.txt"
        file_path.write_text("目录上传测试", encoding="utf-8")

        with TestClient(app) as client:
            with open(file_path, "rb") as f:
                resp = client.post(
                    "/ingest/file",
                    files={"file": ("folder/sub/nested.txt", f, "text/plain")},
                )

            assert resp.status_code == 200
            docs = client.get("/kb/documents").json()
            assert any(doc["title"] == "folder/sub/nested.txt" for doc in docs)


def test_kb_preview_returns_full_content_for_file_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        file_path = Path(tmpdir) / "preview.txt"
        file_path.write_text("第一行\n第二行\n第三行", encoding="utf-8")

        with TestClient(app) as client:
            with open(file_path, "rb") as f:
                ingest = client.post(
                    "/ingest/file", files={"file": ("preview.txt", f, "text/plain")}
                )
            assert ingest.status_code == 200

            docs = client.get("/kb/documents").json()
            source = docs[0]["source_ref"]

            preview = client.get("/kb/preview", params={"source": source})
            assert preview.status_code == 200
            payload = preview.json()
            assert payload["kind"] == "file"
            assert payload["preview_type"] == "text"
            assert "第一行" in payload["content"]


def test_kb_preview_rejects_local_virtual_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            preview = client.get("/kb/preview", params={"source": "local"})
            assert preview.status_code == 400


def test_history():
    """验证历史对话接口可正常响应。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            resp = client.get("/chat/history")
            assert resp.status_code == 200


def test_rag_smalltalk_greeting_returns_conversational_reply():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            resp = client.post("/chat/rag", json={"question": "你好"})

        assert resp.status_code == 200
        data = resp.json()
        assert "你好！我是本机知识库助手" in data["answer"]
        assert data["citations"] == []
        assert "没有足够的依据" not in data["answer"]


def test_rag_smalltalk_greeting_stream_returns_conversational_reply():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            resp = client.post("/chat/rag", json={"question": "你好", "stream": True})

        assert resp.status_code == 200
        payloads = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

        done_events = [item for item in payloads if item.get("done")]
        assert done_events
        done = done_events[-1]
        assert "你好！我是本机知识库助手" in done.get("answer", "")
        assert done.get("citations") == []


def test_agent_smalltalk_greeting_returns_conversational_reply():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            resp = client.post("/chat/agent", json={"question": "你好"})

        assert resp.status_code == 200
        data = resp.json()
        assert "你好！我是本机知识库助手" in data["answer"]
        assert data["citations"] == []
        assert "没有足够的依据" not in data["answer"]


def test_agent_smalltalk_greeting_stream_returns_conversational_reply():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            resp = client.post("/chat/agent", json={"question": "你好", "stream": True})

        assert resp.status_code == 200
        payloads = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

        done_events = [item for item in payloads if item.get("done")]
        assert done_events
        done = done_events[-1]
        assert "你好！我是本机知识库助手" in done.get("answer", "")
        assert done.get("citations") == []


def test_search_chunks_empty_query_returns_empty():
    """空查询应直接返回空结果，避免无意义全表扫描。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("doc.txt", "file", "doc.txt")
        app_db.insert_chunks(doc_id, [{"content": "alpha beta", "page": 1}])

        assert app_db.search_chunks("", limit=6) == []
        assert app_db.search_chunks("   ", limit=6) == []


def test_search_chunks_non_match_returns_empty():
    """无匹配时返回空列表。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("doc.txt", "file", "doc.txt")
        app_db.insert_chunks(
            doc_id,
            [
                {"content": "python fastapi", "page": 1},
                {"content": "sqlite index", "page": 2},
            ],
        )

        assert app_db.search_chunks("golang", limit=6) == []


def test_search_chunks_orders_by_relevance_desc():
    """匹配结果按相关性降序返回。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("doc.txt", "file", "doc.txt")
        app_db.insert_chunks(
            doc_id,
            [
                {"content": "apple", "page": 1},
                {"content": "apple apple", "page": 2},
                {"content": "apple apple apple", "page": 3},
                {"content": "orange", "page": 4},
            ],
        )

        hits = app_db.search_chunks("apple", limit=2)
        assert hits == [
            {"content": "apple apple apple", "page": 3},
            {"content": "apple apple", "page": 2},
        ]


def test_rag_stream_done_payload_contains_answer_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("doc.txt", "file", "doc.txt")
        app_db.insert_chunks(
            doc_id, [{"content": "你好，这是测试知识内容。", "page": 1}]
        )

        with TestClient(app) as client:
            resp = client.post("/chat/rag", json={"question": "你好", "stream": True})

        assert resp.status_code == 200
        payloads = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

        done_events = [item for item in payloads if item.get("done")]
        assert done_events
        assert isinstance(done_events[-1].get("answer"), str)
        assert done_events[-1].get("answer")


def test_rag_stream_emits_multiple_chunks_for_long_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("stream.txt", "file", "stream.txt")
        long_content = (
            "streamtoken " + "这是一个用于验证流式输出分片行为的长文本。" * 20
        )
        app_db.insert_chunks(doc_id, [{"content": long_content, "page": 1}])

        with TestClient(app) as client:
            resp = client.post(
                "/chat/rag", json={"question": "streamtoken", "stream": True}
            )

        assert resp.status_code == 200
        payloads = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

        chunk_events = [item for item in payloads if item.get("chunk")]
        done_events = [item for item in payloads if item.get("done")]

        assert len(chunk_events) >= 2
        assert done_events


def test_agent_stream_emits_multiple_chunks_for_long_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("agent-stream.txt", "file", "agent-stream.txt")
        long_content = (
            "agenttoken " + "这是一段用于验证 Agent 流式响应分片输出的内容。" * 20
        )
        app_db.insert_chunks(doc_id, [{"content": long_content, "page": 1}])

        with TestClient(app) as client:
            resp = client.post(
                "/chat/agent", json={"question": "agenttoken", "stream": True}
            )

        assert resp.status_code == 200
        payloads = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

        chunk_events = [item for item in payloads if item.get("chunk")]
        done_events = [item for item in payloads if item.get("done")]

        assert len(chunk_events) >= 2
        assert done_events


def test_retrieval_eval_endpoint_returns_expected_metrics_for_controlled_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document("bench.txt", "file", "bench.txt")
        app_db.insert_chunks(
            doc_id,
            [
                {"content": "apple orange", "page": 1},
                {"content": "apple banana", "page": 2},
                {"content": "paris is the capital of france", "page": 3},
            ],
        )

        payload = {
            "k": 2,
            "parameter_grid": {"candidate_top_k": [1, 2], "use_reranker": [False]},
            "cases": [
                {
                    "id": "banana-second",
                    "query": "apple",
                    "relevant_snippets": ["banana"],
                },
                {
                    "id": "capital-paris",
                    "query": "capital",
                    "relevant_snippets": ["paris"],
                },
            ],
        }

        with TestClient(app) as client:
            resp = client.post("/eval/retrieval", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset"] == "request"
        assert data["k"] == 2
        assert data["config_count"] == 2
        assert len(data["results"]) == 2

        config0 = data["results"][0]["metrics"]
        config1 = data["results"][1]["metrics"]

        assert config0["precision@k"] == 0.25
        assert config0["recall@k"] == 0.5
        assert config0["hit_rate@k"] == 0.5
        assert config0["mrr@k"] == 0.5
        assert config0["ndcg@k"] == 0.5

        assert config1["precision@k"] == 0.5
        assert config1["recall@k"] == 1.0
        assert config1["hit_rate@k"] == 1.0
        assert config1["mrr@k"] == 0.75
        assert round(config1["ndcg@k"], 6) == 0.815465

        assert data["best_config"]["config_id"] == 1
        assert data["best_config"]["config"]["candidate_top_k"] == 2


def test_retrieval_eval_uses_default_benchmark_when_cases_omitted():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        app_db = importlib.import_module("backend.app.db")
        app_db.init_db()

        doc_id = app_db.insert_document(
            "default-bench.txt", "file", "default-bench.txt"
        )
        app_db.insert_chunks(
            doc_id,
            [
                {"content": "Paris is the capital city of France.", "page": 1},
                {"content": "Python is a language used for scripting.", "page": 2},
            ],
        )

        with TestClient(app) as client:
            resp = client.post("/eval/retrieval", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset"] == "default"
        assert data["k"] == 2
        assert data["config_count"] == 2
        assert data["best_config"] is not None
        assert all("metrics" in result for result in data["results"])


def test_knowledge_base_crud_and_document_isolation():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            listed = client.get("/kb")
            assert listed.status_code == 200
            defaults = listed.json()
            assert len(defaults) == 1
            assert defaults[0]["id"] == 1

            created = client.post(
                "/kb",
                json={"name": "隔离知识库", "description": "用于隔离测试"},
            )
            assert created.status_code == 200
            kb_id = created.json()["id"]

            file_a = Path(tmpdir) / "kb-a.txt"
            file_a.write_text("alpha document", encoding="utf-8")
            file_b = Path(tmpdir) / "kb-b.txt"
            file_b.write_text("beta document", encoding="utf-8")

            with open(file_a, "rb") as f:
                upload_default = client.post(
                    "/ingest/file?kb_id=1",
                    files={"file": ("kb-a.txt", f, "text/plain")},
                )
            assert upload_default.status_code == 200

            with open(file_b, "rb") as f:
                upload_isolated = client.post(
                    f"/ingest/file?kb_id={kb_id}",
                    files={"file": ("kb-b.txt", f, "text/plain")},
                )
            assert upload_isolated.status_code == 200

            docs_default = client.get("/kb/documents", params={"kb_id": 1})
            docs_isolated = client.get("/kb/documents", params={"kb_id": kb_id})
            assert docs_default.status_code == 200
            assert docs_isolated.status_code == 200

            default_titles = [doc["title"] for doc in docs_default.json()]
            isolated_titles = [doc["title"] for doc in docs_isolated.json()]
            assert "kb-a.txt" in default_titles
            assert "kb-b.txt" not in default_titles
            assert "kb-b.txt" in isolated_titles
            assert "kb-a.txt" not in isolated_titles

            delete_default = client.delete("/kb/1")
            assert delete_default.status_code == 400

            delete_created = client.delete(f"/kb/{kb_id}")
            assert delete_created.status_code == 200


def test_chat_and_history_are_scoped_by_knowledge_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            create_kb = client.post(
                "/kb",
                json={"name": "会话隔离", "description": "history scope"},
            )
            assert create_kb.status_code == 200
            kb2 = create_kb.json()["id"]

            first_chat = client.post("/chat/rag", json={"question": "你好", "kb_id": 1})
            assert first_chat.status_code == 200
            chat_id_1 = first_chat.json()["chat_id"]

            second_chat = client.post(
                "/chat/rag", json={"question": "你好", "kb_id": kb2}
            )
            assert second_chat.status_code == 200
            chat_id_2 = second_chat.json()["chat_id"]

            history_1 = client.get("/chat/history", params={"kb_id": 1})
            history_2 = client.get("/chat/history", params={"kb_id": kb2})
            assert history_1.status_code == 200
            assert history_2.status_code == 200

            history_ids_1 = [item["id"] for item in history_1.json()]
            history_ids_2 = [item["id"] for item in history_2.json()]
            assert chat_id_1 in history_ids_1
            assert chat_id_2 not in history_ids_1
            assert chat_id_2 in history_ids_2
            assert chat_id_1 not in history_ids_2

            wrong_scope = client.get(f"/chat/{chat_id_1}", params={"kb_id": kb2})
            assert wrong_scope.status_code == 404


def test_retrieve_endpoint_returns_hits_and_rewrites_for_selected_kb():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            create_kb = client.post(
                "/kb",
                json={"name": "检索KB", "description": "retrieve scope"},
            )
            assert create_kb.status_code == 200
            kb2 = create_kb.json()["id"]

            file_a = Path(tmpdir) / "scope-a.txt"
            file_a.write_text("alpha evidence only", encoding="utf-8")
            file_b = Path(tmpdir) / "scope-b.txt"
            file_b.write_text("beta evidence only", encoding="utf-8")

            with open(file_a, "rb") as f:
                resp_a = client.post(
                    "/ingest/file?kb_id=1",
                    files={"file": ("scope-a.txt", f, "text/plain")},
                )
            with open(file_b, "rb") as f:
                resp_b = client.post(
                    f"/ingest/file?kb_id={kb2}",
                    files={"file": ("scope-b.txt", f, "text/plain")},
                )
            assert resp_a.status_code == 200
            assert resp_b.status_code == 200

            retrieved = client.post(
                "/retrieve",
                json={"question": "alpha", "kb_id": 1, "top_k": 3},
            )
            assert retrieved.status_code == 200
            payload = retrieved.json()
            assert payload["kb_id"] == 1
            assert isinstance(payload["rewrites"], list)
            assert any("alpha" in (hit.get("content") or "") for hit in payload["hits"])


def test_retrieval_eval_respects_kb_id_scope():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            created = client.post(
                "/kb",
                json={"name": "评估隔离", "description": "eval scope"},
            )
            assert created.status_code == 200
            kb2 = created.json()["id"]

            file_a = Path(tmpdir) / "eval-a.txt"
            file_a.write_text("alpha special token", encoding="utf-8")
            file_b = Path(tmpdir) / "eval-b.txt"
            file_b.write_text("beta special token", encoding="utf-8")

            with open(file_a, "rb") as f:
                client.post(
                    "/ingest/file?kb_id=1",
                    files={"file": ("eval-a.txt", f, "text/plain")},
                )
            with open(file_b, "rb") as f:
                client.post(
                    f"/ingest/file?kb_id={kb2}",
                    files={"file": ("eval-b.txt", f, "text/plain")},
                )

            payload = {
                "kb_id": 1,
                "k": 2,
                "parameter_grid": {"candidate_top_k": [2], "use_reranker": [False]},
                "cases": [
                    {
                        "id": "alpha-case",
                        "query": "alpha",
                        "relevant_snippets": ["alpha"],
                    }
                ],
            }
            eval_kb1 = client.post("/eval/retrieval", json=payload)
            assert eval_kb1.status_code == 200

            payload["kb_id"] = kb2
            eval_kb2 = client.post("/eval/retrieval", json=payload)
            assert eval_kb2.status_code == 200

            precision_kb1 = eval_kb1.json()["results"][0]["metrics"]["precision@k"]
            precision_kb2 = eval_kb2.json()["results"][0]["metrics"]["precision@k"]
            assert precision_kb1 >= precision_kb2


def test_generate_retrieval_dataset_without_llm_works_in_mock_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            sample = Path(tmpdir) / "dataset.txt"
            sample.write_text(
                "仓颉语言是一门现代编程语言，强调并发、安全与工程化实践。",
                encoding="utf-8",
            )
            with open(sample, "rb") as f:
                ingest = client.post(
                    "/ingest/file?kb_id=1",
                    files={"file": ("dataset.txt", f, "text/plain")},
                )
            assert ingest.status_code == 200

            generated = client.post(
                "/eval/retrieval/generate-dataset",
                json={"kb_id": 1, "case_count": 4, "use_llm": False},
            )
            assert generated.status_code == 200
            payload = generated.json()
            assert payload["kb_id"] == 1
            assert payload["generated"] > 0
            assert isinstance(payload["cases"], list)
