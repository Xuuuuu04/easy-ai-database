from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


CASES_PATH = Path(__file__).resolve().parent / "fixtures" / "rag_agent_cases.json"
USE_MOCK = os.getenv("RAG_AGENT_TEST_USE_MOCK", "0") == "1"


def _load_cases() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def create_app(tmpdir: str):
    os.environ["DATA_DIR"] = tmpdir
    os.environ["DB_PATH"] = str(Path(tmpdir) / "app.db")
    os.environ["INDEX_DIR"] = str(Path(tmpdir) / "index")
    os.environ["MOCK_MODE"] = "1" if USE_MOCK else "0"
    os.environ["ENABLE_MULTI_TURN"] = "0"
    os.environ["REQUIRE_CITATIONS"] = "1"
    os.environ["MIN_CITATIONS"] = "1"

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


def _seed_documents(client: TestClient, tmpdir: str, cases: dict[str, Any]) -> None:
    for doc in cases["documents"]:
        relative_name = str(doc["filename"])
        doc_path = Path(tmpdir) / relative_name
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(str(doc["content"]), encoding="utf-8")

        with doc_path.open("rb") as file_handle:
            response = client.post(
                "/ingest/file",
                files={"file": (relative_name, file_handle, "text/plain")},
            )

        assert response.status_code == 200, (
            f"failed to seed document {relative_name}: {response.status_code} "
            f"{response.text}"
        )


def _assert_answer_contains_any(
    *,
    case_id: str,
    answer: str,
    citations: list[dict[str, Any]] | list[Any],
    expected_tokens: list[str],
    failures: list[str],
) -> None:
    if not expected_tokens:
        return
    citation_snippets = " ".join(
        str((item or {}).get("snippet") or "")
        for item in citations
        if isinstance(item, dict)
    )
    haystack = f"{answer}\n{citation_snippets}"
    if any(token in haystack for token in expected_tokens):
        return
    failures.append(
        f"{case_id}: expected any of {expected_tokens} in answer/citations, got answer={answer!r}"
    )


def _parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def test_rag_dataset_cases() -> None:
    cases = _load_cases()
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            _seed_documents(client, tmpdir, cases)
            failures: list[str] = []

            for case in cases["rag_cases"]:
                response = client.post(
                    "/chat/rag",
                    json={"question": case["question"], "kb_id": 1},
                )
                if response.status_code != 200:
                    failures.append(
                        f"{case['id']}: HTTP {response.status_code}, body={response.text}"
                    )
                    continue

                payload = response.json()
                answer = str(payload.get("answer") or "")
                citations = payload.get("citations") or []
                min_citations = int(case.get("min_citations", 0))

                _assert_answer_contains_any(
                    case_id=case["id"],
                    answer=answer,
                    citations=citations,
                    expected_tokens=[str(item) for item in case.get("expect_any", [])],
                    failures=failures,
                )

                if len(citations) < min_citations:
                    failures.append(
                        f"{case['id']}: expected citations>={min_citations}, got {len(citations)}"
                    )

            assert not failures, "\n".join(failures)


def test_agent_dataset_cases() -> None:
    cases = _load_cases()
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            _seed_documents(client, tmpdir, cases)
            failures: list[str] = []

            for case in cases["agent_cases"]:
                response = client.post(
                    "/chat/agent",
                    json={"question": case["question"], "kb_id": 1},
                )
                if response.status_code != 200:
                    failures.append(
                        f"{case['id']}: HTTP {response.status_code}, body={response.text}"
                    )
                    continue

                payload = response.json()
                answer = str(payload.get("answer") or "")
                citations = payload.get("citations") or []
                steps = payload.get("steps") or []

                _assert_answer_contains_any(
                    case_id=case["id"],
                    answer=answer,
                    citations=citations,
                    expected_tokens=[str(item) for item in case.get("expect_any", [])],
                    failures=failures,
                )

                min_citations = int(case.get("min_citations", 0))
                if len(citations) < min_citations:
                    failures.append(
                        f"{case['id']}: expected citations>={min_citations}, got {len(citations)}"
                    )

                min_steps = int(case.get("min_steps", 0))
                if len(steps) < min_steps:
                    failures.append(
                        f"{case['id']}: expected steps>={min_steps}, got {len(steps)}"
                    )

            assert not failures, "\n".join(failures)


def test_retrieve_dataset_cases() -> None:
    cases = _load_cases()
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            _seed_documents(client, tmpdir, cases)
            failures: list[str] = []

            for case in cases["retrieve_cases"]:
                response = client.post(
                    "/retrieve",
                    json={"question": case["question"], "kb_id": 1, "top_k": 4},
                )
                if response.status_code != 200:
                    failures.append(
                        f"{case['id']}: HTTP {response.status_code}, body={response.text}"
                    )
                    continue

                payload = response.json()
                hits = payload.get("hits") or []
                diagnostics = payload.get("diagnostics") or {}
                expected = str(case.get("expect_hit_contains") or "")

                if not hits:
                    failures.append(f"{case['id']}: expected non-empty hits")
                    continue

                top_hit_content = str(hits[0].get("content") or hits[0].get("snippet") or "")
                if expected and expected not in top_hit_content:
                    failures.append(
                        f"{case['id']}: expected top hit to include {expected!r}, got {top_hit_content!r}"
                    )

                if "candidate_count" not in diagnostics:
                    failures.append(f"{case['id']}: diagnostics missing candidate_count")

            assert not failures, "\n".join(failures)


def test_streaming_dataset_cases() -> None:
    cases = _load_cases()
    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            _seed_documents(client, tmpdir, cases)
            failures: list[str] = []

            for case in cases["stream_cases"]:
                mode = str(case["mode"])
                endpoint = "/chat/rag" if mode == "rag" else "/chat/agent"
                response = client.post(
                    endpoint,
                    json={"question": case["question"], "kb_id": 1, "stream": True},
                )
                if response.status_code != 200:
                    failures.append(
                        f"{case['id']}: HTTP {response.status_code}, body={response.text}"
                    )
                    continue

                payloads = _parse_sse_payloads(response.text)
                if not payloads:
                    failures.append(f"{case['id']}: no SSE payload emitted")
                    continue

                if mode == "rag":
                    done_payloads = [item for item in payloads if item.get("done")]
                    if not done_payloads:
                        failures.append(f"{case['id']}: rag stream missing done payload")
                        continue
                    done = done_payloads[-1]
                    answer = str(done.get("answer") or "")
                    citations = done.get("citations") or []
                else:
                    event_names = {
                        str(item.get("event") or "")
                        for item in payloads
                        if item.get("event")
                    }
                    required_events = {
                        "agent_research_start",
                        "agent_summary_start",
                        "agent_done",
                    }
                    missing_events = sorted(required_events - event_names)
                    if missing_events:
                        failures.append(
                            f"{case['id']}: missing agent stream events {missing_events}"
                        )
                    done_payloads = [
                        item for item in payloads if item.get("event") == "agent_done"
                    ]
                    if not done_payloads:
                        failures.append(f"{case['id']}: agent stream missing done payload")
                        continue
                    answer = str(done_payloads[-1].get("answer") or "")
                    citations = done_payloads[-1].get("citations") or []

                _assert_answer_contains_any(
                    case_id=case["id"],
                    answer=answer,
                    citations=citations,
                    expected_tokens=[str(item) for item in case.get("expect_any", [])],
                    failures=failures,
                )

            assert not failures, "\n".join(failures)


def test_multi_turn_history_persistence() -> None:
    cases = _load_cases()
    first_question = str(cases["multi_turn_case"]["first_question"])
    second_question = str(cases["multi_turn_case"]["second_question"])

    with tempfile.TemporaryDirectory() as tmpdir:
        app = create_app(tmpdir)
        with TestClient(app) as client:
            _seed_documents(client, tmpdir, cases)

            first_resp = client.post(
                "/chat/rag",
                json={"question": first_question, "kb_id": 1},
            )
            assert first_resp.status_code == 200
            first_payload = first_resp.json()
            chat_id = first_payload.get("chat_id")
            assert isinstance(chat_id, int) and chat_id > 0

            second_resp = client.post(
                "/chat/agent",
                json={"question": second_question, "chat_id": chat_id, "kb_id": 1},
            )
            assert second_resp.status_code == 200
            second_payload = second_resp.json()
            assert "AGENT_TOOL_SEQUENCE" in str(second_payload.get("answer") or "")

            detail_resp = client.get(f"/chat/{chat_id}", params={"kb_id": 1})
            assert detail_resp.status_code == 200
            detail_payload = detail_resp.json()

            messages = detail_payload.get("messages") or []
            agent_steps = detail_payload.get("agent_steps") or []
            assert len(messages) >= 4
            assert any(msg.get("role") == "assistant" for msg in messages)
            assert len(agent_steps) >= 1
