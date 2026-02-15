from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_APP = REPO_ROOT / "src" / "backend" / "app"


def _import_from_entries(file_path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    entries: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            entries.append((node.level, module))
    return entries


def test_main_keeps_composition_role_only() -> None:
    main_file = BACKEND_APP / "main.py"
    entries = _import_from_entries(main_file)

    imported_modules = {
        module.split(".")[0] for level, module in entries if level > 0 and module
    }
    forbidden = {
        "agent",
        "rag",
        "ingest",
        "retrieval_eval",
        "mcp_server",
        "hybrid_search",
    }
    assert imported_modules.isdisjoint(forbidden)


def test_routes_do_not_depend_on_main_or_other_routes() -> None:
    routes_dir = BACKEND_APP / "routes"
    for route_file in routes_dir.glob("*.py"):
        if route_file.name == "__init__.py":
            continue
        entries = _import_from_entries(route_file)
        for level, module in entries:
            if level == 0:
                continue
            assert not (level >= 1 and module == "main")
            assert not (level >= 2 and module.startswith("routes"))


def test_state_does_not_depend_on_routes() -> None:
    entries = _import_from_entries(BACKEND_APP / "state.py")
    for level, module in entries:
        if level == 0:
            continue
        assert not (level >= 1 and module == "routes")


def test_frontend_panels_remain_shell_size() -> None:
    chat_panel = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ChatPanel.tsx"
    kb_panel = (
        REPO_ROOT / "src" / "frontend" / "src" / "components" / "KnowledgeBasePanel.tsx"
    )

    chat_lines = len(chat_panel.read_text(encoding="utf-8").splitlines())
    kb_lines = len(kb_panel.read_text(encoding="utf-8").splitlines())

    assert chat_lines <= 220
    assert kb_lines <= 220
