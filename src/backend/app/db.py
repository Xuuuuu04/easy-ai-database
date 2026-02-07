"""SQLite 持久化辅助方法（文档、对话、Agent 步骤）。"""

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import settings


def get_conn() -> sqlite3.Connection:
    """打开支持按列名访问的 SQLite 连接。

    Returns:
        配置完成的 SQLite 连接。
    """
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """创建数据库表（若不存在）。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            page INTEGER,
            start_offset INTEGER,
            end_offset INTEGER,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            tool TEXT NOT NULL,
            input TEXT NOT NULL,
            output TEXT NOT NULL,
            citations TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
        """
    )
    conn.commit()
    conn.close()


def insert_document(title: str, source_type: str, source_ref: str) -> int:
    """插入文档记录并返回 id。

    Args:
        title: 文档展示标题。
        source_type: 来源类型（"file" 或 "url"）。
        source_ref: 原始来源引用。

    Returns:
        新插入的文档 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents(title, source_type, source_ref) VALUES (?, ?, ?)",
        (title, source_type, source_ref),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(doc_id)


def insert_chunks(document_id: int, chunks: Iterable[dict[str, Any]]) -> None:
    """插入文档的分块内容。

    Args:
        document_id: 文档 id。
        chunks: 分块字典迭代器。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO chunks(document_id, content, page, start_offset, end_offset)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                document_id,
                chunk["content"],
                chunk.get("page"),
                chunk.get("start_offset"),
                chunk.get("end_offset"),
            )
            for chunk in chunks
        ],
    )
    conn.commit()
    conn.close()


def list_documents() -> list[dict[str, Any]]:
    """按倒序列出所有文档。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents ORDER BY id DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def delete_document(doc_id: int) -> None:
    """删除文档及其分块。

    Args:
        doc_id: 要删除的文档 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


def create_chat(title: Optional[str] = None) -> int:
    """创建对话会话并返回 id。

    Args:
        title: 可选的对话标题。

    Returns:
        新插入的对话 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats(title) VALUES (?)", (title,))
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(chat_id)


def add_message(chat_id: int, role: str, content: str) -> int:
    """向对话追加一条消息。

    Args:
        chat_id: 对话 id。
        role: 角色（"user" 或 "assistant"）。
        content: 消息内容。

    Returns:
        新插入的消息 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages(chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content),
    )
    msg_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(msg_id)


def list_chats() -> list[dict[str, Any]]:
    """按倒序列出对话会话。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chats ORDER BY id DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_chat(chat_id: int) -> Optional[dict[str, Any]]:
    """获取对话及其消息和 Agent 步骤。

    Args:
        chat_id: 要获取的对话 id。

    Returns:
        对话负载，不存在则返回 None。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
    chat = cur.fetchone()
    if not chat:
        conn.close()
        return None
    cur.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    messages = [dict(row) for row in cur.fetchall()]
    cur.execute(
        "SELECT * FROM agent_steps WHERE chat_id = ? ORDER BY step_index ASC",
        (chat_id,),
    )
    steps = [dict(row) for row in cur.fetchall()]
    conn.close()
    return {"chat": dict(chat), "messages": messages, "agent_steps": steps}


def add_agent_step(
    chat_id: int,
    step_index: int,
    tool: str,
    input_text: str,
    output_text: str,
    citations: Optional[str] = None,
) -> int:
    """记录一次 Agent 工具调用步骤。

    Args:
        chat_id: 对话 id。
        step_index: 步骤序号。
        tool: 工具名称。
        input_text: 序列化后的输入。
        output_text: 序列化后的输出。
        citations: 可选的引用信息。

    Returns:
        新插入的步骤 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agent_steps(chat_id, step_index, tool, input, output, citations)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chat_id, step_index, tool, input_text, output_text, citations),
    )
    step_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(step_id)


def search_chunks(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """按子串出现次数进行简单检索。

    Args:
        query: 查询字符串。
        limit: 最大返回数量。

    Returns:
        匹配的分块记录列表。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT content, page FROM chunks")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    scored = []
    for row in rows:
        content = row["content"]
        score = content.count(query)
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:limit]]


def list_chunks(limit: int = 6) -> list[dict[str, Any]]:
    """返回最早的分块用于兜底回答。

    Args:
        limit: 最大返回数量。

    Returns:
        分块记录列表。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT content, page FROM chunks ORDER BY id ASC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows
