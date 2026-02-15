"""SQLite 持久化辅助方法（文档、对话、Agent 步骤）。"""

import re
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
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            slug TEXT NOT NULL UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id INTEGER NOT NULL DEFAULT 1,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id)
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
            kb_id INTEGER NOT NULL DEFAULT 1,
            title TEXT,
            summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id)
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
    _migrate_existing_schema(cur)
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
        ON chunks(document_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_kb_id
        ON documents(kb_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chats_kb_id
        ON chats(kb_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id
        ON messages(chat_id, id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_steps_chat_id_step_index
        ON agent_steps(chat_id, step_index)
        """
    )
    _ensure_default_knowledge_base(cur)
    conn.commit()
    conn.close()


def _table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cur.fetchall()}


def _migrate_existing_schema(cur: sqlite3.Cursor) -> None:
    document_columns = _table_columns(cur, "documents")
    if "kb_id" not in document_columns:
        cur.execute("ALTER TABLE documents ADD COLUMN kb_id INTEGER NOT NULL DEFAULT 1")

    chat_columns = _table_columns(cur, "chats")
    if "kb_id" not in chat_columns:
        cur.execute("ALTER TABLE chats ADD COLUMN kb_id INTEGER NOT NULL DEFAULT 1")


def _slugify_knowledge_base_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", name.strip().lower())
    slug = slug.strip("-")
    return slug or "knowledge-base"


def _ensure_default_knowledge_base(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT 1 FROM knowledge_bases WHERE id = 1")
    if cur.fetchone() is None:
        cur.execute(
            """
            INSERT INTO knowledge_bases(id, name, description, slug)
            VALUES (1, '默认知识库', '系统默认知识库', 'default')
            """
        )


def list_knowledge_bases() -> list[dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            kb.id,
            kb.name,
            kb.description,
            kb.slug,
            kb.created_at,
            COUNT(d.id) AS document_count
        FROM knowledge_bases kb
        LEFT JOIN documents d ON d.kb_id = kb.id
        GROUP BY kb.id
        ORDER BY kb.id ASC
        """
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_knowledge_base(kb_id: int) -> Optional[dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_knowledge_base(name: str, description: str = "") -> dict[str, Any]:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("Knowledge base name is required")

    conn = get_conn()
    cur = conn.cursor()
    base_slug = _slugify_knowledge_base_name(cleaned_name)
    slug = base_slug
    suffix = 1
    while True:
        cur.execute("SELECT 1 FROM knowledge_bases WHERE slug = ?", (slug,))
        if cur.fetchone() is None:
            break
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    cur.execute(
        """
        INSERT INTO knowledge_bases(name, description, slug)
        VALUES (?, ?, ?)
        """,
        (cleaned_name, description.strip(), slug),
    )
    kb_id = int(cur.lastrowid or 0)
    conn.commit()
    conn.close()
    created = get_knowledge_base(kb_id)
    if created is None:
        raise RuntimeError("Failed to create knowledge base")
    return created


def ensure_knowledge_base_exists(kb_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM knowledge_bases WHERE id = ?", (kb_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def delete_knowledge_base(kb_id: int) -> None:
    if kb_id == 1:
        raise ValueError("Default knowledge base cannot be deleted")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE kb_id = ?", (kb_id,))
    doc_ids = [int(row[0]) for row in cur.fetchall()]

    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        cur.execute(
            f"DELETE FROM chunks WHERE document_id IN ({placeholders})",
            tuple(doc_ids),
        )
        cur.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})",
            tuple(doc_ids),
        )

    cur.execute("SELECT id FROM chats WHERE kb_id = ?", (kb_id,))
    chat_ids = [int(row[0]) for row in cur.fetchall()]
    if chat_ids:
        placeholders = ",".join("?" for _ in chat_ids)
        cur.execute(
            f"DELETE FROM messages WHERE chat_id IN ({placeholders})", tuple(chat_ids)
        )
        cur.execute(
            f"DELETE FROM agent_steps WHERE chat_id IN ({placeholders})",
            tuple(chat_ids),
        )
        cur.execute(f"DELETE FROM chats WHERE id IN ({placeholders})", tuple(chat_ids))

    cur.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
    conn.commit()
    conn.close()


def insert_document(
    title: str, source_type: str, source_ref: str, kb_id: int = 1
) -> int:
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
        "INSERT INTO documents(kb_id, title, source_type, source_ref) VALUES (?, ?, ?, ?)",
        (kb_id, title, source_type, source_ref),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(doc_id) if doc_id is not None else 0


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


def replace_document_chunks(document_id: int, chunks: Iterable[dict[str, Any]]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
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


def list_documents(kb_id: Optional[int] = 1) -> list[dict[str, Any]]:
    """按倒序列出所有文档。"""
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute("SELECT * FROM documents ORDER BY id DESC")
    else:
        cur.execute(
            "SELECT * FROM documents WHERE kb_id = ? ORDER BY id DESC", (kb_id,)
        )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_documents_by_ids(doc_ids: list[int], kb_id: int) -> list[dict[str, Any]]:
    normalized_ids = sorted({int(doc_id) for doc_id in doc_ids if int(doc_id) > 0})
    if not normalized_ids:
        return []

    placeholders = ",".join("?" for _ in normalized_ids)
    params = tuple(normalized_ids) + (kb_id,)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"SELECT * FROM documents WHERE id IN ({placeholders}) AND kb_id = ? ORDER BY id ASC",
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def document_title_exists(title: str, kb_id: int = 1) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM documents WHERE kb_id = ? AND LOWER(title) = LOWER(?) LIMIT 1",
        (kb_id, title),
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def delete_document(doc_id: int, kb_id: Optional[int] = None) -> None:
    """删除文档及其分块。

    Args:
        doc_id: 要删除的文档 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    else:
        cur.execute(
            "DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE id = ? AND kb_id = ?)",
            (doc_id, kb_id),
        )
        cur.execute("DELETE FROM documents WHERE id = ? AND kb_id = ?", (doc_id, kb_id))
    conn.commit()
    conn.close()


def batch_delete_documents(doc_ids: list[int], kb_id: int) -> int:
    normalized_ids = sorted({int(doc_id) for doc_id in doc_ids if int(doc_id) > 0})
    if not normalized_ids:
        return 0

    placeholders = ",".join("?" for _ in normalized_ids)
    params = tuple(normalized_ids) + (kb_id,)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        DELETE FROM chunks
        WHERE document_id IN (
            SELECT id FROM documents
            WHERE id IN ({placeholders}) AND kb_id = ?
        )
        """,
        params,
    )
    cur.execute(
        f"DELETE FROM documents WHERE id IN ({placeholders}) AND kb_id = ?",
        params,
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return max(0, deleted)


def create_chat(title: Optional[str] = None, kb_id: int = 1) -> int:
    """创建对话会话并返回 id。

    Args:
        title: 可选的对话标题。

    Returns:
        新插入的对话 id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats(kb_id, title) VALUES (?, ?)", (kb_id, title))
    chat_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(chat_id) if chat_id is not None else 0


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
    return int(msg_id) if msg_id is not None else 0


def list_chats(kb_id: Optional[int] = 1) -> list[dict[str, Any]]:
    """按倒序列出对话会话。"""
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute("SELECT * FROM chats ORDER BY id DESC")
    else:
        cur.execute("SELECT * FROM chats WHERE kb_id = ? ORDER BY id DESC", (kb_id,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_chat(chat_id: int, kb_id: Optional[int] = None) -> Optional[dict[str, Any]]:
    """获取对话及其消息和 Agent 步骤。

    Args:
        chat_id: 要获取的对话 id。

    Returns:
        对话负载，不存在则返回 None。
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
    else:
        cur.execute("SELECT * FROM chats WHERE id = ? AND kb_id = ?", (chat_id, kb_id))
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
    return int(step_id) if step_id is not None else 0


def search_chunks(
    query: str, limit: int = 6, kb_id: Optional[int] = 1
) -> list[dict[str, Any]]:
    """按子串出现次数进行简单检索。

    Args:
        query: 查询字符串。
        limit: 最大返回数量。

    Returns:
        匹配的分块记录列表。
    """
    normalized_query = query.strip()
    if not normalized_query or limit <= 0:
        return []

    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute(
            """
            SELECT content, page
            FROM (
                SELECT
                    id,
                    content,
                    page,
                    (LENGTH(content) - LENGTH(REPLACE(content, ?, ''))) / LENGTH(?) AS score
                FROM chunks
                WHERE INSTR(content, ?) > 0
            ) ranked
            ORDER BY score DESC, id ASC
            LIMIT ?
            """,
            (normalized_query, normalized_query, normalized_query, limit),
        )
    else:
        cur.execute(
            """
            SELECT ranked.content, ranked.page
            FROM (
                SELECT
                    c.id,
                    c.content,
                    c.page,
                    (LENGTH(c.content) - LENGTH(REPLACE(c.content, ?, ''))) / LENGTH(?) AS score
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.kb_id = ? AND INSTR(c.content, ?) > 0
            ) ranked
            ORDER BY ranked.score DESC, ranked.id ASC
            LIMIT ?
            """,
            (normalized_query, normalized_query, kb_id, normalized_query, limit),
        )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_chat_history(
    chat_id: int, limit: int = 3, kb_id: Optional[int] = None
) -> list[dict[str, Any]]:
    """获取对话历史消息。

    Args:
        chat_id: 对话ID
        limit: 返回的消息对数量

    Returns:
        历史消息列表
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit * 2),
        )
    else:
        cur.execute(
            """
            SELECT m.role, m.content
            FROM messages m
            INNER JOIN chats c ON c.id = m.chat_id
            WHERE m.chat_id = ? AND c.kb_id = ?
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (chat_id, kb_id, limit * 2),
        )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return list(reversed(rows))


def update_chat_summary(
    chat_id: int, summary: str, kb_id: Optional[int] = None
) -> None:
    """更新对话摘要。

    Args:
        chat_id: 对话ID
        summary: 摘要内容
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute(
            "UPDATE chats SET summary = ? WHERE id = ?",
            (summary, chat_id),
        )
    else:
        cur.execute(
            "UPDATE chats SET summary = ? WHERE id = ? AND kb_id = ?",
            (summary, chat_id, kb_id),
        )
    conn.commit()
    conn.close()


def get_chat_summary(chat_id: int, kb_id: Optional[int] = None) -> Optional[str]:
    """获取对话摘要。

    Args:
        chat_id: 对话ID

    Returns:
        摘要内容或None
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute("SELECT summary FROM chats WHERE id = ?", (chat_id,))
    else:
        cur.execute(
            "SELECT summary FROM chats WHERE id = ? AND kb_id = ?", (chat_id, kb_id)
        )
    row = cur.fetchone()
    conn.close()
    return row["summary"] if row else None


def list_chunks(limit: int = 6, kb_id: Optional[int] = 1) -> list[dict[str, Any]]:
    """返回最早的分块用于兜底回答。

    Args:
        limit: 最大返回数量。

    Returns:
        分块记录列表。
    """
    conn = get_conn()
    cur = conn.cursor()
    if kb_id is None:
        cur.execute(
            "SELECT content, page FROM chunks ORDER BY id ASC LIMIT ?", (limit,)
        )
    else:
        cur.execute(
            """
            SELECT c.content, c.page
            FROM chunks c
            INNER JOIN documents d ON d.id = c.document_id
            WHERE d.kb_id = ?
            ORDER BY c.id ASC
            LIMIT ?
            """,
            (kb_id, limit),
        )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def source_ref_registered_for_kb(source_path: Path, kb_id: int) -> bool:
    normalized = source_path.resolve()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT source_ref FROM documents WHERE kb_id = ?", (kb_id,))
    rows = [str(row[0]) for row in cur.fetchall()]
    conn.close()

    for source_ref in rows:
        try:
            candidate = Path(source_ref).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()
        except Exception:
            continue
        if candidate == normalized:
            return True

    return False
