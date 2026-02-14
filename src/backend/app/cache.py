from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .indexer import get_embed_model


class SemanticCache:
    """基于语义相似度的查询缓存。"""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._access_times: dict[str, float] = {}
        self.embed_model = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化SQLite缓存表。"""
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                query_hash TEXT PRIMARY KEY,
                kb_id INTEGER NOT NULL DEFAULT 1,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations TEXT,
                embedding BLOB,
                created_at REAL DEFAULT (unixepoch())
            )
        """)
        cur.execute("PRAGMA table_info(semantic_cache)")
        columns = {str(row[1]) for row in cur.fetchall()}
        if "kb_id" not in columns:
            cur.execute(
                "ALTER TABLE semantic_cache ADD COLUMN kb_id INTEGER NOT NULL DEFAULT 1"
            )
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_created ON semantic_cache(created_at)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_kb_id_created ON semantic_cache(kb_id, created_at)
        """)
        conn.commit()
        conn.close()
        self._load_from_db()

    def _load_from_db(self) -> None:
        """从数据库加载缓存。"""
        try:
            conn = sqlite3.connect(settings.db_path)
            cur = conn.cursor()
            ttl_seconds = settings.cache_ttl_hours * 3600
            cur.execute(
                """
                SELECT query_hash, kb_id, query, answer, citations, embedding, created_at
                FROM semantic_cache
                WHERE unixepoch() - created_at < ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (ttl_seconds, settings.cache_max_size),
            )
            for row in cur.fetchall():
                (
                    query_hash,
                    kb_id,
                    query,
                    answer,
                    citations,
                    embedding_blob,
                    created_at,
                ) = row
                self._cache[query_hash] = {
                    "kb_id": kb_id,
                    "query": query,
                    "answer": answer,
                    "citations": json.loads(citations) if citations else [],
                }
                if embedding_blob:
                    self._embeddings[query_hash] = list(
                        np.frombuffer(embedding_blob, dtype=np.float32)
                    )
                self._access_times[query_hash] = created_at
            conn.close()
        except Exception:
            pass

    def _get_embedding(self, text: str) -> list[float] | None:
        """获取文本的embedding。"""
        if settings.mock_mode:
            return None
        try:
            if self.embed_model is None:
                self.embed_model = get_embed_model()
            return self.embed_model.get_text_embedding(text)
        except Exception:
            return None

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度。"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def get(self, query: str, kb_id: int = 1) -> dict[str, Any] | None:
        """获取缓存结果。"""
        if not settings.enable_semantic_cache:
            return None

        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            query_hash = hashlib.md5(f"{kb_id}:{query}".encode()).hexdigest()
            if query_hash in self._cache:
                return self._cache[query_hash]
            return None

        best_match = None
        best_score = 0.0

        for query_hash, cached_embedding in self._embeddings.items():
            cached_item = self._cache.get(query_hash)
            if cached_item is None or int(cached_item.get("kb_id", 1)) != kb_id:
                continue
            score = self._cosine_similarity(query_embedding, cached_embedding)
            if score > best_score and score >= settings.cache_similarity_threshold:
                best_score = score
                best_match = query_hash

        if best_match:
            self._access_times[best_match] = time.time()
            return self._cache[best_match]

        return None

    def set(
        self,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        kb_id: int = 1,
    ) -> None:
        """设置缓存。"""
        if not settings.enable_semantic_cache:
            return

        query_hash = hashlib.md5(f"{kb_id}:{query}".encode()).hexdigest()
        query_embedding = self._get_embedding(query)

        self._cache[query_hash] = {
            "kb_id": kb_id,
            "query": query,
            "answer": answer,
            "citations": citations,
        }
        self._access_times[query_hash] = time.time()

        if query_embedding:
            self._embeddings[query_hash] = query_embedding

        self._persist_to_db(
            query_hash,
            kb_id,
            query,
            answer,
            citations,
            query_embedding,
        )
        self._cleanup_if_needed()

    def _persist_to_db(
        self,
        query_hash: str,
        kb_id: int,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        embedding: list[float] | None,
    ) -> None:
        """持久化到数据库。"""
        try:
            conn = sqlite3.connect(settings.db_path)
            cur = conn.cursor()
            embedding_blob = (
                np.array(embedding, dtype=np.float32).tobytes() if embedding else None
            )
            cur.execute(
                """
                INSERT OR REPLACE INTO semantic_cache
                (query_hash, kb_id, query, answer, citations, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, unixepoch())
            """,
                (
                    query_hash,
                    kb_id,
                    query,
                    answer,
                    json.dumps(citations),
                    embedding_blob,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _cleanup_if_needed(self) -> None:
        """清理过期和超量缓存。"""
        if len(self._cache) <= settings.cache_max_size:
            return

        sorted_items = sorted(
            self._access_times.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        to_remove = [k for k, _ in sorted_items[settings.cache_max_size :]]
        for query_hash in to_remove:
            del self._cache[query_hash]
            del self._access_times[query_hash]
            if query_hash in self._embeddings:
                del self._embeddings[query_hash]

        self._cleanup_db()

    def _cleanup_db(self) -> None:
        """清理数据库中的过期记录。"""
        try:
            conn = sqlite3.connect(settings.db_path)
            cur = conn.cursor()
            ttl_seconds = settings.cache_ttl_hours * 3600
            cur.execute(
                """
                DELETE FROM semantic_cache
                WHERE unixepoch() - created_at > ?
            """,
                (ttl_seconds,),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def clear(self, kb_id: int | None = None) -> None:
        """清空缓存。"""
        if kb_id is None:
            self._cache.clear()
            self._embeddings.clear()
            self._access_times.clear()
        else:
            remove_keys = [
                key
                for key, value in self._cache.items()
                if int(value.get("kb_id", 1)) == kb_id
            ]
            for key in remove_keys:
                self._cache.pop(key, None)
                self._embeddings.pop(key, None)
                self._access_times.pop(key, None)
        try:
            conn = sqlite3.connect(settings.db_path)
            cur = conn.cursor()
            if kb_id is None:
                cur.execute("DELETE FROM semantic_cache")
            else:
                cur.execute("DELETE FROM semantic_cache WHERE kb_id = ?", (kb_id,))
            conn.commit()
            conn.close()
        except Exception:
            pass


semantic_cache = SemanticCache()
