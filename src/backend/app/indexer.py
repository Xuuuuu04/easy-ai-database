from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import Iterable

import faiss
import httpx
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore

from .config import settings


def _resolve_proxy_url() -> str | None:
    proxy = (
        os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if not proxy:
        return None
    if proxy.startswith("socks5://"):
        return "socks5h://" + proxy[len("socks5://") :]
    return proxy


_PROXY_URL = _resolve_proxy_url()
_SYNC_HTTP_CLIENT = (
    httpx.Client(proxy=_PROXY_URL, trust_env=False) if _PROXY_URL else None
)
_ASYNC_HTTP_CLIENT = (
    httpx.AsyncClient(proxy=_PROXY_URL, trust_env=False) if _PROXY_URL else None
)


def get_embed_model() -> OpenAIEmbedding:
    signature = inspect.signature(OpenAIEmbedding.__init__)
    supports_batch = "embed_batch_size" in signature.parameters
    supports_workers = "num_workers" in signature.parameters

    if supports_batch and supports_workers:
        return OpenAIEmbedding(
            model="text-embedding-3-small",
            model_name=settings.embed_model,
            api_base=settings.embed_base_url,
            api_key=settings.embed_api_key,
            timeout=60.0,
            max_retries=3,
            http_client=_SYNC_HTTP_CLIENT,
            async_http_client=_ASYNC_HTTP_CLIENT,
            embed_batch_size=max(8, settings.embed_batch_size),
            num_workers=max(1, settings.embed_num_workers),
        )

    if supports_batch:
        return OpenAIEmbedding(
            model="text-embedding-3-small",
            model_name=settings.embed_model,
            api_base=settings.embed_base_url,
            api_key=settings.embed_api_key,
            timeout=60.0,
            max_retries=3,
            http_client=_SYNC_HTTP_CLIENT,
            async_http_client=_ASYNC_HTTP_CLIENT,
            embed_batch_size=max(8, settings.embed_batch_size),
        )

    return OpenAIEmbedding(
        model="text-embedding-3-small",
        model_name=settings.embed_model,
        api_base=settings.embed_base_url,
        api_key=settings.embed_api_key,
        timeout=60.0,
        max_retries=3,
        http_client=_SYNC_HTTP_CLIENT,
        async_http_client=_ASYNC_HTTP_CLIENT,
    )


def get_llm() -> OpenAI:
    return OpenAI(
        model="gpt-3.5-turbo",
        additional_kwargs={"model": settings.llm_model},
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.2,
        timeout=120.0,
        max_retries=3,
        http_client=_SYNC_HTTP_CLIENT,
        async_http_client=_ASYNC_HTTP_CLIENT,
    )


def _infer_embedding_dim(embed_model: OpenAIEmbedding) -> int:
    try:
        vec = embed_model.get_text_embedding("dimension probe")
        return len(vec)
    except Exception:
        return 1536


def _get_faiss_store(dim: int) -> FaissVectorStore:
    index = faiss.IndexFlatL2(dim)
    return FaissVectorStore(index)


def _resolve_index_dir(index_dir: str | Path | None = None) -> Path:
    if index_dir is None:
        return Path(settings.index_dir)
    return Path(index_dir)


def load_or_create_index(index_dir: str | Path | None = None) -> VectorStoreIndex:
    index_dir = _resolve_index_dir(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embed_model = get_embed_model()
    if (index_dir / "default__vector_store.json").exists():
        vector_store = FaissVectorStore.from_persist_dir(str(index_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(index_dir),
        )
        loaded_index = load_index_from_storage(storage_context, embed_model=embed_model)
        if isinstance(loaded_index, VectorStoreIndex):
            return loaded_index
        raise TypeError("Persisted index is not a VectorStoreIndex")

    dim = _infer_embedding_dim(embed_model)
    vector_store = _get_faiss_store(dim)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        [], storage_context=storage_context, embed_model=embed_model
    )


def persist_index(index: VectorStoreIndex) -> None:
    persist_index_to(index, settings.index_dir)


def persist_index_to(index: VectorStoreIndex, index_dir: str | Path) -> None:
    resolved = _resolve_index_dir(index_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(resolved))


def insert_nodes(
    index: VectorStoreIndex,
    nodes: Iterable[BaseNode],
    index_dir: str | Path | None = None,
) -> None:
    index.insert_nodes(list(nodes))
    persist_index_to(index, _resolve_index_dir(index_dir))
