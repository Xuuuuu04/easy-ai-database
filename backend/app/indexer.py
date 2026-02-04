from __future__ import annotations

from pathlib import Path
from typing import Iterable
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import faiss

from .config import settings


def get_embed_model() -> OpenAIEmbedding:
    return OpenAIEmbedding(
        model_name=settings.embed_model,
        api_base=settings.embed_base_url,
        api_key=settings.embed_api_key,
        timeout=30.0,
        max_retries=2,
        reuse_client=False,
    )


def get_llm() -> OpenAI:
    return OpenAI(
        model="gpt-3.5-turbo",
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.2,
        timeout=60.0,
        max_retries=2,
        reuse_client=False,
        additional_kwargs={"model": settings.llm_model},
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


def load_or_create_index() -> VectorStoreIndex:
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embed_model = get_embed_model()
    if (index_dir / "vector_store.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        return VectorStoreIndex.from_vector_store(
            storage_context.vector_store, embed_model=embed_model
        )

    dim = _infer_embedding_dim(embed_model)
    vector_store = _get_faiss_store(dim)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)


def persist_index(index: VectorStoreIndex) -> None:
    index.storage_context.persist(persist_dir=str(settings.index_dir))


def insert_nodes(index: VectorStoreIndex, nodes: Iterable[BaseNode]) -> None:
    index.insert_nodes(list(nodes))
    persist_index(index)
