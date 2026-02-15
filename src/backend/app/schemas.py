from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class UrlIngestRequest(BaseModel):
    url: str
    kb_id: int = 1


class ChatRequest(BaseModel):
    question: str
    chat_id: Optional[int] = None
    kb_id: int = 1
    stream: bool = False


class KnowledgeBaseCreateRequest(BaseModel):
    name: str
    description: str = ""


class BatchDocumentDeleteRequest(BaseModel):
    kb_id: int = 1
    document_ids: list[int] = Field(default_factory=list)


class BatchDocumentReindexRequest(BaseModel):
    kb_id: int = 1
    document_ids: list[int] = Field(default_factory=list)


class RetrievalEvalCase(BaseModel):
    id: Optional[str] = None
    query: str
    relevant_ids: list[int] = Field(default_factory=list)
    relevant_sources: list[str] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    relevant_snippets: list[str] = Field(default_factory=list)
    hard_negative_ids: list[int] = Field(default_factory=list)
    hard_negative_sources: list[str] = Field(default_factory=list)
    hard_negative_pages: list[int] = Field(default_factory=list)
    hard_negative_snippets: list[str] = Field(default_factory=list)


class RetrievalEvalRequest(BaseModel):
    cases: Optional[list[RetrievalEvalCase]] = None
    parameter_grid: Optional[dict[str, list[Any]]] = None
    kb_id: int = 1
    k: int = 5
    auto_tune: bool = False
    include_case_results: bool = False
    include_llm_judge: bool = False
    llm_judge_sample_size: int = 10
    llm_judge_on_all_configs: bool = False


class RetrievalDatasetGenerateRequest(BaseModel):
    kb_id: int = 1
    case_count: int = 20
    use_llm: bool = True


class RetrievalRequest(BaseModel):
    question: str
    kb_id: int = 1
    chat_id: Optional[int] = None
    top_k: int = 6
