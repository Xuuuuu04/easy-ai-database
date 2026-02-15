from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..db import ensure_knowledge_base_exists
from ..retrieval_eval import (
    build_default_tuning_parameter_grid,
    generate_retrieval_benchmark_dataset,
    load_default_benchmark_dataset,
    run_retrieval_evaluation,
)
from ..schemas import RetrievalDatasetGenerateRequest, RetrievalEvalRequest
from ..state import get_or_create_kb_index

router = APIRouter()


@router.post("/eval/retrieval")
def evaluate_retrieval(req: RetrievalEvalRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    dataset_source = "request"
    cases: list[dict[str, Any]]
    parameter_grid = req.parameter_grid
    k = req.k

    if req.cases is None:
        dataset_source = "default"
        dataset = load_default_benchmark_dataset()
        raw_cases = dataset.get("cases", [])
        if not isinstance(raw_cases, list) or not raw_cases:
            raise HTTPException(
                status_code=400, detail="Default benchmark dataset is empty"
            )
        cases = [dict(case) for case in raw_cases]
        if parameter_grid is None and isinstance(dataset.get("parameter_grid"), dict):
            parameter_grid = dataset["parameter_grid"]
        if parameter_grid is None and req.auto_tune:
            parameter_grid = build_default_tuning_parameter_grid()
        k = int(dataset.get("k", req.k))
    else:
        if not req.cases:
            raise HTTPException(status_code=400, detail="At least one case is required")
        cases = [case.model_dump() for case in req.cases]
        if parameter_grid is None and req.auto_tune:
            parameter_grid = build_default_tuning_parameter_grid()

    kb_index = get_or_create_kb_index(req.kb_id)
    result = run_retrieval_evaluation(
        index=kb_index,
        cases=cases,
        parameter_grid=parameter_grid,
        kb_id=req.kb_id,
        k=max(1, k),
        include_case_results=req.include_case_results,
        include_llm_judge=req.include_llm_judge,
        llm_judge_sample_size=max(1, req.llm_judge_sample_size),
        llm_judge_on_all_configs=req.llm_judge_on_all_configs,
    )
    result["dataset"] = dataset_source
    return result


@router.post("/eval/retrieval/generate-dataset")
def generate_retrieval_dataset(req: RetrievalDatasetGenerateRequest) -> dict[str, Any]:
    if not ensure_knowledge_base_exists(req.kb_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return generate_retrieval_benchmark_dataset(
        kb_id=req.kb_id,
        count=max(1, req.case_count),
        use_llm=req.use_llm,
    )
