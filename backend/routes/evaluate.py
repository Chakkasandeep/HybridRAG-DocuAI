from fastapi import APIRouter, Header, HTTPException, status
from typing import Dict, Any, Optional
from backend.services.evaluation_service import EvaluationService

router = APIRouter()
eval_service = EvaluationService()

@router.post("/evaluate")
async def evaluate_pipeline(x_groq_api_key: Optional[str] = Header(None)):
    """API endpoint to trigger RAG pipeline evaluation benchmarks."""
    result = eval_service.trigger_evaluation(override_api_key=x_groq_api_key)
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
    return result["results"]

@router.get("/evaluation/results")
async def get_evaluation_results():
    """API endpoint to retrieve the latest evaluation results."""
    result = eval_service.get_results()
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
    return result["results"]
