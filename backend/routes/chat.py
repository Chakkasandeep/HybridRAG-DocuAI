from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

class ChatRequest(BaseModel):
    question: str

class SourceMetadata(BaseModel):
    id: str
    source: str
    page: Any
    fusion_score: float
    rerank_confidence: str
    content: str

class ChatMetrics(BaseModel):
    retrieval_latency: float
    rerank_latency: float
    generation_latency: float
    num_candidates: int
    num_reranked: int

class ChatResponse(BaseModel):
    answer: str
    metrics: ChatMetrics
    sources: List[SourceMetadata]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, x_groq_api_key: Optional[str] = Header(None)):
    """API endpoint to ask questions using the Hybrid RAG pipeline."""
    result = rag_service.chat(request.question, override_api_key=x_groq_api_key)
    
    if "error" in result:
        # Check if it's missing API key or missing index
        if "API Key" in result["error"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
            
    return result
