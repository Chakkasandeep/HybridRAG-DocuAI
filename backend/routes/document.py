from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List, Dict, Any, Optional
from backend.services.document_service import DocumentService
from pydantic import BaseModel

router = APIRouter()
doc_service = DocumentService()

class DocumentItem(BaseModel):
    filename: str
    size_kb: float
    status: str
    is_indexed: bool

class ProcessResponse(BaseModel):
    status: str
    message: str
    num_documents: Optional[int] = None
    num_chunks: Optional[int] = None

class DeleteResponse(BaseModel):
    status: str
    message: str
    remaining_count: int

@router.post("/upload", response_model=List[Dict[str, Any]])
async def upload_files(files: List[UploadFile] = File(...)):
    """API endpoint to upload multiple PDF files."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded."
        )
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} is not a PDF."
            )
            
    try:
        results = doc_service.save_uploads(files)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploads: {str(e)}"
        )

@router.post("/process", response_model=ProcessResponse)
async def process_documents():
    """API endpoint to parse uploaded PDFs and build the FAISS vector index."""
    result = doc_service.process_and_index()
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    return result

@router.get("/documents", response_model=List[DocumentItem])
async def list_documents():
    """API endpoint to list all uploaded documents and their indexing status."""
    try:
        return doc_service.list_documents()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.delete("/document/{id:path}", response_model=DeleteResponse)
async def delete_document(id: str):
    """API endpoint to delete a document and rebuild the vector index."""
    result = doc_service.delete_document(id)
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
    return result
