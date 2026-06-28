import os
import shutil
from typing import List, Dict, Any
from fastapi import UploadFile
from backend.utils.config_loader import load_config
from backend.utils.pdf_processor import process_pdfs
from backend.retrieval.vector_store import VectorStoreManager
from backend.utils.logging_utils import setup_logger

logger = setup_logger("document_service")

class DocumentService:
    def __init__(self, uploads_dir: str = "backend/uploads"):
        self.config = load_config()
        self.uploads_dir = os.path.abspath(uploads_dir)
        os.makedirs(self.uploads_dir, exist_ok=True)
        
        self.index_path = os.path.abspath(self.config["paths"]["faiss_index"])
        
    def save_uploads(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """Saves uploaded files to the uploads directory."""
        saved_files = []
        for file in files:
            file_path = os.path.join(self.uploads_dir, file.filename)
            logger.info(f"Saving upload file to {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Get size in KB
            size_kb = round(os.path.getsize(file_path) / 1024, 2)
            saved_files.append({
                "filename": file.filename,
                "size_kb": size_kb,
                "status": "uploaded"
            })
        return saved_files

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists all uploaded documents and their indexing status."""
        documents = []
        
        # Load indexed sources from FAISS to check indexing status
        indexed_sources = set()
        index_exists = os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "documents.pkl"))
        if index_exists:
            try:
                vs_manager = VectorStoreManager(
                    model_name=self.config["embedding"]["model_name"],
                    device=self.config["embedding"]["device"],
                    index_path=self.index_path
                )
                docs = vs_manager.load_documents()
                indexed_sources = set(d.metadata.get("source", "unknown") for d in docs)
            except Exception as e:
                logger.error(f"Error loading FAISS index documents: {e}")
        
        # Check files in uploads directory
        if os.path.exists(self.uploads_dir):
            for filename in os.listdir(self.uploads_dir):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(self.uploads_dir, filename)
                    size_kb = round(os.path.getsize(file_path) / 1024, 2)
                    is_indexed = filename in indexed_sources
                    documents.append({
                        "filename": filename,
                        "size_kb": size_kb,
                        "status": "indexed" if is_indexed else "uploaded",
                        "is_indexed": is_indexed
                    })
        return documents

    def process_and_index(self) -> Dict[str, Any]:
        """Processes all uploaded PDFs and builds/updates the FAISS index."""
        # Find all PDF files in uploads folder
        pdf_paths = []
        if os.path.exists(self.uploads_dir):
            for filename in os.listdir(self.uploads_dir):
                if filename.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(self.uploads_dir, filename))
        
        if not pdf_paths:
            logger.warning("No PDF documents found to process.")
            return {"status": "error", "message": "No PDF files found in uploads directory to process."}
            
        logger.info(f"Processing and indexing {len(pdf_paths)} PDFs...")
        
        # 1. Process and split PDFs using the existing processor
        documents = process_pdfs(
            pdf_paths,
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"]
        )
        
        if not documents:
            return {"status": "error", "message": "No extractable text found in the PDF files."}
            
        # 2. Build FAISS index using vector store manager
        vs_manager = VectorStoreManager(
            model_name=self.config["embedding"]["model_name"],
            device=self.config["embedding"]["device"],
            index_path=self.index_path
        )
        vs_manager.create_and_save(documents)
        
        return {
            "status": "success",
            "message": f"Successfully indexed {len(pdf_paths)} documents.",
            "num_documents": len(pdf_paths),
            "num_chunks": len(documents)
        }

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Deletes a document from uploads directory and rebuilds the FAISS index."""
        file_path = os.path.join(self.uploads_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"File {filename} not found in uploads directory.")
            return {"status": "error", "message": f"File {filename} not found."}
            
        # Delete file from disk
        os.remove(file_path)
        logger.info(f"Deleted PDF file {filename} from disk.")
        
        # Check remaining PDFs
        remaining_pdfs = [
            os.path.join(self.uploads_dir, f)
            for f in os.listdir(self.uploads_dir)
            if f.lower().endswith(".pdf")
        ]
        
        if not remaining_pdfs:
            # Delete FAISS index folder if no files remain
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
                logger.info(f"Removed FAISS index directory since no documents remain.")
            return {
                "status": "success",
                "message": f"Deleted {filename}. No PDFs remain; FAISS index cleared.",
                "remaining_count": 0
            }
        
        # Re-index remaining PDFs
        logger.info(f"Re-indexing remaining {len(remaining_pdfs)} PDFs...")
        documents = process_pdfs(
            remaining_pdfs,
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"]
        )
        
        if documents:
            vs_manager = VectorStoreManager(
                model_name=self.config["embedding"]["model_name"],
                device=self.config["embedding"]["device"],
                index_path=self.index_path
            )
            vs_manager.create_and_save(documents)
            logger.info(f"Successfully rebuilt FAISS index with remaining documents.")
        else:
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
                logger.info(f"Removed FAISS index directory due to lack of text in remaining documents.")
                
        return {
            "status": "success",
            "message": f"Deleted {filename} and rebuilt index with remaining files.",
            "remaining_count": len(remaining_pdfs)
        }
