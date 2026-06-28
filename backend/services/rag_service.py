import os
import time
from typing import List, Dict, Any, Optional
from backend.utils.config_loader import load_config
from backend.retrieval.vector_store import VectorStoreManager
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.reranker.reranker import DocumentReranker
from backend.models.llm_manager import LLMManager
from backend.utils.logging_utils import setup_logger

logger = setup_logger("rag_service")

class RAGService:
    def __init__(self):
        self.config = load_config()
        self.index_path = os.path.abspath(self.config["paths"]["faiss_index"])
        
    def chat(self, question: str, override_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Runs the RAG pipeline for a user question."""
        # 1. Resolve Groq API Key
        api_key = override_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("Groq API Key is not configured.")
            return {
                "error": "Groq API Key is missing. Please provide it in the X-Groq-API-Key header, or configure GROQ_API_KEY in the backend .env file."
            }

        # 2. Check if index exists
        index_exists = os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "documents.pkl"))
        if not index_exists:
            logger.error("FAISS index does not exist.")
            return {
                "error": "No documents have been indexed yet. Please upload and process documents before chatting."
            }

        # 3. Setup retrieval
        t0 = time.time()
        vs_manager = VectorStoreManager(
            model_name=self.config["embedding"]["model_name"],
            device=self.config["embedding"]["device"],
            index_path=self.index_path
        )
        faiss_store = vs_manager.load()
        documents = vs_manager.load_documents()
        
        retriever = HybridRetriever(
            vector_store=faiss_store,
            documents=documents,
            fusion_method=self.config["retrieval"]["fusion_method"],
            rrf_k=self.config["retrieval"]["rrf_k"],
            semantic_weight=self.config["retrieval"]["weights"]["semantic"],
            keyword_weight=self.config["retrieval"]["weights"]["keyword"]
        )
        
        candidates = retriever.retrieve(question, top_k=self.config["retrieval"]["top_k_retrieve"])
        retrieval_latency = time.time() - t0

        # 4. Reranking
        t1 = time.time()
        reranker = DocumentReranker(
            model_name=self.config["reranker"]["model_name"],
            device=self.config["reranker"]["device"]
        )
        reranked_docs = reranker.rerank(question, candidates, top_k=self.config["reranker"]["top_k_rerank"])
        rerank_latency = time.time() - t1

        # 5. Generation
        t2 = time.time()
        llm_manager = LLMManager(
            groq_api_key=api_key,
            model_name=self.config["llm"]["model_name"],
            temperature=self.config["llm"]["temperature"]
        )
        
        chain = llm_manager.get_qa_chain()
        result = llm_manager.generate_answer(chain, question, reranked_docs)
        generation_latency = time.time() - t2

        # 6. Format response sources
        sources_serialized = []
        for idx, doc in enumerate(result["sources"]):
            sources_serialized.append({
                "id": doc.metadata.get("chunk_id", f"chunk_{idx}"),
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "fusion_score": float(doc.metadata.get("fusion_score", 0.0)),
                "rerank_confidence": doc.metadata.get("rerank_confidence", "N/A"),
                "content": doc.page_content
            })

        return {
            "answer": result["answer"],
            "metrics": {
                "retrieval_latency": retrieval_latency,
                "rerank_latency": rerank_latency,
                "generation_latency": generation_latency,
                "num_candidates": len(candidates),
                "num_reranked": len(reranked_docs)
            },
            "sources": sources_serialized
        }
