from sentence_transformers import CrossEncoder
from backend.utils.logging_utils import setup_logger
from typing import List
from langchain_core.documents import Document

logger = setup_logger("reranker")

class DocumentReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """Initializes the CrossEncoder reranker.
        
        Args:
            model_name (str): CrossEncoder model name.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        logger.info(f"Loading CrossEncoder model: {model_name} on {device}...")
        self.model = CrossEncoder(model_name, device=device)
        logger.info("CrossEncoder model loaded successfully.")
        
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Reranks retrieved documents based on query semantic relevance.
        
        Args:
            query (str): The search query.
            documents (List[Document]): Retrieved documents to rerank.
            top_k (int): Number of top reranked documents to return.
            
        Returns:
            List[Document]: Reranked documents.
        """
        if not documents:
            logger.warning("Reranking requested but document list is empty.")
            return []
            
        pairs = [[query, doc.page_content] for doc in documents]
        logger.info(f"Reranking {len(documents)} documents using CrossEncoder...")
        scores = self.model.predict(pairs)
        
        # Pair documents with their score
        doc_scores = list(zip(documents, scores))
        # Sort by score descending (higher cross-encoder score means more relevant)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate min and max of retrieved candidates for relative confidence scaling
        raw_scores = [float(s) for _, s in doc_scores]
        max_score = max(raw_scores) if raw_scores else 0.0
        min_score = min(raw_scores) if raw_scores else 0.0
        score_range = max_score - min_score
        
        # Save reranker scores and rank in metadata
        reranked_docs = []
        for idx, (doc, score) in enumerate(doc_scores[:top_k]):
            f_score = float(score)
            # Scale relative relevance to a user-friendly 50% - 98% range
            rel_score = (f_score - min_score) / score_range if score_range > 0 else 1.0
            confidence_pct = 50.0 + (rel_score * 48.0)
            
            doc.metadata["rerank_score"] = f_score
            doc.metadata["rerank_confidence"] = f"{confidence_pct:.1f}%"
            doc.metadata["rerank_rank"] = idx + 1
            reranked_docs.append(doc)
            
        logger.info(f"Reranking complete. Selected top {len(reranked_docs)} chunks.")
        return reranked_docs
