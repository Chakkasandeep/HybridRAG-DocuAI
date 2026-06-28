import re
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from backend.utils.logging_utils import setup_logger

logger = setup_logger("hybrid_retriever")

def clean_tokenize(text: str) -> List[str]:
    """Basic tokenizer that converts text to lowercase and extracts words.
    
    Args:
        text (str): Input text.
        
    Returns:
        List[str]: List of tokenized words.
    """
    return re.findall(r'\w+', text.lower())

class HybridRetriever:
    def __init__(
        self,
        vector_store,
        documents: List[Document],
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5
    ):
        """Initializes the Hybrid Retriever with FAISS and rank-bm25.
        
        Args:
            vector_store: FAISS vector store object.
            documents (List[Document]): The list of raw Document chunks.
            fusion_method (str): "rrf" or "weighted".
            rrf_k (int): RRF constant.
            semantic_weight (float): Weight for semantic search.
            keyword_weight (float): Weight for keyword search.
        """
        self.vector_store = vector_store
        self.documents = documents
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Initialize BM25
        logger.info(f"Initializing BM25 on {len(documents)} documents...")
        tokenized_corpus = [clean_tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 initialized successfully.")
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Document]:
        """Retrieves documents using hybrid search (semantic + keyword).
        
        Args:
            query (str): The search query.
            top_k (int): Number of top documents to retrieve.
            
        Returns:
            List[Document]: Combined retrieved documents.
        """
        if not self.documents:
            logger.warning("Retrieval requested but document list is empty.")
            return []
            
        # Get semantic search results (using similarity_search_with_score)
        # top_k * 2 is retrieved from each to ensure enough overlap for fusion
        retrieve_n = max(top_k * 2, 20)
        semantic_results = self.vector_store.similarity_search_with_score(query, k=retrieve_n)
        
        # Get BM25 keyword search results
        tokenized_query = clean_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort documents by BM25 score
        bm25_results = []
        for doc, score in zip(self.documents, bm25_scores):
            bm25_results.append((doc, float(score)))
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_results = bm25_results[:retrieve_n]
        
        if self.fusion_method.lower() == "rrf":
            return self._reciprocal_rank_fusion(semantic_results, bm25_results, top_k)
        elif self.fusion_method.lower() == "weighted":
            return self._weighted_score_fusion(semantic_results, bm25_results, top_k)
        else:
            logger.warning(f"Unknown fusion method '{self.fusion_method}'. Defaulting to RRF.")
            return self._reciprocal_rank_fusion(semantic_results, bm25_results, top_k)
            
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Document]:
        """Performs Reciprocal Rank Fusion (RRF) on retrieved results."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Process semantic ranks
        for rank, (doc, _) in enumerate(semantic_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            doc_map[chunk_id] = doc
            # 1-indexed rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + self.semantic_weight / (self.rrf_k + (rank + 1))
            
        # Process keyword ranks
        for rank, (doc, _) in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + self.keyword_weight / (self.rrf_k + (rank + 1))
            
        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        # Build final document list with fusion scores in metadata
        final_docs = []
        for chunk_id in sorted_chunks[:top_k]:
            doc = doc_map[chunk_id]
            doc.metadata["fusion_score"] = rrf_scores[chunk_id]
            doc.metadata["retrieval_method"] = "hybrid_rrf"
            final_docs.append(doc)
            
        return final_docs

    def _weighted_score_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Document]:
        """Performs Weighted Score Fusion by normalizing scores."""
        # 1. Normalize Semantic scores (FAISS L2 distance: lower is better, convert to similarity)
        sem_docs = [doc for doc, _ in semantic_results]
        sem_distances = [dist for _, dist in semantic_results]
        
        sem_sims = []
        for dist in sem_distances:
            sem_sims.append(1.0 / (1.0 + dist))
            
        min_sem = min(sem_sims) if sem_sims else 0.0
        max_sem = max(sem_sims) if sem_sims else 1.0
        sem_range = max_sem - min_sem
        
        norm_semantic: Dict[str, float] = {}
        for doc, sim in zip(sem_docs, sem_sims):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            norm_semantic[chunk_id] = (sim - min_sem) / sem_range if sem_range > 0 else 1.0
            
        # 2. Normalize BM25 scores (BM25 scores: higher is better)
        kw_docs = [doc for doc, _ in bm25_results]
        kw_scores = [score for _, score in bm25_results]
        
        min_kw = min(kw_scores) if kw_scores else 0.0
        max_kw = max(kw_scores) if kw_scores else 1.0
        kw_range = max_kw - min_kw
        
        norm_keyword: Dict[str, float] = {}
        for doc, score in zip(kw_docs, kw_scores):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            norm_keyword[chunk_id] = (score - min_kw) / kw_range if kw_range > 0 else 1.0
            
        # 3. Combine scores
        combined_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Populate document map
        for doc in sem_docs + kw_docs:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            doc_map[chunk_id] = doc
            
        # Compute combined score
        for chunk_id in doc_map.keys():
            s_score = norm_semantic.get(chunk_id, 0.0)
            k_score = norm_keyword.get(chunk_id, 0.0)
            combined_scores[chunk_id] = self.semantic_weight * s_score + self.keyword_weight * k_score
            
        # Sort by combined score descending
        sorted_chunks = sorted(combined_scores.keys(), key=lambda cid: combined_scores[cid], reverse=True)
        
        final_docs = []
        for chunk_id in sorted_chunks[:top_k]:
            doc = doc_map[chunk_id]
            doc.metadata["fusion_score"] = combined_scores[chunk_id]
            doc.metadata["retrieval_method"] = "hybrid_weighted"
            final_docs.append(doc)
            
        return final_docs
