import numpy as np
from typing import List, Set
from langchain_core.documents import Document

def is_match(retrieved_text: str, ground_truth_text: str) -> bool:
    """Checks if a retrieved chunk matches a ground truth relevant chunk.
    
    Uses substring containment and Jaccard word overlap as a fast robust metric.
    
    Args:
        retrieved_text (str): Content of the retrieved document.
        ground_truth_text (str): Content of the ground truth relevant chunk.
        
    Returns:
        bool: True if matched, False otherwise.
    """
    ret_text_lower = retrieved_text.lower().strip()
    gt_text_lower = ground_truth_text.lower().strip()
    
    # Substring containment
    if gt_text_lower in ret_text_lower or ret_text_lower in gt_text_lower:
        return True
        
    # Jaccard overlap of words
    ret_words = set(ret_text_lower.split())
    gt_words = set(gt_text_lower.split())
    if not ret_words or not gt_words:
        return False
    intersection = ret_words.intersection(gt_words)
    union = ret_words.union(gt_words)
    jaccard = len(intersection) / len(union)
    
    return jaccard > 0.45  # Standard threshold for matching text overlap

def evaluate_hit_rate(retrieved_docs: List[Document], ground_truth_chunks: List[str], k: int) -> float:
    """Computes Hit Rate @ k. Returns 1.0 if at least one relevant chunk is retrieved, else 0.0.
    
    Args:
        retrieved_docs (List[Document]): List of retrieved documents.
        ground_truth_chunks (List[str]): List of relevant chunks.
        k (int): Top-k constraint.
        
    Returns:
        float: Hit Rate @ k (0.0 or 1.0).
    """
    docs_to_check = retrieved_docs[:k]
    for gt in ground_truth_chunks:
        for doc in docs_to_check:
            if is_match(doc.page_content, gt):
                return 1.0
    return 0.0

def evaluate_precision_at_k(retrieved_docs: List[Document], ground_truth_chunks: List[str], k: int) -> float:
    """Computes Precision @ k. Proportion of retrieved documents in top-k that are relevant.
    
    Args:
        retrieved_docs (List[Document]): List of retrieved documents.
        ground_truth_chunks (List[str]): List of relevant chunks.
        k (int): Top-k constraint.
        
    Returns:
        float: Precision @ k.
    """
    if k == 0:
        return 0.0
    docs_to_check = retrieved_docs[:k]
    hits = 0
    # Iterate over retrieved docs and see if they match ANY ground truth
    for doc in docs_to_check:
        matched = False
        for gt in ground_truth_chunks:
            if is_match(doc.page_content, gt):
                matched = True
                break
        if matched:
            hits += 1
            
    return hits / k

def evaluate_recall_at_k(retrieved_docs: List[Document], ground_truth_chunks: List[str], k: int) -> float:
    """Computes Recall @ k. Proportion of ground truth relevant chunks retrieved in top-k.
    
    Args:
        retrieved_docs (List[Document]): List of retrieved documents.
        ground_truth_chunks (List[str]): List of relevant chunks.
        k (int): Top-k constraint.
        
    Returns:
        float: Recall @ k.
    """
    if not ground_truth_chunks:
        return 0.0
    docs_to_check = retrieved_docs[:k]
    
    matched_gts = 0
    for gt in ground_truth_chunks:
        matched = False
        for doc in docs_to_check:
            if is_match(doc.page_content, gt):
                matched = True
                break
        if matched:
            matched_gts += 1
            
    return matched_gts / len(ground_truth_chunks)

def evaluate_mrr(retrieved_docs: List[Document], ground_truth_chunks: List[str]) -> float:
    """Computes Reciprocal Rank (RR). RR is 1 / rank of first relevant doc (1-indexed).
    
    Args:
        retrieved_docs (List[Document]): List of retrieved documents.
        ground_truth_chunks (List[str]): List of relevant chunks.
        
    Returns:
        float: Reciprocal Rank.
    """
    for rank, doc in enumerate(retrieved_docs):
        for gt in ground_truth_chunks:
            if is_match(doc.page_content, gt):
                return 1.0 / (rank + 1)
    return 0.0

def evaluate_ndcg_at_k(retrieved_docs: List[Document], ground_truth_chunks: List[str], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG) @ k.
    
    Args:
        retrieved_docs (List[Document]): List of retrieved documents.
        ground_truth_chunks (List[str]): List of relevant chunks.
        k (int): Top-k constraint.
        
    Returns:
        float: nDCG @ k.
    """
    if k == 0:
        return 0.0
    docs_to_check = retrieved_docs[:k]
    
    # Calculate DCG
    dcg = 0.0
    for idx, doc in enumerate(docs_to_check):
        rank = idx + 1
        is_rel = 0
        for gt in ground_truth_chunks:
            if is_match(doc.page_content, gt):
                is_rel = 1
                break
        if is_rel:
            dcg += 1.0 / np.log2(rank + 1)
            
    # Calculate Ideal DCG (IDCG)
    idcg = 0.0
    num_relevant = len(ground_truth_chunks)
    for idx in range(min(k, num_relevant)):
        rank = idx + 1
        idcg += 1.0 / np.log2(rank + 1)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
