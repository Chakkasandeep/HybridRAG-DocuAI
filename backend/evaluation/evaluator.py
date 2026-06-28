import sys
import os
# Add workspace root to system path to enable modules import from root CWD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

from backend.utils.config_loader import load_config
from backend.utils.logging_utils import setup_logger
from backend.utils.pdf_processor import Document
from backend.retrieval.vector_store import VectorStoreManager
from backend.retrieval.hybrid_retriever import HybridRetriever, clean_tokenize
from backend.reranker.reranker import DocumentReranker
from backend.models.llm_manager import LLMManager
from backend.evaluation.metrics import (
    evaluate_hit_rate,
    evaluate_precision_at_k,
    evaluate_recall_at_k,
    evaluate_mrr,
    evaluate_ndcg_at_k
)

logger = setup_logger("evaluator")

DEFAULT_DATASET = [
    {
        "question": "What is the primary function of the attention mechanism?",
        "ground_truth": "The primary function of the attention mechanism is to allow the model to focus on different parts of the input sequence when generating the output, enabling better handling of long-range dependencies.",
        "relevant_chunks": [
            "The attention mechanism dynamically weights different parts of the input sequence to help the model focus on relevant context.",
            "By allowing the decoder to focus on specific parts of the encoder output, attention resolves the bottleneck of traditional sequence-to-sequence models."
        ]
    },
    {
        "question": "Explain the concept of Hybrid Retrieval.",
        "ground_truth": "Hybrid retrieval combines dense retrieval (semantic search) and sparse retrieval (keyword search) to leverage both semantic understanding and exact keyword matching.",
        "relevant_chunks": [
            "Hybrid retrieval integrates semantic dense vectors and BM25 sparse keyword indices.",
            "Combining vector database search with classical keyword search yields higher recall and better relevance."
        ]
    },
    {
        "question": "What is cross-encoder reranking?",
        "ground_truth": "Cross-encoder reranking evaluates the query and document candidates together to produce high-accuracy relevance scores, filtering and ordering top results.",
        "relevant_chunks": [
            "Cross-encoders process the query and document jointly, capturing full token interaction.",
            "Reranking candidate chunks from a fast first-stage retriever improves retrieval precision before passing to the LLM."
        ]
    }
]

def load_or_create_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Loads evaluation dataset or creates a default one if not found.
    
    Args:
        dataset_path (str): Path to the dataset JSON.
        
    Returns:
        List[Dict[str, Any]]: List of query objects.
    """
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded evaluation dataset from {dataset_path}")
                return data
        except Exception as e:
            logger.error(f"Error reading dataset: {e}. Using default.")
            
    # Create directory if needed and write default
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_DATASET, f, indent=2)
    logger.info(f"Created default dataset at {dataset_path}")
    return DEFAULT_DATASET

def run_evaluation(dataset_path: str = None) -> Dict[str, Any]:
    """Runs retrieval evaluation and saves reports and plots.
    
    Args:
        dataset_path (str): Optional path to evaluation dataset.
        
    Returns:
        Dict[str, Any]: Configuration metrics dictionary.
    """
    config = load_config()
    
    if dataset_path is None:
        dataset_path = os.path.join(config["paths"]["evaluation_dir"], "dummy_dataset.json")
        
    dataset = load_or_create_dataset(dataset_path)
    
    # Extract unique relevant chunks to index for a clean evaluation environment
    logger.info("Setting up evaluation database...")
    eval_docs = []
    chunk_index = 0
    for idx, item in enumerate(dataset):
        filename = f"eval_doc_{idx+1}.pdf"
        for chunk_text in item["relevant_chunks"]:
            eval_docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": filename,
                    "page": 1,
                    "chunk_id": f"{filename}_p1_c{chunk_index}"
                }
            ))
            chunk_index += 1
            
    # Add 30 challenging distractor chunks to simulate realistic RAG retrieval
    distractor_texts = [
        "The solar system consists of the Sun and the objects that orbit it, including eight planets.",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "The Great Wall of China is a series of fortifications built across the historical northern borders of China.",
        "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.",
        "Artificial neural networks are computational models inspired by the biological structure of brains.",
        "Java is a high-level, class-based, object-oriented programming language designed to have few implementation dependencies.",
        "The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity.",
        "SQL is a domain-specific language used in programming and designed for managing data held in a relational database.",
        "Mitochondria are double-membrane-bound organelles found in most eukaryotic organisms, often called the powerhouse of the cell.",
        "The Renaissance was a fervent period of European cultural, artistic, political and scientific rebirth following the Middle Ages.",
        "Python is an interpreted, high-level, general-purpose programming language created by Guido van Rossum.",
        "Global warming refers to the long-term rise in the average temperature of the Earth's climate system.",
        "The industrial revolution was the transition to new manufacturing processes in Europe and the United States.",
        "Docker is a set of platform as a service products that use OS-level virtualization to deliver software in packages called containers.",
        "Kubernetes is an open-source container-orchestration system for automating computer application deployment, scaling, and management.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "Git is a software for tracking changes in any set of files, usually used for coordinating work among programmers.",
        "Apache Spark is an open-source unified analytics engine for large-scale data processing.",
        "Hadoop is a collection of open-source software utilities that facilitates using a network of many computers to solve problems.",
        "Machine learning is the study of computer algorithms that improve automatically through experience and by the use of data.",
        "An database is an organized collection of data, generally stored and accessed electronically from a computer system.",
        "The attention mechanism in neural networks mimics cognitive attention, dynamically focusing on specific features of input data.",
        "BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors, developed by Facebook AI Research.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with interactions.",
        "Cross-encoders perform full attention over the query and document together, yielding high precision at the expense of speed.",
        "Bi-encoders embed the query and documents independently into a shared vector space, allowing fast vector search.",
        "Reciprocal Rank Fusion is an algorithm that evaluates multiple ranked lists and aggregates them into a single consolidated ranking.",
        "Information retrieval is the activity of obtaining information system resources that are relevant to an information need.",
        "A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part."
    ]
    
    for d_idx, text in enumerate(distractor_texts):
        filename = f"distractor_doc_{d_idx+1}.pdf"
        eval_docs.append(Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": 1,
                "chunk_id": f"{filename}_p1_c{chunk_index}"
            }
        ))
        chunk_index += 1
            
    # Save a temporary evaluation index
    eval_index_dir = os.path.join(config["paths"]["evaluation_dir"], "eval_faiss_index")
    vs_manager = VectorStoreManager(
        model_name=config["embedding"]["model_name"],
        device=config["embedding"]["device"],
        index_path=eval_index_dir
    )
    vs_manager.create_and_save(eval_docs)
    
    # Load components
    faiss_store = vs_manager.load()
    documents = vs_manager.load_documents()
    
    hybrid_retriever = HybridRetriever(
        vector_store=faiss_store,
        documents=documents,
        fusion_method=config["retrieval"]["fusion_method"],
        rrf_k=config["retrieval"]["rrf_k"],
        semantic_weight=config["retrieval"]["weights"]["semantic"],
        keyword_weight=config["retrieval"]["weights"]["keyword"]
    )
    
    reranker = DocumentReranker(
        model_name=config["reranker"]["model_name"],
        device=config["reranker"]["device"]
    )
    
    results = []
    
    for item in dataset:
        query = item["question"]
        gts = item["relevant_chunks"]
        
        # --- Config 1: Semantic Only ---
        t0 = time.time()
        sem_results = faiss_store.similarity_search(query, k=5)
        sem_time = time.time() - t0
        
        # --- Config 2: Keyword Only ---
        t0 = time.time()
        tokenized_query = clean_tokenize(query)
        kw_scores = hybrid_retriever.bm25.get_scores(tokenized_query)
        kw_idx = np.argsort(kw_scores)[::-1][:5]
        kw_results = [documents[i] for i in kw_idx]
        kw_time = time.time() - t0
        
        # --- Config 3: Hybrid Only ---
        t0 = time.time()
        hyb_results = hybrid_retriever.retrieve(query, top_k=5)
        hyb_time = time.time() - t0
        
        # --- Config 4: Hybrid + Reranker ---
        t0 = time.time()
        hyb_candidates = hybrid_retriever.retrieve(query, top_k=20)
        reranked_results = reranker.rerank(query, hyb_candidates, top_k=5)
        rerank_time = time.time() - t0
        
        # Record metrics for each
        for config_name, retrieved, latency in [
            ("Semantic_Only", sem_results, sem_time),
            ("Keyword_Only", kw_results, kw_time),
            ("Hybrid_Only", hyb_results, hyb_time),
            ("Hybrid_Reranked", reranked_results, rerank_time)
        ]:
            results.append({
                "question": query,
                "config": config_name,
                "precision@5": evaluate_precision_at_k(retrieved, gts, 5),
                "recall@5": evaluate_recall_at_k(retrieved, gts, 5),
                "hit_rate@5": evaluate_hit_rate(retrieved, gts, 5),
                "mrr": evaluate_mrr(retrieved, gts),
                "ndcg@5": evaluate_ndcg_at_k(retrieved, gts, 5),
                "retrieval_latency": latency
            })
            
    df = pd.DataFrame(results)
    
    # Calculate average metrics per config
    summary = df.groupby("config").mean(numeric_only=True).reset_index()
    logger.info(f"Evaluation summary:\n{summary}")
    
    # Save evaluation report CSV
    report_file = config["paths"]["report_file"]
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    df.to_csv(report_file, index=False)
    logger.info(f"Saved evaluation report to {report_file}")
    
    # Save metrics JSON
    metrics_dict = {}
    for idx, row in summary.iterrows():
        metrics_dict[row["config"]] = {
            "precision@5": float(row["precision@5"]),
            "recall@5": float(row["recall@5"]),
            "hit_rate@5": float(row["hit_rate@5"]),
            "mrr": float(row["mrr"]),
            "ndcg@5": float(row["ndcg@5"]),
            "retrieval_latency_ms": float(row["retrieval_latency"] * 1000)
        }
    
    metrics_file = config["paths"]["metrics_file"]
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"Saved metrics summary to {metrics_file}")
    
    # Generate Plots
    generate_comparison_plots(summary, config["paths"]["plot_dir"])
    
    return metrics_dict

def generate_comparison_plots(summary_df: pd.DataFrame, output_dir: str):
    """Generates comparison bar plots for retrieval configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Accuracy Metrics Comparison
    metrics = ["precision@5", "recall@5", "hit_rate@5", "mrr", "ndcg@5"]
    melted_df = pd.melt(summary_df, id_vars=["config"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Metric", y="Score", hue="config", data=melted_df, palette="viridis")
    plt.title("Retrieval Performance Comparison Across Configurations", fontsize=14, fontweight="bold", pad=15)
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 8),
                    textcoords='offset points',
                    fontsize=9, fontweight='semibold')
                    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "retrieval_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved metrics comparison plot to {plot_path}")
    
    # Plot 2: Latency Comparison
    plt.figure(figsize=(8, 5))
    summary_df["retrieval_latency_ms"] = summary_df["retrieval_latency"] * 1000
    ax = sns.barplot(x="config", y="retrieval_latency_ms", data=summary_df, palette="coolwarm")
    plt.title("Average Retrieval Latency (ms)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.xlabel("Configuration", fontsize=12)
    
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(f'{height:.1f} ms',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 8),
                    textcoords='offset points',
                    fontsize=10, fontweight='semibold')
                    
    plt.tight_layout()
    latency_plot_path = os.path.join(output_dir, "retrieval_latency_comparison.png")
    plt.savefig(latency_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved latency comparison plot to {latency_plot_path}")

if __name__ == "__main__":
    run_evaluation()
