import os
import yaml
from typing import Any, Dict

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads and returns the configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        return {
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "retrieval": {
                "top_k_retrieve": 20,
                "fusion_method": "rrf",
                "rrf_k": 60,
                "weights": {"semantic": 0.5, "keyword": 0.5}
            },
            "reranker": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
                "top_k_rerank": 5
            },
            "llm": {
                "model_name": "llama-3.3-70b-versatile",
                "temperature": 0.2
            },
            "paths": {
                "faiss_index": "faiss_index",
                "evaluation_dir": "evaluation",
                "metrics_file": "evaluation/metrics.json",
                "report_file": "evaluation/evaluation_report.csv",
                "plot_dir": "evaluation"
            }
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
