import os
import json
from typing import Dict, Any, Optional
import random
from langchain_groq import ChatGroq
from backend.utils.config_loader import load_config
from backend.evaluation.evaluator import run_evaluation
from backend.retrieval.vector_store import VectorStoreManager
from backend.utils.logging_utils import setup_logger

logger = setup_logger("evaluation_service")

class EvaluationService:
    def __init__(self):
        self.config = load_config()
        self.metrics_file = os.path.abspath(self.config["paths"]["metrics_file"])
        self.index_path = os.path.abspath(self.config["paths"]["faiss_index"])
        
    def _generate_synthetic_dataset(self, api_key: str) -> Optional[str]:
        """Generates a synthetic evaluation dataset based on the user's indexed PDFs."""
        logger.info("Attempting to generate synthetic dataset from user PDFs...")
        try:
            # 1. Load current indexed documents
            docs_file = os.path.join(self.index_path, "documents.pkl")
            if not os.path.exists(docs_file):
                logger.warning("No user documents indexed yet. Falling back to dummy dataset.")
                return None
                
            vs_manager = VectorStoreManager(
                model_name=self.config["embedding"]["model_name"],
                device=self.config["embedding"]["device"],
                index_path=self.index_path
            )
            documents = vs_manager.load_documents()
            if not documents:
                logger.warning("Indexed document list is empty.")
                return None

            # 2. Select 3 diverse chunks from the document list
            num_docs = len(documents)
            selected_indices = []
            if num_docs <= 3:
                selected_indices = list(range(num_docs))
            else:
                # Pick spaced chunks
                selected_indices = [num_docs // 4, num_docs // 2, (3 * num_docs) // 4]
                
            selected_docs = [documents[i] for i in selected_indices]
            logger.info(f"Selected {len(selected_docs)} chunks for question generation.")

            # 3. Instantiate Groq Chat Model
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=self.config["llm"]["model_name"],
                temperature=0.3
            )

            synthetic_dataset = []
            for idx, doc in enumerate(selected_docs):
                source_name = doc.metadata.get("source", f"doc_{idx}")
                content = doc.page_content.strip()
                
                # Ask LLM to generate a question
                prompt = (
                    "You are an expert evaluator. Based on the following document chunk, generate exactly one direct, clear question that can be answered using ONLY the information in this chunk.\n"
                    "Do not refer to 'the text', 'the chunk', or 'the document' in your question. Make it sound like a natural query a user would ask.\n\n"
                    f"Document Chunk (from {source_name}):\n{content}\n\n"
                    "Question:"
                )
                
                response = llm.invoke(prompt)
                question = response.content.strip().strip('"')
                
                synthetic_dataset.append({
                    "question": question,
                    "ground_truth": content,
                    "relevant_chunks": [content]
                })
                logger.info(f"Generated Question {idx+1}: {question}")

            # 4. Save synthetic dataset to a JSON file
            dataset_path = os.path.join(self.config["paths"]["evaluation_dir"], "user_dataset.json")
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(synthetic_dataset, f, indent=2)
                
            logger.info(f"Successfully generated synthetic dataset at {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Error generating synthetic dataset: {e}. Falling back to default.")
            return None

    def trigger_evaluation(self, override_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Runs the RAG evaluation benchmarks."""
        logger.info("Triggering evaluation suite...")
        api_key = override_api_key or os.getenv("GROQ_API_KEY")
        
        # Try generating dynamic dataset first if API key is present
        dataset_path = None
        if api_key:
            dataset_path = self._generate_synthetic_dataset(api_key)
            
        try:
            # If dataset_path is None, run_evaluation falls back to dummy_dataset.json
            metrics_dict = run_evaluation(dataset_path=dataset_path)
            return {
                "status": "success",
                "message": "Evaluation suite ran successfully on " + ("user PDFs" if dataset_path else "default dummy data") + ".",
                "results": metrics_dict
            }
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            return {
                "status": "error",
                "message": f"Error running evaluation suite: {str(e)}"
            }

    def get_results(self) -> Dict[str, Any]:
        """Loads evaluation results from metrics.json."""
        if not os.path.exists(self.metrics_file):
            logger.warning(f"Metrics file {self.metrics_file} not found.")
            return {
                "status": "error",
                "message": "No evaluation reports generated yet. Please run evaluation first.",
                "results": {}
            }
            
        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            return {
                "status": "success",
                "results": metrics_data
            }
        except Exception as e:
            logger.error(f"Error reading metrics file: {e}")
            return {
                "status": "error",
                "message": f"Error reading metrics report: {str(e)}",
                "results": {}
            }
