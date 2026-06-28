import os
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from backend.utils.logging_utils import setup_logger
from typing import List
from langchain_core.documents import Document

logger = setup_logger("vector_store")

class VectorStoreManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu", index_path: str = "faiss_index"):
        """Initializes the VectorStoreManager.
        
        Args:
            model_name (str): SentenceTransformers embedding model.
            device (str): Device to run embeddings model on ('cpu' or 'cuda').
            index_path (str): Path to save/load FAISS index.
        """
        self.model_name = model_name
        self.device = device
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device}
        )
        
    def create_and_save(self, documents: List[Document]) -> FAISS:
        """Creates a FAISS vector store from documents and saves it locally, along with raw documents.
        
        Args:
            documents (List[Document]): Chunks to index.
            
        Returns:
            FAISS: Created vector store instance.
        """
        logger.info(f"Creating FAISS vector store with {len(documents)} documents using {self.model_name}")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save FAISS index
        vector_store.save_local(self.index_path)
        logger.info(f"Saved FAISS index to {self.index_path}")
        
        # Save raw documents to a pickle file inside the index directory
        docs_file = os.path.join(self.index_path, "documents.pkl")
        with open(docs_file, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Saved raw documents pickle to {docs_file}")
        
        return vector_store
        
    def load(self) -> FAISS:
        """Loads the FAISS vector store from the local directory.
        
        Returns:
            FAISS: Loaded vector store instance.
        """
        logger.info(f"Loading FAISS index from {self.index_path}")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index directory {self.index_path} not found.")
            
        vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
        
    def load_documents(self) -> List[Document]:
        """Loads the raw documents from the index directory.
        
        Returns:
            List[Document]: List of raw chunks.
        """
        docs_file = os.path.join(self.index_path, "documents.pkl")
        if not os.path.exists(docs_file):
            raise FileNotFoundError(f"Documents pickle file {docs_file} not found.")
            
        with open(docs_file, "rb") as f:
            documents = pickle.load(f)
        return documents
