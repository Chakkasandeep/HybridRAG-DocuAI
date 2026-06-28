from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from backend.utils.logging_utils import setup_logger
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = setup_logger("llm_manager")

class LLMManager:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
        """Initializes the LLM Manager.
        
        Args:
            groq_api_key (str): Groq API key.
            model_name (str): Groq LLM model name.
            temperature (float): Generation temperature.
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.temperature = temperature
        
        if not self.groq_api_key:
            logger.error("Groq API Key is missing!")
            
    def get_qa_chain(self):
        """Creates and returns the LangChain QA chain with a custom prompt instructing citations.
        
        Returns:
            Any: LangChain QA chain instance.
        """
        prompt_template = """You are an expert AI assistant. Answer the user's question as detailed and accurately as possible, based ONLY on the provided context.
        
CRITICAL RULES:
1. Base your answer STRICTLY on the facts directly mentioned in the context. Do not make up answers, extrapolate, or assume anything.
2. The provided context is extracted from the user's uploaded PDF files. If the user asks about "the PDF", "the document", or requests a summary of the uploaded files, you should describe or summarize the information present in the context, as the context represents the contents of those documents.
3. If the answer (or topic) is not contained in the context at all, clearly state: "The answer is not available in the context."
4. Inline Citations: You MUST cite your sources inside the text of your answer. Whenever you use facts from a source, append an inline citation in the format `[filename.pdf, Page X]` (e.g., `[contract.pdf, Page 4]`). Match the source filename and page number exactly from the context header.

Context:
{context}

Question:
{question}

Answer:"""
        
        llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Document prompt to ensure the source filename and page number are injected into {context}
        doc_prompt = PromptTemplate(
            template="[Source: {source}, Page: {page}]\nContent: {page_content}\n---",
            input_variables=["page_content", "source", "page"]
        )
        
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt, document_prompt=doc_prompt)
        return chain
        
    def generate_answer(self, chain, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generates an answer using the QA chain and retrieved documents.
        
        Args:
            chain: The QA chain.
            query (str): Question query.
            documents (List[Document]): Chunks to pass to the LLM.
            
        Returns:
            Dict[str, Any]: Answer text and list of source documents.
        """
        logger.info(f"Generating answer for query: {query}")
        
        # Prompt to structure each document inside the context
        doc_prompt = PromptTemplate(
            template="[Source: {source}, Page: {page}]\nContent: {page_content}\n---",
            input_variables=["page_content", "source", "page"]
        )
        
        response = chain(
            {
                "input_documents": documents,
                "question": query,
                "document_prompt": doc_prompt
            },
            return_only_outputs=True
        )
        
        return {
            "answer": response["output_text"],
            "sources": documents
        }
