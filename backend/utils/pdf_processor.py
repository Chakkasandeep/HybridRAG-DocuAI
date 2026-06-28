import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.utils.logging_utils import setup_logger
from typing import List, Union, Any

logger = setup_logger("pdf_processor")

def process_pdfs(pdf_docs: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Processes PDF files, splits them into chunks, and preserves metadata (filename, page, chunk_id).
    
    Args:
        pdf_docs (List[Any]): List of PDF file-like objects (e.g. from Streamlit) or file paths.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap size of text chunks.
        
    Returns:
        List[Document]: List of Langchain Document objects with metadata.
    """
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for pdf in pdf_docs:
        # Determine filename
        if isinstance(pdf, str):
            filename = os.path.basename(pdf)
            pdf_reader = PdfReader(pdf)
        else:
            filename = getattr(pdf, "name", "uploaded_file.pdf")
            pdf_reader = PdfReader(pdf)
            
        logger.info(f"Processing PDF: {filename} with {len(pdf_reader.pages)} pages")
        
        for page_idx, page in enumerate(pdf_reader.pages):
            page_num = page_idx + 1
            text = page.extract_text()
            if not text or not text.strip():
                continue
                
            # Split page text
            chunks = splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "chunk_id": chunk_id
                    }
                )
                documents.append(doc)
                
    logger.info(f"Generated {len(documents)} document chunks in total")
    return documents
