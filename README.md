📚 DocuAI – Chat with Your PDFs
=======================================================================

An AI-powered multi-PDF reader built with Streamlit + LangChain + Groq.  
It uses **Retrieval-Augmented Generation (RAG)** with FAISS, HuggingFace 
Embeddings, and Groq LLaMA models to let you chat with your documents 
and get context-aware answers instantly.

-----------------------------------------------------------------------

🔗 **Live Demo**: [DocuAI Chat with PDFs](https://docuai-chat-with-your-pdfs.streamlit.app/)  

-----------------------------------------------------------------------


🚀 Features:
- Upload multiple PDFs and extract content
- Split text into chunks for efficient retrieval
- Store & query embeddings using FAISS Vector Database
- Semantic search powered by HuggingFace embeddings
- Contextual answers generated using Groq LLaMA-3.3 models
- Simple, interactive Streamlit UI

-----------------------------------------------------------------------
🛠️ Tech Stack:
- Frontend/UI   : Streamlit
- PDF Parsing   : PyPDF2
- LLM           : Groq API (LLaMA-3.3-70B / LLaMA-3-8B)
- Framework     : LangChain
- Embeddings    : HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- Vector Store  : FAISS

-----------------------------------------------------------------------
⚡ Requirements:
- Python >= 3.9
- streamlit
- PyPDF2
- python-dotenv
- langchain
- langchain-community
- langchain-groq
- sentence-transformers
- faiss-cpu

(Install all using: pip install -r requirements.txt)

-----------------------------------------------------------------------
▶️ Process Flow:
1. Upload PDFs → Extract text with PyPDF2
2. Split text into chunks with RecursiveCharacterTextSplitter
3. Convert chunks to embeddings using HuggingFace
4. Store embeddings in FAISS vector database
5. On user query → Search top relevant chunks
6. Pass chunks + query → Groq LLaMA model
7. Streamlit displays the final answer
-----------------------------------------------------------------------
**Add Your API Keys**
```
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here
```

**▶️ Run the App**
```
streamlit run app.py
```
**Clone the repo:**
```
    git clone https://github.com/your-username/docuai-pdf-chat.git
    cd docuai-pdf-chat
```
