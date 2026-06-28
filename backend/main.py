import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes import chat, document, evaluate

# Load env variables
load_dotenv()

app = FastAPI(
    title="DocuAI - Hybrid RAG Backend",
    description="Production-style API backend for document assistant with Hybrid RAG & CrossEncoder reranking.",
    version="1.0.0"
)

# Configure CORS
# React frontend typically runs on http://localhost:5173 (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router)
app.include_router(document.router)
app.include_router(evaluate.router)

# Mount static folder for evaluation plots if they exist
eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation"))
if os.path.exists(eval_dir):
    app.mount("/static", StaticFiles(directory=eval_dir), name="static")

@app.get("/")
async def root():
    return {
        "app": "DocuAI - Hybrid RAG API Backend",
        "status": "online",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
