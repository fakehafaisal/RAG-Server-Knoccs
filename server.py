from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from datetime import datetime
import os
from pathlib import Path

from src.search import RAGSearch
from src.vectorstore import PgVectorStore
from src.data_loader import load_all_documents

# Initialize FastAPI app
app = FastAPI(
    title="Knoccs RAG API",
    description="RAG-powered Knowledge Base API with Supabase pgvector",
    version="1.0.0"
)

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],  # Add your Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance (loaded on startup)
rag_search: Optional[RAGSearch] = None
DATA_FOLDER = "data"

# ============= REQUEST/RESPONSE MODELS =============

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_query_expansion: Optional[bool] = True
    return_sources: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None
    timestamp: str
    query_expansions: Optional[List[str]] = None

class RebuildRequest(BaseModel):
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 128

class StatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    kb_status: str

class DocumentInfo(BaseModel):
    filename: str
    source: str
    chunks: int
    uploaded_at: str

# ============= STARTUP/SHUTDOWN =============

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_search
    try:
        # Check if KB exists
        store = PgVectorStore()
        stats = store.get_stats()
        
        if stats['total_chunks'] > 0:
            rag_search = RAGSearch(use_query_expansion=True)
            print(f"[INFO] RAG system initialized with {stats['total_chunks']} chunks")
        else:
            print("[WARNING] Knowledge base is empty. Please build it first.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG system: {e}")
        print("[INFO] You can still use the API to build the knowledge base")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("[INFO] Shutting down RAG API")

# ============= HEALTH CHECK =============

@app.get("/")
def root():
    """Root endpoint - health check"""
    return {
        "service": "Knoccs RAG API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/rag/query",
            "search": "/rag/search",
            "stats": "/rag/stats",
            "rebuild": "/rag/rebuild",
            "clear": "/rag/clear"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        store = PgVectorStore()
        stats = store.get_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "kb_chunks": stats['total_chunks'],
            "rag_loaded": rag_search is not None
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# ============= RAG QUERY ENDPOINTS =============

@app.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Main RAG query endpoint - asks a question to the knowledge base
    """
    global rag_search
    
    # Check if RAG is initialized
    if rag_search is None:
        try:
            rag_search = RAGSearch(use_query_expansion=request.use_query_expansion)
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"RAG system not initialized. Please build knowledge base first. Error: {str(e)}"
            )
    
    # Update query expansion setting
    rag_search.use_query_expansion = request.use_query_expansion
    
    try:
        # Get expanded queries if needed
        query_expansions = None
        if request.use_query_expansion:
            query_expansions = rag_search.expand_query(request.query)
        
        # Get answer
        answer = rag_search.search_and_summarize(
            request.query, 
            top_k=request.top_k,
            initial_k=request.top_k * 3
        )
        
        # Get sources if requested
        sources = None
        if request.return_sources:
            sources = rag_search.search_only(request.query, top_k=request.top_k)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query_expansions=query_expansions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/rag/search")
def search_only(request: QueryRequest):
    """
    Search-only endpoint - returns raw search results without LLM summarization
    Useful for debugging or building custom UI
    """
    global rag_search
    
    if rag_search is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = rag_search.search_only(request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ============= KNOWLEDGE BASE MANAGEMENT =============

@app.get("/rag/stats", response_model=StatsResponse)
def get_stats():
    """
    Get statistics about the knowledge base
    """
    try:
        store = PgVectorStore()
        stats = store.get_stats()
        
        kb_status = "empty"
        if stats['total_chunks'] > 0:
            kb_status = "ready"
        
        return StatsResponse(
            total_chunks=stats['total_chunks'],
            total_sources=stats['total_sources'],
            kb_status=kb_status
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/rag/rebuild")
def rebuild_knowledge_base(request: RebuildRequest):
    """
    Rebuild the entire knowledge base from the data/ folder
    """
    global rag_search
    
    try:
        # Load documents
        docs = load_all_documents(DATA_FOLDER)
        
        if not docs:
            raise HTTPException(
                status_code=404, 
                detail=f"No documents found in {DATA_FOLDER}/ folder"
            )
        
        # Build vector store
        store = PgVectorStore(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            use_reranker=True
        )
        store.build_from_documents(docs)
        
        # Reinitialize RAG
        rag_search = RAGSearch(use_query_expansion=True)
        
        # Get final stats
        stats = store.get_stats()
        
        return {
            "message": "Knowledge base rebuilt successfully",
            "total_documents": len(docs),
            "total_chunks": stats['total_chunks'],
            "total_sources": stats['total_sources'],
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

@app.delete("/rag/clear")
def clear_knowledge_base():
    """
    Clear all data from the knowledge base (WARNING: This is irreversible!)
    """
    global rag_search
    
    try:
        store = PgVectorStore()
        store.clear()
        
        # Reset RAG instance
        rag_search = None
        
        return {
            "message": "Knowledge base cleared successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

# Document management endpoints removed - not needed for this implementation

# ============= ERROR HANDLERS =============

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "path": str(request.url)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

# ============= MAIN =============

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )