from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import traceback
from datetime import datetime
import os

from src.search import RAGSearch
from src.vectorstore import PgVectorStore
from src.data_loader import load_all_documents
from src.deepeval import DeepEvalService

# Initialize FastAPI app
app = FastAPI(
    title="Knoccs RAG API",
    description="RAG-powered Knowledge Base API with Supabase pgvector",
    version="1.0.0"
)

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_search: Optional[RAGSearch] = None
deepeval_service: Optional[DeepEvalService] = None
DATA_FOLDER = "data"
GROUND_TRUTH_FILE = "ground_truth.json"

# ==================== REQUEST/RESPONSE MODELS ====================
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 15
    use_query_expansion: Optional[bool] = False
    return_sources: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None
    timestamp: str
    query_expansions: Optional[List[str]] = None

class RebuildRequest(BaseModel):
    chunk_size: Optional[int] = 1024
    chunk_overlap: Optional[int] = 256

class StatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    kb_status: str

class EvaluateRequest(BaseModel):
    query: str
    top_k: Optional[int] = 15

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system and DeepEval service on startup"""
    global rag_search, deepeval_service
    try:
        # Initialize RAG search
        store = PgVectorStore()
        stats = store.get_stats()
        if stats['total_chunks'] > 0:
            rag_search = RAGSearch(llm_model="gpt-5", use_query_expansion=False, debug=False, vectorstore=store)
            print(f"RAG system initialized with {stats['total_chunks']} chunks")
        else:
            print("[WARNING] Knowledge base is empty. Please build it first.")
        
        # Initialize DeepEval service (loads ground truth if available)
        try:
            deepeval_service = DeepEvalService(rag_search, ground_truth_file=GROUND_TRUTH_FILE)
            print("[DeepEval service initialized with ground truth support")
        except Exception as e:
            print(f"[WARNING] DeepEval initialization failed: {e}")
            print("[WARNING] Continuing without DeepEval - evaluation endpoint may fail")
            deepeval_service = None
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize systems: {e}")
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down RAG API")

# ==================== HEALTH CHECK ====================
@app.get("/")
def root():
    """Root endpoint - service info"""
    return {
        "service": "Knoccs RAG API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/rag/query",
            "search": "/rag/search",
            "stats": "/rag/stats",
            "rebuild": "/rag/rebuild",
            "clear": "/rag/clear",
            "evaluate": "/rag/evaluate"
        }
    }

@app.get("/health")
def health_check():
    global rag_search
    
    try:
        # Reuse existing vectorstore if available
        if rag_search and hasattr(rag_search, 'vectorstore'):
            stats = rag_search.vectorstore.get_stats()
        else:
            store = PgVectorStore(use_reranker=False)
            stats = store.get_stats()
            
        return {
            "status": "healthy",
            "database": "connected",
            "kb_chunks": stats['total_chunks'],
            "rag_loaded": rag_search is not None
        }
    except Exception as e:
        print(f"[ERROR] Health check failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# ==================== QUERY ENDPOINT ====================
@app.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    global rag_search

    if rag_search is None:
        try:
            rag_search = RAGSearch(llm_model="gpt-5", use_query_expansion=False)
        except Exception as e:
            print(f"[ERROR] Failed to initialize RAG: {str(e)}")
            traceback.print_exc()
            raise HTTPException(
                status_code=503,
                detail=f"RAG system not initialized. Please build knowledge base first. Error: {str(e)}"
            )

    rag_search.use_query_expansion = request.use_query_expansion

    try:
        # Get expanded queries if needed
        query_expansions = None
        if request.use_query_expansion:
            query_expansions = rag_search.expand_query(request.query)

        # Get answer from RAG
        answer = rag_search.search_and_summarize(
            request.query,
            top_k=request.top_k,
            initial_k=request.top_k * 3
        )

        # Get sources if requested
        sources = None
        if request.return_sources:
            raw_results = rag_search.search_only(request.query, top_k=request.top_k)
            sources = [
                {
                    "text": r.get("metadata", {}).get("text", ""),
                    "source": r.get("metadata", {}).get("source", "unknown"),
                    "document_name": r.get("metadata", {}).get("document_name", "unknown"),
                    "document_type": r.get("metadata", {}).get("document_type", "unknown"),
                    "chunk_id": r.get("chunk_id"),
                    "rerank_score": r.get("rerank_score")
                } for r in raw_results
            ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query_expansions=query_expansions
        )

    except Exception as e:
        print(f"[ERROR] Query failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ============= DEEPEVAL ENDPOINT (WITH GROUND TRUTH) =============

@app.post("/rag/evaluate")
def evaluate_query_deepeval(request: EvaluateRequest):
    """
    Evaluate a query using DeepEval metrics (3 standard RAG metrics)
    Referenceless evaluation - no ground truth needed
    
    Returns metrics scores: Answer Relevancy, Faithfulness, Contextual Precision
    """
    global rag_search, deepeval_service
    
    print(f"\n[INFO] ========== EVALUATE ENDPOINT CALLED ==========")
    print(f"[INFO] Query: {request.query[:60]}...")
    
    if rag_search is None:
        print("[ERROR] RAG search is None - RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please rebuild knowledge base first.")
    
    if deepeval_service is None:
        print("[ERROR] DeepEval service is None - not initialized at startup")
        raise HTTPException(status_code=503, detail="DeepEval service not initialized. Check server logs.")
    
    try:
        print("[INFO] Running evaluation with 3 metrics (referenceless mode)...")
        result = deepeval_service.evaluate_query(request.query)
        
        result['timestamp'] = datetime.utcnow().isoformat() + "Z"
        
        # Format metrics for frontend
        metrics_summary = {}
        
        # Check if there's a retrieval error
        if 'error' in result.get('metrics', {}):
            # If retrieval failed, just return the error
            metrics_summary = result['metrics']
        else:
            # Process each metric normally
            for metric_name, metric_data in result.get('metrics', {}).items():
                # Ensure metric_data is a dictionary before calling .get()
                if isinstance(metric_data, dict):
                    if 'error' not in metric_data:
                        metrics_summary[metric_name] = {
                            "score": metric_data.get('score', 0),
                            "passed": metric_data.get('passed', False),
                            "reason": metric_data.get('reason'),
                            "description": metric_data.get('description', '')
                        }
                    else:
                        metrics_summary[metric_name] = {
                            "score": 0,
                            "passed": False,
                            "error": metric_data.get('error'),
                            "description": metric_data.get('description', '')
                        }
                else:
                    # If metric_data is not a dict, skip it
                    print(f"[WARNING] Unexpected metric_data type for {metric_name}: {type(metric_data)}")
                    continue
        
        response = {
            "query": result['query'],
            "answer": result['answer'],
            "expected_answer": result.get('expected_answer'),  # Always None in referenceless mode
            "num_retrieved_chunks": result['num_retrieved_chunks'],
            "metrics": metrics_summary,
            "timestamp": result['timestamp']
        }
        
        print(f"[INFO] ✓ Evaluation complete!")
        print(f"[INFO] ========== EVALUATE ENDPOINT COMPLETE ==========\n")
        
        return response
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# ============= KNOWLEDGE BASE MANAGEMENT =============

@app.get("/rag/stats", response_model=StatsResponse)
def get_stats():
    """Get knowledge base statistics"""
    global rag_search
    
    try:
        # Reuse existing vectorstore if RAG is already initialized
        if rag_search and hasattr(rag_search, 'vectorstore'):
            stats = rag_search.vectorstore.get_stats()
        else:
            # Create new instance WITHOUT reranker (we just need stats)
            store = PgVectorStore(use_reranker=False)
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
        print(f"[ERROR] Stats endpoint failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/rag/rebuild")
def rebuild_knowledge_base(request: RebuildRequest):
    """Rebuild the entire knowledge base from the data/ folder"""
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
            use_reranker=True,
            debug=True 
        )
        
        store.clear()  # Clear existing data first
        store.build_from_documents(docs)  # Build new data
        rag_search = RAGSearch(llm_model="gpt-5", use_query_expansion=False, vectorstore=store)  # Reinitialize RAG
        stats = store.get_stats()  # Get final stats
        
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
        print(f"[ERROR] Rebuild failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

@app.delete("/rag/clear")
def clear_knowledge_base():
    """Clear the entire knowledge base"""
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
        print(f"[ERROR] Clear failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

# ============= ERROR HANDLERS =============

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={
        "error": exc.detail if hasattr(exc, 'detail') else "HTTP error",
        "path": str(request.url)
    })


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    # Log stack trace for debugging
    print(f"[ERROR] Unhandled exception: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={
        "error": "Internal server error",
        "detail": str(exc)
    })

# Serve favicon if requested (avoid 500s from missing favicons)
@app.get("/favicon.ico")
async def favicon():
    ico_path = "favicon.ico"
    if os.path.exists(ico_path):
        return FileResponse(ico_path)
    else:
        return Response(status_code=204)

# ============= MAIN =============

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )