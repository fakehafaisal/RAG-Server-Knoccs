from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import traceback
from datetime import datetime, timezone
import os

from src.config import ServerConfig, LLMConfig, ChunkingConfig, SearchConfig, validate_config
from src.logger import get_server_logger
from src.search import RAGSearch
from src.vectorstore import PgVectorStore
from src.data_loader import load_all_documents
from src.deepeval import DeepEvalService

# Initialize logger
logger = get_server_logger()

# Global instances
rag_search: Optional[RAGSearch] = None
deepeval_service: Optional[DeepEvalService] = None


# ==================== LIFESPAN (replaces deprecated on_event) ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    global rag_search, deepeval_service

    # ===== STARTUP =====
    logger.info("Starting Knoccs RAG API...")

    # Validate configuration before proceeding
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    try:
        # Initialize RAG search
        store = PgVectorStore()
        stats = store.get_stats()
        if stats['total_chunks'] > 0:
            rag_search = RAGSearch(
                llm_model=LLMConfig.MODEL,
                use_query_expansion=False,
                debug=False,
                vectorstore=store
            )
            logger.info(f"RAG system initialized with {stats['total_chunks']} chunks")
        else:
            logger.warning("Knowledge base is empty. Please build it first.")

        # Initialize DeepEval service
        try:
            deepeval_service = DeepEvalService(rag_search, ground_truth_file=ServerConfig.GROUND_TRUTH_FILE)
            logger.info(f"DeepEval initialized with {LLMConfig.DEEPEVAL_MODEL}")
        except Exception as e:
            logger.warning(f"DeepEval initialization failed: {e}")
            logger.warning("Continuing without DeepEval - evaluation endpoint may fail")
            deepeval_service = None

    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        traceback.print_exc()

    yield  # Server is running

    # ===== SHUTDOWN =====
    logger.info("Shutting down Knoccs RAG API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Knoccs RAG API",
    description="RAG-powered Knowledge Base API with Supabase pgvector",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== REQUEST/RESPONSE MODELS ====================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    top_k: Optional[int] = Field(default=SearchConfig.TOP_K, ge=1, le=100)
    use_query_expansion: Optional[bool] = False
    return_sources: Optional[bool] = True
    conversation_history: Optional[List[Dict]] = Field(default=[], max_length=50)

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict]] = None
    timestamp: str
    query_expansions: Optional[List[str]] = None
    conversation_history: Optional[List[Dict]] = []


class RebuildRequest(BaseModel):
    chunk_size: Optional[int] = ChunkingConfig.SIZE
    chunk_overlap: Optional[int] = ChunkingConfig.OVERLAP


class StatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    kb_status: str


class EvaluateRequest(BaseModel):
    query: str
    top_k: Optional[int] = SearchConfig.TOP_K


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
        logger.error(f"Health check failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again later.")


# ==================== QUERY ENDPOINT ====================
@app.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    global rag_search

    if rag_search is None:
        try:
            rag_search = RAGSearch(llm_model=LLMConfig.MODEL, use_query_expansion=False)
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {str(e)}")
            traceback.print_exc()
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please build the knowledge base first."
            )

    rag_search.use_query_expansion = request.use_query_expansion

    try:
        # Get expanded queries if needed
        query_expansions = None
        if request.use_query_expansion:
            query_expansions = rag_search.expand_query(request.query)

        # Get answer from RAG (returns tuple: answer, conversation_history)
        answer, updated_history = rag_search.search_and_summarize(
            request.query,
            top_k=request.top_k,
            initial_k=request.top_k * 3,
            conversation_history=request.conversation_history
        )

        # Get sources if requested (deduplicated by document)
        sources = None
        if request.return_sources:
            raw_results = rag_search.search_only(request.query, top_k=request.top_k)
            seen_sources = {}
            for r in raw_results:
                source_key = r.get("metadata", {}).get("source", "unknown")
                score = r.get("rerank_score") or 0
                if source_key not in seen_sources or score > (seen_sources[source_key].get("rerank_score") or 0):
                    seen_sources[source_key] = {
                        "text": r.get("metadata", {}).get("text", ""),
                        "source": source_key,
                        "document_name": r.get("metadata", {}).get("document_name", "unknown"),
                        "document_type": r.get("metadata", {}).get("document_type", "unknown"),
                        "chunk_id": r.get("chunk_id"),
                        "rerank_score": r.get("rerank_score")
                    }
            sources = list(seen_sources.values())

        return QueryResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            query_expansions=query_expansions,
            conversation_history=updated_history
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your query. Please try again.")


# ============= DEEPEVAL ENDPOINT (WITH GROUND TRUTH) =============

@app.post("/rag/evaluate")
def evaluate_query_deepeval(request: EvaluateRequest):
    """
    Evaluate a query using DeepEval metrics (3 standard RAG metrics)
    Reference-based evaluation - uses ground truth from ground_truth.json

    Returns metrics scores: Answer Relevancy, Faithfulness, Contextual Precision
    """
    global rag_search, deepeval_service

    logger.info(f"Evaluate endpoint called - Query: {request.query[:60]}...")

    if rag_search is None:
        logger.error("RAG search is None - RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please rebuild knowledge base first.")

    if deepeval_service is None:
        logger.error("DeepEval service is None - not initialized at startup")
        raise HTTPException(status_code=503, detail="DeepEval service not initialized. Check server logs.")

    try:
        logger.info("Running evaluation with 3 metrics...")
        result = deepeval_service.evaluate_query(request.query)

        result['timestamp'] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Format metrics for frontend
        metrics_summary = {}

        # Check if there's a retrieval error
        if 'error' in result.get('metrics', {}):
            metrics_summary = result['metrics']
        else:
            # Process each metric normally
            for metric_name, metric_data in result.get('metrics', {}).items():
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
                    logger.warning(f"Unexpected metric_data type for {metric_name}: {type(metric_data)}")
                    continue

        response = {
            "query": result['query'],
            "answer": result['answer'],
            "expected_answer": result.get('expected_answer'),
            "num_retrieved_chunks": result['num_retrieved_chunks'],
            "metrics": metrics_summary,
            "timestamp": result['timestamp']
        }

        logger.info("Evaluation complete!")

        return response

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred during evaluation. Please try again.")


# ============= KNOWLEDGE BASE MANAGEMENT =============

@app.get("/rag/stats", response_model=StatsResponse)
def get_stats():
    """Get knowledge base statistics"""
    global rag_search

    try:
        if rag_search and hasattr(rag_search, 'vectorstore'):
            stats = rag_search.vectorstore.get_stats()
        else:
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
        logger.error(f"Stats endpoint failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching statistics. Please try again.")


@app.post("/rag/rebuild")
def rebuild_knowledge_base(request: RebuildRequest):
    """Rebuild the entire knowledge base from the data/ folder"""
    global rag_search
    try:
        # Load documents
        docs = load_all_documents(ServerConfig.DATA_FOLDER)

        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found in {ServerConfig.DATA_FOLDER}/ folder"
            )

        # Build vector store
        store = PgVectorStore(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            use_reranker=True,
            debug=False
        )

        store.clear()
        store.build_from_documents(docs)
        rag_search = RAGSearch(llm_model=LLMConfig.MODEL, use_query_expansion=False, vectorstore=store)
        stats = store.get_stats()

        return {
            "message": "Knowledge base rebuilt successfully",
            "total_documents": len(docs),
            "total_chunks": stats['total_chunks'],
            "total_sources": stats['total_sources'],
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

    except Exception as e:
        logger.error(f"Rebuild failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while rebuilding the knowledge base. Please try again.")


@app.delete("/rag/clear")
def clear_knowledge_base():
    """Clear the entire knowledge base"""
    global rag_search
    try:
        store = PgVectorStore()
        store.clear()

        rag_search = None

        return {
            "message": "Knowledge base cleared successfully",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
    except Exception as e:
        logger.error(f"Clear failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while clearing the knowledge base. Please try again.")


# ============= ERROR HANDLERS =============

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={
        "error": exc.detail if hasattr(exc, 'detail') else "HTTP error",
        "path": str(request.url)
    })


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={
        "error": "Internal server error",
        "detail": "An unexpected error occurred. Please try again later."
    })


# Serve favicon if requested
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

    uvicorn.run(
        "server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=True,
        log_level="info"
    )
