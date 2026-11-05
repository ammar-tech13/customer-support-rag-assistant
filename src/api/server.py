"""FastAPI server for the RAG system."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.vector_store.faiss_store import FAISSVectorStore
from src.agents.rag_agent import RAGAgent
from src.core.document_processor import DocumentProcessor
from config.config import DOCUMENTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Support RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_store = None
rag_agent = None
document_processor = None


class QueryRequest(BaseModel):
    """Request model for query."""
    query: str
    k: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for query."""
    query: str
    answer: str
    confidence_score: float
    retrieved_chunks: List[Dict[str, Any]]
    validation: Dict[str, Any]


class ProcessDocumentRequest(BaseModel):
    """Request model for document processing."""
    file_path: Optional[str] = None


def initialize_components():
    """Initialize global components."""
    global vector_store, rag_agent, document_processor
    
    try:
        vector_store = FAISSVectorStore()
        # Try to load existing vector store
        if not vector_store.load():
            logger.info("No existing vector store found, starting fresh")
        
        from src.validation.validator import ValidationModule
        validation_module = ValidationModule()
        rag_agent = RAGAgent(vector_store, validation_module)
        document_processor = DocumentProcessor(vector_store)
        
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_components()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Support RAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "process_document": "/api/process_document",
            "stats": "/api/stats",
            "health": "/api/health"
        }
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        if not rag_agent:
            raise HTTPException(status_code=500, detail="RAG agent not initialized")
        
        result = rag_agent.query(request.query, k=request.k or 5)
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            confidence_score=result["confidence_score"],
            retrieved_chunks=result["retrieved_chunks"],
            validation=result["validation"]
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process_document")
async def process_document(request: ProcessDocumentRequest):
    """Process a document and add it to the vector store."""
    try:
        if not document_processor:
            raise HTTPException(status_code=500, detail="Document processor not initialized")
        
        file_path = request.file_path
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")
        
        success = document_processor.process_document(file_path)
        
        if success:
            # Save vector store after processing
            document_processor.save_vector_store()
            return {"status": "success", "message": f"Document {file_path} processed successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to process document: {file_path}")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get statistics about the vector store."""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        stats = vector_store.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "rag_agent_initialized": rag_agent is not None
    }
