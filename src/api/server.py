
"""FastAPI server for the Customer Support RAG system."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.vector_store.faiss_store import FAISSVectorStore
from src.agents.rag_agent import RAGAgent
from src.core.document_processor import DocumentProcessor
from config.config import DOCUMENTS_DIR


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Support RAG API",
    description="Document-based AI customer support using RAG",
    version="1.0.0"
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Global components
# --------------------------------------------------

vector_store = None
rag_agent = None
document_processor = None


# --------------------------------------------------
# Request and response models
# --------------------------------------------------

class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str
    k: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str
    answer: str
    confidence_score: float
    retrieved_chunks: List[Dict[str, Any]]
    validation: Dict[str, Any]


# --------------------------------------------------
# Initialize RAG components
# --------------------------------------------------

def initialize_components():
    """Initialize vector store, RAG agent, and document processor."""

    global vector_store, rag_agent, document_processor

    try:
        logger.info("Initializing RAG components...")

        vector_store = FAISSVectorStore()

        # Load existing vector store if available
        if not vector_store.load():
            logger.info("No existing vector store found. Starting fresh.")

        from src.validation.validator import ValidationModule

        validation_module = ValidationModule()

        rag_agent = RAGAgent(
            vector_store,
            validation_module
        )

        document_processor = DocumentProcessor(
            vector_store
        )

        logger.info("RAG components initialized successfully.")

    except Exception:
        logger.exception("Failed to initialize RAG components.")
        raise


# --------------------------------------------------
# Startup event
# --------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize components when the API starts."""

    initialize_components()


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    """API information."""

    return {
        "message": "Customer Support RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/api/query",
            "process_document": "/api/process_document",
            "stats": "/api/stats",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/api/health")
async def health():
    """Check API and component initialization status."""

    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "rag_agent_initialized": rag_agent is not None,
        "document_processor_initialized": document_processor is not None
    }


# --------------------------------------------------
# Query endpoint
# --------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a question using the RAG agent."""

    try:
        if rag_agent is None:
            raise HTTPException(
                status_code=503,
                detail="RAG agent is not initialized"
            )

        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        k = max(1, min(request.k or 5, 20))

        result = rag_agent.query(
            request.query,
            k=k
        )

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            confidence_score=result["confidence_score"],
            retrieved_chunks=result["retrieved_chunks"],
            validation=result["validation"]
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error processing query")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# Document upload and processing endpoint
# --------------------------------------------------

@app.post("/api/process_document")
async def process_document(file: UploadFile = File(...)):
    """
    Upload and process PDF or DOCX documents.

    The uploaded document is saved in DOCUMENTS_DIR,
    processed, and added to the FAISS vector store.
    """

    file_path = None

    try:
        if document_processor is None:
            raise HTTPException(
                status_code=503,
                detail="Document processor is not initialized"
            )

        # Validate filename
        filename = Path(file.filename or "").name

        if not filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename"
            )

        # Validate file extension
        if not filename.lower().endswith((".pdf", ".docx")):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and DOCX files are supported"
            )

        # Read uploaded file
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Save document
        file_path = Path(DOCUMENTS_DIR) / filename
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if file_path.exists():
            raise HTTPException(
                status_code=409,
                detail="A file with this name already exists"
            )

        file_path.write_bytes(content)

        logger.info("Processing document: %s", filename)

        # Process document
        success = document_processor.process_document(
            str(file_path)
        )

        if not success:
            file_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=500,
                detail="Document processing failed"
            )

        # Save FAISS vector store
        document_processor.save_vector_store()

        logger.info("Document processed successfully: %s", filename)

        return {
            "status": "success",
            "message": f"{filename} processed successfully"
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Document processing error")

        # Remove failed upload
        if file_path and file_path.exists():
            file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:
        await file.close()


# --------------------------------------------------
# Vector store statistics endpoint
# --------------------------------------------------

@app.get("/api/stats")
async def get_stats():
    """Return vector store statistics."""

    try:
        if vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Vector store is not initialized"
            )

        return vector_store.get_stats()

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error retrieving statistics")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
