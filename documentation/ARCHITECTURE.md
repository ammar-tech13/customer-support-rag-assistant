# Customer Support RAG System - Architecture Documentation

## Overview

This is an Agentic Retrieval-Augmented Generation (RAG) system designed for customer support automation. It uses vector search and Large Language Models (LLMs) to deliver accurate, context-aware responses from internal knowledge bases.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  ┌──────────────────┐              ┌──────────────────┐          │
│  │  Streamlit UI    │              │   API Clients   │          │
│  └────────┬─────────┘              └────────┬─────────┘          │
└───────────┼──────────────────────────────────┼──────────────────┘
            │                                  │
            │ HTTP Requests                    │
            │                                  │
┌───────────▼──────────────────────────────────▼──────────────────┐
│                      API Server (FastAPI)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Endpoints:                                               │  │
│  │  - /api/query          - Query the RAG system            │  │
│  │  - /api/process_document - Process new documents         │  │
│  │  - /api/stats          - Get vector store statistics     │  │
│  │  - /api/health         - Health check                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────┬────────────────────────────────────────────────────┘
            │
┌───────────▼────────────────────────────────────────────────────┐
│                  Agentic RAG Layer (LangGraph)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  State Graph:                                            │  │
│  │  1. Retrieve Node - Search vector store                  │  │
│  │  2. Generate Node - Generate answer with LLM             │  │
│  │  3. Validate Node - Validate context and answer           │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────┬────────────────────────────────────────────────────┘
            │
┌───────────▼────────────────────────────────────────────────────┐
│                    Processing Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Document   │  │   Chunking   │  │   Vector     │        │
│  │   Loader     │→ │   Strategy   │→ │   Store      │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              File Watcher (Auto-trigger)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Loader (`src/loaders/document_loader.py`)

**Purpose**: Loads and extracts content from PDF and Word documents.

**Features**:
- Supports PDF (.pdf) and Word (.docx, .doc) formats
- Extracts text content
- Extracts tables (using pandas)
- Extracts images with OCR (using pytesseract)
- Preserves page numbers and document metadata

**Key Methods**:
- `load_document(file_path)`: Load a single document
- `load_directory(directory)`: Load all documents from a directory

### 2. Chunking Strategy (`src/chunking/chunking_strategy.py`)

**Purpose**: Intelligently chunks documents while preserving metadata.

**Features**:
- Uses RecursiveCharacterTextSplitter from LangChain
- Preserves page numbers, file names, and chunk types
- Handles text, tables, and images separately
- Configurable chunk size and overlap
- Creates unique chunk IDs for tracking

**Key Methods**:
- `chunk_document(document)`: Chunk a document into smaller pieces
- `_chunk_text()`: Chunk text content
- `_create_table_chunk()`: Create chunks from tables
- `_create_image_chunk()`: Create chunks from image OCR text

### 3. Vector Store (`src/vector_store/faiss_store.py`)

**Purpose**: Manages vector embeddings and similarity search using FAISS.

**Features**:
- Uses FAISS for efficient similarity search
- Supports BAAI/bge-base-en-v1.5
- Stores metadata alongside embeddings
- Persistent storage to disk
- Configurable similarity thresholds

**Key Methods**:
- `add_chunks(chunks)`: Add chunks to the vector store
- `search(query, k)`: Search for similar chunks
- `save(name)`: Save vector store to disk
- `load(name)`: Load vector store from disk
- `get_stats()`: Get statistics about the store

### 4. File Watcher (`src/core/file_watcher.py`)

**Purpose**: Automatically detects and processes new documents.

**Features**:
- Uses watchdog library for file system monitoring
- Triggers processing when new files are added
- Handles file creation and modification events
- Prevents duplicate processing
- Can process existing files on startup

**Key Methods**:
- `start(callback)`: Start watching for new files
- `stop()`: Stop watching
- `process_existing_files(callback)`: Process files already in directory

### 5. Validation Module (`src/validation/validator.py`)

**Purpose**: Validates context and generated answers.

**Features**:
- Validates retrieved context quality
- Validates generated answer quality
- Calculates confidence scores
- Checks similarity thresholds
- Provides detailed validation reports

**Key Methods**:
- `validate_context(retrieved_chunks, query)`: Validate retrieved context
- `validate_answer(answer, query, context)`: Validate generated answer
- `validate_complete(query, chunks, answer)`: Complete validation

### 6. RAG Agent (`src/agents/rag_agent.py`)

**Purpose**: Agentic RAG implementation using LangGraph.

**Features**:
- Uses LangGraph for state management
- Implements multi-step reasoning pipeline
- Integrates with Gemini 2.5 Flash LLM
- Retrieves context, generates answers, and validates results
- Returns confidence scores and retrieved chunks

**State Graph Flow**:
1. **Retrieve Node**: Searches vector store for relevant chunks
2. **Generate Node**: Uses LLM to generate answer from context
3. **Validate Node**: Validates both context and answer

**Key Methods**:
- `query(question, k)`: Process a query through the agent

### 7. Document Processor (`src/core/document_processor.py`)

**Purpose**: Orchestrates the document processing pipeline.

**Features**:
- Coordinates loading, chunking, and indexing
- Manages vector store operations
- Handles batch processing
- Provides high-level interface

**Key Methods**:
- `process_document(file_path)`: Process a single document
- `process_directory(directory)`: Process all documents in a directory
- `save_vector_store(name)`: Save the vector store
- `load_vector_store(name)`: Load the vector store

### 8. API Server (`src/api/server.py`)

**Purpose**: RESTful API for the RAG system.

**Endpoints**:
- `POST /api/query`: Query the RAG system
- `POST /api/process_document`: Process a new document
- `GET /api/stats`: Get vector store statistics
- `GET /api/health`: Health check

**Features**:
- FastAPI-based REST API
- CORS support
- Error handling
- Request/response validation

### 9. Streamlit UI (`src/ui/streamlit_app.py`)

**Purpose**: User-friendly web interface.

**Features**:
- Query interface with confidence scores
- Display retrieved chunks
- Show validation details
- Document upload
- System statistics dashboard
- Document processing interface

## Data Flow

### Document Ingestion Flow

```
1. Document added to ./data/documents/
   ↓
2. File Watcher detects new file
   ↓
3. Document Loader extracts content (text, tables, images)
   ↓
4. Chunking Strategy creates chunks with metadata
   ↓
5. Vector Store embeds chunks and stores them
   ↓
6. Vector Store saved to disk
```

### Query Flow

```
1. User submits query via UI or API
   ↓
2. RAG Agent receives query
   ↓
3. Retrieve Node: Vector Store searches for similar chunks
   ↓
4. Generate Node: LLM generates answer from retrieved context
   ↓
5. Validate Node: Validation module checks quality
   ↓
6. Response returned with answer, chunks, and confidence score
```

## Configuration

Configuration is managed through:
- `.env` file for API keys and settings
- `config/config.py` for application configuration

Key configuration options:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDING_MODEL`: Embedding model name (default: jinaai/jina-embeddings-v3)
- `LLM_MODEL`: LLM model name (default: gemini-2.0-flash-exp)
- `MIN_CONFIDENCE_SCORE`: Minimum confidence threshold (default: 0.7)
- `MIN_SIMILARITY_SCORE`: Minimum similarity threshold (default: 0.6)

## Technology Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: JinaAI jina-embeddings-v3
- **Vector Store**: FAISS
- **Framework**: LangChain + LangGraph
- **API**: FastAPI
- **UI**: Streamlit
- **File Watching**: Watchdog
- **Document Processing**: PyPDF, python-docx, pytesseract

## Project Structure

```
CustSupport_RAG/
├── config/
│   └── config.py              # Configuration
├── src/
│   ├── agents/
│   │   └── rag_agent.py       # Agentic RAG agent
│   ├── api/
│   │   └── server.py            # FastAPI server
│   ├── chunking/
│   │   └── chunking_strategy.py  # Chunking logic
│   ├── core/
│   │   ├── document_processor.py  # Processing pipeline
│   │   └── file_watcher.py       # File watcher
│   ├── loaders/
│   │   └── document_loader.py    # Document loading
│   ├── ui/
│   │   └── streamlit_app.py      # Streamlit UI
│   ├── validation/
│   │   └── validator.py          # Validation module
│   └── vector_store/
│       └── faiss_store.py        # FAISS vector store
├── data/
│   ├── documents/                # Input documents
│   └── vector_store/             # Vector store files
├── main.py                        # Main entry point
├── run_api.py                     # API server runner
├── run_ui.py                      # UI runner
├── requirements.txt               # Dependencies
└── ARCHITECTURE.md                # This file
```

## Usage

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Process Documents

```bash
# Start document processor and file watcher
python main.py
```

### 3. Start API Server

```bash
python run_api.py
```

### 4. Start UI

```bash
python run_ui.py
```

## Agentic RAG Features

The system implements an agentic RAG approach using LangGraph:

1. **Multi-step Reasoning**: The agent processes queries through multiple steps (retrieve → generate → validate)
2. **State Management**: LangGraph manages state across processing steps
3. **Validation**: Built-in validation ensures quality of both context and answers
4. **Confidence Scoring**: Provides confidence scores for transparency
5. **Chunk Visibility**: Shows which chunks were used to generate answers

## Future Enhancements

- Support for more document formats
- Advanced table extraction (tabula-py, camelot)
- Image understanding beyond OCR
- Multi-modal RAG (text + images)
- Query rewriting and expansion
- Conversation memory
- Feedback loop for improving answers
