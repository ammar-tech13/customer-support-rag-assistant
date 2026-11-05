# Customer Support RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for customer support automation using vector search and LLMs.

## Features

- 📄 **Document Loading**: Supports PDF and Word documents with text, table, and image extraction
- 🔄 **Auto-Processing**: Automatically processes new documents when added to the folder
- 🔍 **Vector Search**: FAISS-based vector store with JinaAI embeddings
- 🤖 **Agentic RAG**: LangGraph-based agentic RAG with multi-step reasoning
- ✅ **Validation**: Built-in validation for context and answer quality
- 📊 **Confidence Scores**: Shows confidence scores and retrieved chunks
- 🌐 **Client-Server**: FastAPI backend with Streamlit frontend
- 🔐 **Secure**: Environment-based API key management

## Technology Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: JinaAI jina-embeddings-v3
- **Vector Store**: FAISS
- **Framework**: LangChain + LangGraph
- **API**: FastAPI
- **UI**: Streamlit

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your API keys:
# GOOGLE_API_KEY=your_google_api_key_here
# JINAAI_API_KEY=your_jinaai_api_key_here
```

## Usage

### 1. Process Documents

Start the document processor and file watcher:

```bash
python main.py
```

This will:
- Process all existing documents in `./data/documents/`
- Watch for new documents and automatically process them
- Save the vector store to `./data/vector_store/`

### 2. Start the API Server

In a new terminal:

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### 3. Start the Streamlit UI

In another terminal:

```bash
python run_ui.py
```

Or directly:

```bash
streamlit run src/ui/streamlit_app.py
```

The UI will be available at `http://localhost:8501`

## Project Structure

```
CustSupport_RAG/
├── config/                 # Configuration files
├── src/
│   ├── agents/            # RAG agent implementation
│   ├── api/               # FastAPI server
│   ├── chunking/          # Document chunking strategies
│   ├── core/              # Core utilities
│   ├── loaders/           # Document loaders
│   ├── ui/                # Streamlit UI
│   ├── validation/        # Validation modules
│   └── vector_store/      # Vector store implementation
├── data/
│   ├── documents/         # Place your documents here
│   └── vector_store/      # Vector store files (auto-generated)
├── main.py                # Main entry point
├── run_api.py             # API server runner
├── run_ui.py              # UI runner
└── requirements.txt       # Dependencies
```

## API Endpoints

### Query the RAG System

```bash
POST /api/query
Content-Type: application/json

{
  "query": "What is the refund policy?",
  "k": 5
}
```

Response:
```json
{
  "query": "What is the refund policy?",
  "answer": "...",
  "confidence_score": 0.85,
  "retrieved_chunks": [...],
  "validation": {...}
}
```

### Process a Document

```bash
POST /api/process_document
Content-Type: application/json

{
  "file_path": "./data/documents/manual.pdf"
}
```

### Get Statistics

```bash
GET /api/stats
```

### Health Check

```bash
GET /api/health
```

## Configuration

Edit `config/config.py` or set environment variables:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDING_MODEL`: Embedding model (default: jinaai/jina-embeddings-v3)
- `LLM_MODEL`: LLM model (default: gemini-2.0-flash-exp)
- `MIN_CONFIDENCE_SCORE`: Minimum confidence threshold (default: 0.7)
- `MIN_SIMILARITY_SCORE`: Minimum similarity threshold (default: 0.6)

## How It Works

1. **Document Ingestion**:
   - Documents are loaded from `./data/documents/`
   - Text, tables, and images are extracted
   - Documents are chunked with metadata preservation
   - Chunks are embedded and stored in FAISS

2. **Query Processing**:
   - User query is embedded
   - Similar chunks are retrieved from vector store
   - LLM generates answer from retrieved context
   - Validation module checks quality
   - Response includes answer, chunks, and confidence score

3. **Auto-Processing**:
   - File watcher monitors the documents directory
   - New files are automatically processed and indexed
   - Vector store is updated and saved

## Validation

The system includes comprehensive validation:

- **Context Validation**: Checks if retrieved chunks are relevant
- **Answer Validation**: Validates answer quality and relevance
- **Confidence Scoring**: Calculates overall confidence score
- **Threshold Checking**: Ensures minimum quality standards

## Troubleshooting

### API Key Issues

Make sure your `.env` file contains valid API keys:
- `GOOGLE_API_KEY`: For Gemini LLM

### Vector Store Issues

If the vector store fails to load:
- Delete `./data/vector_store/` and reprocess documents
- Check file permissions

### Document Processing Issues

- Ensure documents are in supported formats (PDF, DOCX, DOC)
- Check that required dependencies are installed (pytesseract for OCR)
- Verify file paths are correct

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
