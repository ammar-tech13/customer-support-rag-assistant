# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini
- (Optional) JinaAI API key for embeddings

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env

# Edit .env and add your keys:
GOOGLE_API_KEY=your_google_api_key_here

```

### 3. Add Documents

Place your PDF or Word documents in the `./data/documents/` directory.

### 4. Process Documents (Option 1: Manual)

```bash
python main.py
```

This will:
- Process all documents in `./data/documents/`
- Create chunks and embeddings
- Save to vector store
- Start watching for new files

### 5. Start the API Server

In a new terminal:

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### 6. Start the Streamlit UI

In another terminal:

```bash
python run_ui.py
```

Or:

```bash
streamlit run src/ui/streamlit_app.py
```

The UI will be available at `http://localhost:8501`

## Usage

### Via Streamlit UI

1. Open `http://localhost:8501` in your browser
2. Enter your question in the query box
3. Click "Search"
4. View the answer, confidence score, and retrieved chunks

### Via API

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "k": 5}'
```

### Process New Documents

**Via UI:**
- Go to the "Process Documents" tab
- Upload a file or enter a file path
- Click "Process Document"

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/process_document" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/documents/new_document.pdf"}'
```

**Automatic:**
- Simply add files to `./data/documents/`
- The file watcher will automatically process them (if `main.py` is running)

## Troubleshooting

### API Key Issues

- Ensure `.env` file exists and contains valid API keys
- Check that keys are not wrapped in quotes
- Verify API keys are active and have proper permissions

### Import Errors

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Vector Store Issues

- If vector store fails to load, delete `./data/vector_store/` and reprocess documents
- Check file permissions on the vector store directory

### Document Processing Errors

- Ensure documents are in supported formats (PDF, DOCX, DOC)
- Check that documents are not corrupted
- Verify file paths are correct

### Port Already in Use

- Change ports in `.env` file:
  - `API_PORT=8001` (instead of 8000)
  - `STREAMLIT_PORT=8502` (instead of 8501)

## Next Steps

- Read `ARCHITECTURE.md` for detailed system architecture
- Read `README.md` for comprehensive documentation
- Customize configuration in `config/config.py`
- Add more document formats (extend `document_loader.py`)
