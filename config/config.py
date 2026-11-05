"""Configuration module for the RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(DATA_DIR / "documents")))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(DATA_DIR / "vector_store")))

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")

# Server Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Validation thresholds
MIN_CONFIDENCE_SCORE = 0.7
MIN_SIMILARITY_SCORE = 0.6
