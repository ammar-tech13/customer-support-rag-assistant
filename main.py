"""Main entry point for the RAG system."""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.document_processor import DocumentProcessor
from src.core.file_watcher import DocumentWatcher
from src.vector_store.faiss_store import FAISSVectorStore
from config.config import DOCUMENTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_document_callback(file_path: str):
    """Callback function for file watcher."""
    logger.info(f"Processing new document: {file_path}")
    
    try:
        processor = DocumentProcessor()
        processor.load_vector_store()  # Load existing store
        
        success = processor.process_document(file_path)
        
        if success:
            processor.save_vector_store()  # Save updated store
            logger.info(f"Successfully processed and indexed: {file_path}")
        else:
            logger.error(f"Failed to process: {file_path}")
    except Exception as e:
        logger.error(f"Error in document processing callback: {e}")


def main():
    """Main function to start document processing and file watching."""
    logger.info("Starting Customer Support RAG System")
    
    # Initialize vector store
    vector_store = FAISSVectorStore()
    if vector_store.load():
        logger.info("Loaded existing vector store")
    else:
        logger.info("Starting with empty vector store")
    
    # Process existing documents
    processor = DocumentProcessor(vector_store)
    logger.info(f"Processing existing documents in {DOCUMENTS_DIR}")
    processed = processor.process_directory(str(DOCUMENTS_DIR))
    logger.info(f"Processed {processed} existing documents")
    
    # Save vector store
    processor.save_vector_store()
    
    # Start file watcher
    watcher = DocumentWatcher(
        documents_dir=str(DOCUMENTS_DIR),
        callback=process_document_callback
    )
    
    logger.info("Starting file watcher...")
    watcher.process_existing_files(process_document_callback)
    watcher.start(process_document_callback)
    
    try:
        logger.info("File watcher is running. Press Ctrl+C to stop.")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        watcher.stop()
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
