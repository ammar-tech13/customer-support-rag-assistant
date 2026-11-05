"""Document processing pipeline using LangChain components."""
import logging
from pathlib import Path
from typing import Optional, List
from langchain_core.documents import Document

from src.loaders.document_loader import DocumentLoader
from src.chunking.chunking_strategy import AdvancedChunkingStrategy
from src.vector_store.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates document loading, chunking, and vector store indexing using LangChain."""
    
    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        """
        Initialize document processor.
        
        Args:
            vector_store: Optional vector store instance (creates new one if not provided)
        """
        self.loader = DocumentLoader()
        self.chunker = AdvancedChunkingStrategy()
        self.vector_store = vector_store or FAISSVectorStore()
    
    def process_document(self, file_path: str) -> bool:
        """
        Process a single document: load, chunk, and index using LangChain.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Load document using LangChain loader
            documents: List[Document] = self.loader.load_document(file_path)
            
            if not documents:
                logger.warning(f"No content loaded from document: {file_path}")
                return False
            
            # Chunk documents using LangChain splitter
            chunks: List[Document] = self.chunker.chunk_documents(documents)
            
            if not chunks:
                logger.warning(f"No chunks created from document: {file_path}")
                return False
            
            # Add chunks to vector store using LangChain
            self.vector_store.add_documents(chunks)
            
            logger.info(f"Successfully processed document: {file_path} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False
    
    def process_directory(self, directory: str) -> int:
        """
        Process all documents in a directory using LangChain loaders.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            Number of successfully processed documents
        """
        try:
            # Load all documents using LangChain DirectoryLoader
            all_documents: List[Document] = self.loader.load_directory(directory)
            
            if not all_documents:
                logger.warning(f"No documents found in {directory}")
                return 0
            
            # Chunk all documents
            all_chunks: List[Document] = self.chunker.chunk_documents(all_documents)
            
            if not all_chunks:
                logger.warning(f"No chunks created from documents in {directory}")
                return 0
            
            # Add all chunks to vector store
            self.vector_store.add_documents(all_chunks)
            
            # Count unique files processed
            unique_files = set()
            for chunk in all_chunks:
                file_path = chunk.metadata.get('file_path', '')
                if file_path:
                    unique_files.add(file_path)
            
            processed_count = len(unique_files)
            logger.info(f"Processed {processed_count} files, created {len(all_chunks)} chunks")
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
            return 0
    
    def save_vector_store(self, name: str = "vector_store") -> None:
        """Save the vector store to disk."""
        self.vector_store.save(name)
    
    def load_vector_store(self, name: str = "vector_store") -> bool:
        """Load the vector store from disk."""
        return self.vector_store.load(name)
