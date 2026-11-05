"""FAISS vector store using LangChain with HuggingFace embeddings support."""
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.config import VECTOR_STORE_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS vector store using LangChain with HuggingFace embeddings."""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL, store_path: Optional[str] = None):
        """
        Initialize FAISS vector store using LangChain.
        
        Args:
            embedding_model: Name of the embedding model (HuggingFace model name)
            store_path: Path to save/load the vector store
        """
        self.embedding_model = embedding_model
        self.store_path = Path(store_path) if store_path else VECTOR_STORE_DIR
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings using HuggingFaceEmbeddings
        # BAAI/bge-base-en-v1.5 and other HuggingFace models work directly
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use CPU by default, can be changed to 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
        )
        
        # Get embedding dimension (BAAI/bge-base-en-v1.5 has 768 dimensions)
        # For other models, this will be determined automatically
        self.embedding_dim = 768  # Default for BAAI/bge-base-en-v1.5
        
        self.vector_store: Optional[FAISS] = None
        self.metadata_store = {}  # Additional metadata tracking
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store using LangChain.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            return
        
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = FAISS.from_documents(
                documents,
                embedding=self.embeddings
            )
        else:
            # Add to existing store
            self.vector_store.add_documents(documents)
        
        # Track metadata
        for doc in documents:
            chunk_id = doc.metadata.get('chunk_id', '')
            if chunk_id:
                self.metadata_store[chunk_id] = doc.metadata
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def get_retriever(self, k: int = 5, search_kwargs: Optional[Dict[str, Any]] = None) -> BaseRetriever:
        """
        Get a LangChain retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve
            search_kwargs: Additional search parameters
            
        Returns:
            LangChain BaseRetriever
        """
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Add documents first.")
        
        search_kwargs = search_kwargs or {}
        search_kwargs['k'] = k
        
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search the vector store with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            if filter:
                # Filter by metadata
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the vector store and return formatted results.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with content, metadata, and similarity scores
        """
        results = self.similarity_search_with_score(query, k=k, filter=filter_dict)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            # Convert distance to similarity (FAISS returns L2 distance)
            # For normalized embeddings, cosine similarity = 1 - distance/2
            similarity = 1 - (score / 2) if score <= 2 else 1 / (1 + score)
            
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(similarity),
                'distance': float(score)
            })
        
        return formatted_results
    
    def save(self, name: str = "vector_store") -> None:
        """Save the vector store to disk using LangChain's save method."""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        try:
            # Save using LangChain's method
            self.vector_store.save_local(str(self.store_path), index_name=name)
            
            # Save additional metadata
            metadata_file = self.store_path / f"{name}_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.info(f"Vector store saved to {self.store_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load(self, name: str = "vector_store") -> bool:
        """
        Load the vector store from disk using LangChain's load method.
        
        Args:
            name: Name of the stored vector store
            
        Returns:
            True if loaded successfully, False otherwise
        """
        store_file = self.store_path / f"{name}.faiss"
        
        if not store_file.exists():
            logger.warning(f"Vector store file not found: {store_file}")
            return False
        
        try:
            # Load using LangChain's method
            self.vector_store = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
                index_name=name
            )
            
            # Load additional metadata
            metadata_file = self.store_path / f"{name}_metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
            
            logger.info(f"Vector store loaded from {self.store_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.vector_store is None:
            return {
                'total_chunks': 0,
                'embedding_dim': self.embedding_dim,
                'embedding_model': self.embedding_model
            }
        
        try:
            # Get index size
            index = self.vector_store.index
            total_chunks = index.ntotal if hasattr(index, 'ntotal') else 0
            
            return {
                'total_chunks': total_chunks,
                'embedding_dim': self.embedding_dim,
                'embedding_model': self.embedding_model,
                'metadata_count': len(self.metadata_store)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_chunks': 0,
                'embedding_dim': self.embedding_dim,
                'embedding_model': self.embedding_model
            }
