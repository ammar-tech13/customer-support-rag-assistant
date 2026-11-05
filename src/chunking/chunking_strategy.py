"""Chunking strategy using LangChain with separate handling for text, tables, and images."""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import pandas as pd

from config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class AdvancedChunkingStrategy:
    """Advanced chunking using LangChain with separate handling for different content types."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize chunking strategy.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use LangChain's RecursiveCharacterTextSplitter for text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents while preserving metadata and handling different content types.
        
        Args:
            documents: List of LangChain Document objects (text, tables, images)
            
        Returns:
            List of chunked Document objects with enhanced metadata
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            content_type = doc.metadata.get('content_type', 'text')
            
            if content_type == 'table':
                # Handle tables separately - keep as single chunk or split if too large
                chunks = self._chunk_table(doc)
            elif content_type == 'image':
                # Handle images separately - keep OCR text as single chunk
                chunks = self._chunk_image(doc)
            else:
                # Handle text with recursive splitting
                chunks = self._chunk_text(doc)
            
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _chunk_text(self, doc: Document) -> List[Document]:
        """Chunk text content using LangChain splitter."""
        # Split the document
        chunks = self.text_splitter.split_documents([doc])
        
        # Enhance metadata for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.metadata:
                chunk.metadata = {}
            
            # Preserve original metadata
            original_metadata = doc.metadata.copy() if doc.metadata else {}
            chunk.metadata.update(original_metadata)
            
            # Add chunk-specific metadata
            chunk.metadata['chunk_index'] = chunk_idx
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_id'] = f"{original_metadata.get('file_name', 'doc')}_text_chunk{chunk_idx}"
            chunk.metadata['char_count'] = len(chunk.page_content)
            chunk.metadata['word_count'] = len(chunk.page_content.split())
            chunk.metadata['content_type'] = 'text'
        
        return chunks
    
    def _chunk_table(self, doc: Document) -> List[Document]:
        """Chunk table content - keep as single chunk or split if too large."""
        table_text = doc.page_content
        
        # If table is small enough, keep as single chunk
        if len(table_text) <= self.chunk_size:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['chunk_index'] = 0
            doc.metadata['total_chunks'] = 1
            doc.metadata['chunk_id'] = f"{doc.metadata.get('file_name', 'doc')}_table{doc.metadata.get('table_index', 0)}"
            doc.metadata['content_type'] = 'table'
            return [doc]
        
        # If table is too large, try to split by rows
        try:
            # Try to parse as DataFrame if it's a table
            lines = table_text.split('\n')
            if len(lines) > 1:
                # Split into smaller chunks by rows
                rows_per_chunk = max(1, len(lines) // (len(table_text) // self.chunk_size + 1))
                chunks = []
                
                for i in range(0, len(lines), rows_per_chunk):
                    chunk_lines = lines[i:i+rows_per_chunk]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                    chunk_metadata['chunk_index'] = i // rows_per_chunk
                    chunk_metadata['total_chunks'] = (len(lines) + rows_per_chunk - 1) // rows_per_chunk
                    chunk_metadata['chunk_id'] = f"{chunk_metadata.get('file_name', 'doc')}_table{chunk_metadata.get('table_index', 0)}_part{i//rows_per_chunk}"
                    chunk_metadata['content_type'] = 'table'
                    
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    ))
                
                return chunks
        except Exception as e:
            logger.warning(f"Error splitting large table: {e}, keeping as single chunk")
        
        # Fallback: keep as single chunk even if large
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata['chunk_index'] = 0
        doc.metadata['total_chunks'] = 1
        doc.metadata['chunk_id'] = f"{doc.metadata.get('file_name', 'doc')}_table{doc.metadata.get('table_index', 0)}"
        doc.metadata['content_type'] = 'table'
        return [doc]
    
    def _chunk_image(self, doc: Document) -> List[Document]:
        """Chunk image OCR text - typically keep as single chunk."""
        ocr_text = doc.page_content.strip()
        
        if not ocr_text:
            return []
        
        # If OCR text is small, keep as single chunk
        if len(ocr_text) <= self.chunk_size:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['chunk_index'] = 0
            doc.metadata['total_chunks'] = 1
            doc.metadata['chunk_id'] = f"{doc.metadata.get('file_name', 'doc')}_image"
            doc.metadata['content_type'] = 'image'
            return [doc]
        
        # If OCR text is large, split it like regular text
        chunks = self.text_splitter.split_documents([doc])
        
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.metadata:
                chunk.metadata = {}
            
            original_metadata = doc.metadata.copy() if doc.metadata else {}
            chunk.metadata.update(original_metadata)
            chunk.metadata['chunk_index'] = chunk_idx
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_id'] = f"{original_metadata.get('file_name', 'doc')}_image_chunk{chunk_idx}"
            chunk.metadata['content_type'] = 'image'
        
        return chunks
