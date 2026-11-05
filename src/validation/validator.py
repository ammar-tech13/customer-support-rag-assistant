"""Validation module for context and results."""
import logging
from typing import List, Dict, Any, Optional
#from langchain_community.evaluation.embedding_distance import EmbeddingDistanceEvaluator
import numpy as np

from config.config import MIN_CONFIDENCE_SCORE, MIN_SIMILARITY_SCORE

logger = logging.getLogger(__name__)


class ValidationModule:
    """Validates context and results for RAG system."""
    
    def __init__(self, min_confidence: float = MIN_CONFIDENCE_SCORE, min_similarity: float = MIN_SIMILARITY_SCORE):
        """
        Initialize validation module.
        
        Args:
            min_confidence: Minimum confidence score threshold
            min_similarity: Minimum similarity score threshold
        """
        self.min_confidence = min_confidence
        self.min_similarity = min_similarity
    
    def validate_context(self, retrieved_chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Validate the retrieved context.
        
        Args:
            retrieved_chunks: List of retrieved chunks with similarity scores
            query: Original query
            
        Returns:
            Validation results with confidence score and validation status
        """
        if not retrieved_chunks:
            return {
                'is_valid': False,
                'confidence_score': 0.0,
                'reason': 'No chunks retrieved',
                'chunk_count': 0
            }
        
        # Calculate average similarity score
        similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in retrieved_chunks]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        
        # Calculate confidence based on similarity scores
        # Higher similarity = higher confidence
        confidence_score = (avg_similarity + max_similarity) / 2
        
        # Check if we have enough relevant chunks
        relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.get('similarity_score', 0.0) >= self.min_similarity]
        
        is_valid = (
            confidence_score >= self.min_confidence and
            len(relevant_chunks) > 0 and
            max_similarity >= self.min_similarity
        )
        
        return {
            'is_valid': is_valid,
            'confidence_score': float(confidence_score),
            'max_similarity': float(max_similarity),
            'avg_similarity': float(avg_similarity),
            'chunk_count': len(retrieved_chunks),
            'relevant_chunk_count': len(relevant_chunks),
            'reason': 'Context validated' if is_valid else 'Low similarity scores or insufficient relevant chunks'
        }
    
    def validate_answer(self, answer: str, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the generated answer.
        
        Args:
            answer: Generated answer
            query: Original query
            context: Retrieved context chunks
            
        Returns:
            Validation results for the answer
        """
        if not answer or not answer.strip():
            return {
                'is_valid': False,
                'confidence_score': 0.0,
                'reason': 'Empty answer',
                'answer_length': 0
            }
        
        # Basic validation checks
        answer_length = len(answer.strip())
        has_content = answer_length > 10  # Minimum answer length
        
        # Check if answer seems relevant (basic heuristic)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
        
        # Calculate confidence based on multiple factors
        length_score = min(1.0, answer_length / 100)  # Normalize to 0-1
        overlap_score = word_overlap
        
        # If we have context, check if answer references it
        context_score = 0.5  # Default
        if context:
            context_text = ' '.join([chunk.get('content', '')[:100] for chunk in context[:3]])
            context_words = set(context_text.lower().split())
            answer_context_overlap = len(answer_words.intersection(context_words)) / len(answer_words) if answer_words else 0
            context_score = min(1.0, answer_context_overlap * 2)
        
        confidence_score = (length_score * 0.3 + overlap_score * 0.3 + context_score * 0.4)
        
        is_valid = (
            has_content and
            confidence_score >= self.min_confidence and
            word_overlap > 0.1  # At least some word overlap
        )
        
        return {
            'is_valid': is_valid,
            'confidence_score': float(confidence_score),
            'answer_length': answer_length,
            'word_overlap': float(word_overlap),
            'context_score': float(context_score),
            'reason': 'Answer validated' if is_valid else 'Answer does not meet quality thresholds'
        }
    
    def validate_complete(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """
        Complete validation of query, context, and answer.
        
        Args:
            query: Original query
            retrieved_chunks: Retrieved context chunks
            answer: Generated answer
            
        Returns:
            Complete validation results
        """
        context_validation = self.validate_context(retrieved_chunks, query)
        answer_validation = self.validate_answer(answer, query, retrieved_chunks)
        
        # Overall confidence is weighted average
        overall_confidence = (
            context_validation['confidence_score'] * 0.5 +
            answer_validation['confidence_score'] * 0.5
        )
        
        is_valid = (
            context_validation['is_valid'] and
            answer_validation['is_valid'] and
            overall_confidence >= self.min_confidence
        )
        
        return {
            'is_valid': is_valid,
            'overall_confidence': float(overall_confidence),
            'context_validation': context_validation,
            'answer_validation': answer_validation,
            'retrieved_chunks': retrieved_chunks,
            'chunk_count': len(retrieved_chunks),
            'query': query,
            'answer': answer
        }
