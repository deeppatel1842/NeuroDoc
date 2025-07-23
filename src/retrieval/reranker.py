"""
Advanced Reranking Module

This module implements sophisticated reranking strategies to improve the quality
of retrieved documents in the RAG pipeline, targeting the 18% relevance boost.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import re
from collections import Counter
import math

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("CrossEncoder not available for reranking")

from ..config import RETRIEVAL_CONFIG
from ..utils.text_utils import TextCleaner

logger = logging.getLogger(__name__)


class AdvancedReranker:
    """
    Multi-strategy reranker that combines multiple signals to improve retrieval quality.
    Implements cross-encoder reranking, semantic similarity, and heuristic scoring.
    """
    
    def __init__(self):
        self.cross_encoder = None
        self.text_cleaner = TextCleaner()
        
        # Reranking configuration (reduced for faster performance)
        self.max_rerank_candidates = RETRIEVAL_CONFIG.get("max_rerank_candidates", 10)  # Reduced from 20 to 10
        self.cross_encoder_weight = 0.4
        self.semantic_weight = 0.3
        self.heuristic_weight = 0.2
        self.diversity_weight = 0.1
        
        # Initialize cross-encoder if available
        self._initialize_cross_encoder()
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder model for reranking."""
        try:
            # Temporarily disabled for faster performance
            self.cross_encoder = None
            logger.info("Cross-encoder disabled for faster performance")
            
            # if CROSS_ENCODER_AVAILABLE:
            #     # Use a lightweight cross-encoder for speed
            #     model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            #     self.cross_encoder = CrossEncoder(model_name)
            #     logger.info(f"Cross-encoder initialized: {model_name}")
            # else:
            #     logger.info("Cross-encoder not available, using alternative reranking")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.cross_encoder = None
    
    async def rerank_results(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        top_k: Optional[int] = None,
        conversation_context: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using multiple strategies.
        
        Args:
            query: Original search query
            retrieved_docs: List of retrieved documents with scores
            top_k: Number of top results to return after reranking
            conversation_context: Recent user questions for context
            
        Returns:
            Reranked list of documents
        """
        start_time = datetime.utcnow()
        
        if not retrieved_docs:
            return []
        
        # Limit candidates for reranking (performance optimization)
        candidates = retrieved_docs[:self.max_rerank_candidates]
        top_k = top_k or len(candidates)
        
        logger.info(f"Reranking {len(candidates)} candidates for query: '{query[:50]}...'")
        
        try:
            # Enhance query with conversation context
            # Skip context enhancement if no conversation context
            enhanced_query = query
            
            # Apply different reranking strategies
            reranking_scores = await self._compute_reranking_scores(
                enhanced_query, candidates
            )
            
            # Combine scores and rerank
            reranked_docs = self._combine_and_rank(candidates, reranking_scores)
            
            # Apply diversity filtering
            final_results = self._apply_diversity_filtering(reranked_docs, enhanced_query)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Reranking completed in {processing_time:.3f}s")
            
            # Update metadata
            for i, doc in enumerate(final_results[:top_k]):
                doc["rerank_position"] = i + 1
                doc["reranking_time"] = processing_time
                doc["reranking_applied"] = True
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results on failure
            return retrieved_docs[:top_k] if top_k else retrieved_docs
    
    def _enhance_query_with_context(
        self, 
        query: str, 
        conversation_context: Optional[List[str]]
    ) -> str:
        """Enhance query with conversation context for better reranking."""
        if not conversation_context:
            return query
        
        # Extract key terms from recent user questions
        context_terms = []
        for question in conversation_context[-2:]:  # Last 2 questions
            # Extract important nouns and entities (simple approach)
            words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', question)
            context_terms.extend(words[:3])  # Top 3 terms per question
        
        if context_terms:
            # Add most frequent context terms to query
            term_counts = Counter(context_terms)
            top_terms = [term for term, _ in term_counts.most_common(3)]
            enhanced_query = f"{query} {' '.join(top_terms)}"
            logger.debug(f"Enhanced query: {enhanced_query}")
            return enhanced_query
        
        return query
    
    async def _compute_reranking_scores(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Compute different types of reranking scores."""
        scores = {
            "cross_encoder": [],
            "semantic_similarity": [],
            "heuristic": [],
            "diversity": []
        }
        
        # Cross-encoder scores
        if self.cross_encoder:
            scores["cross_encoder"] = await self._compute_cross_encoder_scores(query, candidates)
        else:
            scores["cross_encoder"] = [0.0] * len(candidates)
        
        # Semantic similarity scores
        scores["semantic_similarity"] = self._compute_semantic_scores(query, candidates)
        
        # Heuristic scores
        scores["heuristic"] = self._compute_heuristic_scores(query, candidates)
        
        # Diversity scores
        scores["diversity"] = self._compute_diversity_scores(candidates)
        
        return scores
    
    async def _compute_cross_encoder_scores(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """Compute cross-encoder relevance scores."""
        if not self.cross_encoder:
            return [0.0] * len(candidates)
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in candidates:
                text = doc.get("text", "")[:512]  # Limit text length for performance
                pairs.append([query, text])
            
            # Run cross-encoder in thread pool to avoid blocking
            scores = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.cross_encoder.predict, 
                pairs
            )
            
            # Normalize scores to 0-1 range
            if len(scores) > 1:
                min_score, max_score = min(scores), max(scores)
                if max_score > min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return [0.0] * len(candidates)
    
    def _compute_semantic_scores(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """Compute semantic similarity scores using text overlap and patterns."""
        query_words = set(query.lower().split())
        scores = []
        
        for doc in candidates:
            text = doc.get("text", "").lower()
            doc_words = set(text.split())
            
            # Word overlap score
            overlap = len(query_words.intersection(doc_words))
            overlap_score = overlap / len(query_words) if query_words else 0
            
            # Phrase matching score
            phrase_score = 0
            for word in query_words:
                if len(word) > 3 and word in text:
                    phrase_score += 1
            phrase_score = phrase_score / len(query_words) if query_words else 0
            
            # Position-based scoring (early matches are better)
            position_score = 0
            text_lower = text[:500]  # Focus on beginning
            for word in query_words:
                if word in text_lower:
                    pos = text_lower.find(word)
                    position_score += max(0, 1 - pos / 500)
            position_score = position_score / len(query_words) if query_words else 0
            
            # Combine semantic signals
            semantic_score = (overlap_score * 0.4 + phrase_score * 0.4 + position_score * 0.2)
            scores.append(semantic_score)
        
        return scores
    
    def _compute_heuristic_scores(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """Compute heuristic scores based on document characteristics."""
        scores = []
        
        for doc in candidates:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Length score (moderate length preferred)
            length = len(text.split())
            length_score = min(length / 200, 1.0)  # Optimal around 200 words
            if length > 400:
                length_score *= 0.8  # Penalty for very long chunks
            
            # Structure score (presence of punctuation, formatting)
            sentences = text.count('.') + text.count('!') + text.count('?')
            structure_score = min(sentences / 5, 1.0)  # Up to 5 sentences
            
            # Chunk position score (earlier chunks often more important)
            chunk_index = metadata.get("chunk_index", 0)
            position_score = max(0, 1 - chunk_index / 10)  # Decay over first 10 chunks
            
            # Originalpng retrieval score
            original_score = doc.get("combined_score", doc.get("dense_score", 0))
            
            # Combine heuristic signals
            heuristic_score = (
                length_score * 0.3 + 
                structure_score * 0.2 + 
                position_score * 0.2 + 
                original_score * 0.3
            )
            scores.append(heuristic_score)
        
        return scores
    
    def _compute_diversity_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Compute diversity scores to avoid redundant results."""
        scores = [1.0] * len(candidates)  # Start with max diversity
        
        # Penalize documents that are too similar to higher-ranked ones
        for i in range(len(candidates)):
            for j in range(i):
                # Simple text similarity check
                text_i = candidates[i].get("text", "").lower()
                text_j = candidates[j].get("text", "").lower()
                
                # Jaccard similarity
                words_i = set(text_i.split())
                words_j = set(text_j.split())
                
                if words_i and words_j:
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    similarity = intersection / union if union > 0 else 0
                    
                    # Penalize if too similar
                    if similarity > 0.7:
                        scores[i] *= 0.5  # Reduce diversity score
        
        return scores
    
    def _combine_and_rank(
        self, 
        candidates: List[Dict[str, Any]], 
        reranking_scores: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """Combine different scoring signals and rank documents."""
        combined_docs = []
        
        for i, doc in enumerate(candidates):
            # Weighted combination of all scores
            final_score = (
                reranking_scores["cross_encoder"][i] * self.cross_encoder_weight +
                reranking_scores["semantic_similarity"][i] * self.semantic_weight +
                reranking_scores["heuristic"][i] * self.heuristic_weight +
                reranking_scores["diversity"][i] * self.diversity_weight
            )
            
            # Create new document with reranking information
            reranked_doc = doc.copy()
            reranked_doc.update({
                "rerank_score": final_score,
                "cross_encoder_score": reranking_scores["cross_encoder"][i],
                "semantic_score": reranking_scores["semantic_similarity"][i],
                "heuristic_score": reranking_scores["heuristic"][i],
                "diversity_score": reranking_scores["diversity"][i],
                "original_score": doc.get("combined_score", 0)
            })
            
            combined_docs.append(reranked_doc)
        
        # Sort by final rerank score
        combined_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return combined_docs
    
    def _apply_diversity_filtering(
        self, 
        ranked_docs: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Apply final diversity filtering to avoid redundant results."""
        if len(ranked_docs) <= 3:
            return ranked_docs
        
        filtered_docs = [ranked_docs[0]]  # Always include top result
        
        for doc in ranked_docs[1:]:
            # Check if this document adds sufficient diversity
            is_diverse = True
            
            for existing_doc in filtered_docs:
                # Simple content similarity check
                text1 = doc.get("text", "").lower()[:300]
                text2 = existing_doc.get("text", "").lower()[:300]
                
                words1 = set(text1.split())
                words2 = set(text2.split())
                
                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.8:  # Too similar
                        is_diverse = False
                        break
            
            if is_diverse:
                filtered_docs.append(doc)
            
            # Limit total results for performance
            if len(filtered_docs) >= 10:
                break
        
        return filtered_docs
    
    def get_reranker_stats(self) -> Dict[str, Any]:
        """Get statistics about the reranker configuration."""
        return {
            "cross_encoder_available": self.cross_encoder is not None,
            "max_rerank_candidates": self.max_rerank_candidates,
            "weights": {
                "cross_encoder": self.cross_encoder_weight,
                "semantic": self.semantic_weight,
                "heuristic": self.heuristic_weight,
                "diversity": self.diversity_weight
            },
            "model_info": {
                "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2" if self.cross_encoder else None
            }
        }


class PerformanceReranker:
    """
    Lightweight reranker optimized for speed when cross-encoder is not available.
    Focuses on fast heuristic and lexical matching strategies.
    """
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
    
    async def fast_rerank(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Fast reranking using only lightweight signals."""
        if not retrieved_docs:
            return []
        
        start_time = datetime.utcnow()
        
        # Compute fast scores
        query_terms = set(query.lower().split())
        
        for doc in retrieved_docs:
            text = doc.get("text", "").lower()
            
            # Fast lexical matching
            doc_terms = set(text.split())
            term_overlap = len(query_terms.intersection(doc_terms))
            overlap_ratio = term_overlap / len(query_terms) if query_terms else 0
            
            # Exact phrase matching
            exact_matches = sum(1 for term in query_terms if term in text)
            exact_ratio = exact_matches / len(query_terms) if query_terms else 0
            
            # Position-based scoring
            position_score = 0
            for term in query_terms:
                if term in text:
                    pos = text.find(term)
                    position_score += max(0, 1 - pos / len(text))
            position_score = position_score / len(query_terms) if query_terms else 0
            
            # Combine fast signals
            fast_score = (overlap_ratio * 0.4 + exact_ratio * 0.4 + position_score * 0.2)
            
            # Weight with original score
            original_score = doc.get("combined_score", doc.get("score", 0))
            doc["fast_rerank_score"] = fast_score * 0.6 + original_score * 0.4
        
        # Sort by fast rerank score
        retrieved_docs.sort(key=lambda x: x.get("fast_rerank_score", 0), reverse=True)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Add metadata
        for i, doc in enumerate(retrieved_docs[:top_k]):
            doc["rerank_position"] = i + 1
            doc["reranking_time"] = processing_time
            doc["reranking_type"] = "fast"
        
        logger.info(f"Fast reranking completed in {processing_time:.3f}s")
        
        return retrieved_docs[:top_k]
