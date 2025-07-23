"""
Hybrid Retrieval System

Implements a hybrid retrieval system combining dense vector search
with sparse BM25 retrieval for optimal RAG performance.
"""

import logging
import asyncio
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import re

from ..config import RETRIEVAL_CONFIG
from ..embeddings.generator import EmbeddingGenerator, EmbeddingStorage
from ..embeddings.vector_store import VectorStore
from ..utils.text_utils import TextCleaner
from ..utils.performance_integration import monitor_performance, performance_context
from .reranker import AdvancedReranker, PerformanceReranker

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 sparse retrieval implementation for keyword-based search.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        
        self.corpus = []
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avgdl = 0
        self.N = 0
        self.idf_cache = {}
        
        # Text preprocessing
        self.text_cleaner = TextCleaner()
        self.stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> set:
        """Get common English stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'would', 'there', 'we', 'when', 'where', 'can', 'if', 'do', 'no',
            'all', 'any', 'may', 'your', 'about', 'could', 'or', 'should'
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and preprocess text."""
        # Basic tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Remove stopwords and very short tokens
        tokens = [token for token in tokens 
                 if token not in self.stopwords and len(token) > 2]
        
        return tokens
    
    def fit(self, documents: List[str]):
        """Fit BM25 model on document corpus."""
        self.corpus = []
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        
        # Process documents
        for doc in documents:
            tokens = self._tokenize(doc)
            self.corpus.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.N = len(self.corpus)
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Precompute IDF values
        self.idf_cache = {}
        for term in self.doc_freqs:
            self.idf_cache[term] = self._compute_idf(term)
    
    def _compute_idf(self, term: str) -> float:
        """Compute IDF score for a term."""
        df = self.doc_freqs[term]
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
    
    def _compute_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a document given query tokens."""
        doc_tokens = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        doc_token_counts = Counter(doc_tokens)
        
        for token in query_tokens:
            if token in self.idf_cache:
                tf = doc_token_counts[token]
                idf = self.idf_cache[token]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for relevant documents using BM25."""
        if not self.corpus:
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Compute scores for all documents
        scores = []
        for doc_idx in range(self.N):
            score = self._compute_score(query_tokens, doc_idx)
            scores.append((doc_idx, score))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """
    Hybrid retrieval system combining dense vector search with sparse BM25.
    """
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.embedding_storage = EmbeddingStorage()
        self.vector_stores: Dict[str, VectorStore] = {}
        self.bm25_retrievers: Dict[str, BM25Retriever] = {}
        
        # Main knowledge base store (shared across sessions)
        self.main_vector_store: Optional[VectorStore] = None
        
        # Initialize rerankers
        self.advanced_reranker = AdvancedReranker()
        self.performance_reranker = PerformanceReranker()
        
        # Retrieval configuration
        self.dense_weight = RETRIEVAL_CONFIG.get("dense_weight", 0.7)
        self.sparse_weight = RETRIEVAL_CONFIG.get("sparse_weight", 0.3)
        self.min_score_threshold = RETRIEVAL_CONFIG.get("min_score_threshold", 0.1)
        self.enable_reranking = RETRIEVAL_CONFIG.get("enable_reranking", True)
        self.reranking_threshold = RETRIEVAL_CONFIG.get("reranking_threshold", 5)  # Min docs to trigger reranking
        
    async def initialize(self):
        """Initialize the retriever by loading the main knowledge base."""
        try:
            # Try to load the main knowledge base
            self.main_vector_store = VectorStore(
                dimension=self.embedding_generator.get_embedding_dimension()
            )
            await self.main_vector_store.load_index("main_index")
            logger.info("Main knowledge base loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load main knowledge base: {e}")
            self.main_vector_store = None
        
    async def health_check(self) -> bool:
        """Check if the hybrid retriever is healthy."""
        try:
            # Test embedding generation
            test_embedding = await self.embedding_generator.generate_single_embedding("test")
            
            # Check if main vector store is loaded and has vectors
            main_store_healthy = (
                self.main_vector_store is not None and 
                hasattr(self.main_vector_store, 'index') and 
                self.main_vector_store.index is not None and 
                self.main_vector_store.index.ntotal > 0
            )
            
            return len(test_embedding) > 0 and main_store_healthy
        except Exception as e:
            logger.error(f"Hybrid retriever health check failed: {e}")
            return False
    
    async def index_documents(
        self, 
        session_id: str, 
        document_id: str, 
        chunks: List[Dict[str, Any]]
    ):
        """
        Index document chunks for both dense and sparse retrieval.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            chunks: List of text chunks with metadata
        """
        try:
            logger.info(f"Indexing {len(chunks)} chunks for document {document_id}")
            
            # Extract text from chunks
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate dense embeddings
            embeddings = await self.embedding_generator.generate_embeddings(texts)
            
            # Store embeddings
            await self.embedding_storage.store_embeddings(
                session_id, document_id, chunks, embeddings
            )
            
            # Initialize or get vector store for session
            if session_id not in self.vector_stores:
                self.vector_stores[session_id] = VectorStore(
                    dimension=self.embedding_generator.get_embedding_dimension()
                )
                # Try to load existing index
                await self.vector_stores[session_id].load_index(session_id)
            
            # Add vectors to vector store
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "session_id": session_id,
                    "document_id": document_id,
                    "chunk_id": chunk.get("chunk_id", f"{document_id}_chunk_{i}"),
                    "chunk_index": chunk.get("chunk_index", i),
                    "text": chunk["text"],
                    "word_count": chunk.get("word_count", len(chunk["text"].split())),
                    "created_at": datetime.utcnow().isoformat()
                }
                chunk_metadata.append(metadata)
            
            await self.vector_stores[session_id].add_vectors(embeddings, chunk_metadata)
            
            # Update BM25 index for session
            await self._update_bm25_index(session_id)
            
            # Save vector store
            await self.vector_stores[session_id].save_index(session_id)
            
            logger.info(f"Successfully indexed document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to index document {document_id}: {e}")
            raise
    
    async def _update_bm25_index(self, session_id: str):
        """Update BM25 index with all documents in session."""
        try:
            # Load all embeddings for session
            session_embeddings = await self.embedding_storage.load_session_embeddings(session_id)
            
            # Collect all text chunks
            all_texts = []
            for doc_id, (embeddings, chunks, metadata) in session_embeddings.items():
                for chunk in chunks:
                    all_texts.append(chunk["text"])
            
            # Fit BM25 retriever
            if session_id not in self.bm25_retrievers:
                self.bm25_retrievers[session_id] = BM25Retriever()
            
            self.bm25_retrievers[session_id].fit(all_texts)
            
            logger.debug(f"Updated BM25 index for session {session_id} with {len(all_texts)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")
    
    @monitor_performance("retrieval_latency", include_args=True)
    async def retrieve(
        self, 
        query: str, 
        session_id: str, 
        top_k: int = 10,
        conversation_history: Optional[List[str]] = None,
        session_document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Logic:
        - If user has uploaded documents (session_document_ids), search ONLY in those
        - If no uploaded documents, use the pre-loaded 10 documents as knowledge base
        
        Args:
            query: Search query
            session_id: Session identifier
            top_k: Number of results to return
            conversation_history: Recent user questions for context
            session_document_ids: Document IDs uploaded by user in this session
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            enhanced_query = query
            
            # Check if user has uploaded any documents
            if session_document_ids and len(session_document_ids) > 0:
                # USER MODE: Search ONLY in user's uploaded documents
                logger.info(f"USER MODE: Searching ONLY in user documents: {session_document_ids}")
                
                session_results = await self._retrieve_from_session_docs_only(
                    enhanced_query, session_id, session_document_ids, top_k
                )
                
                if len(session_results) == 0:
                    logger.warning("No results found in user documents")
                    return [{
                        "text": "I couldn't find any relevant information in your uploaded documents. Please make sure your question is related to the content you uploaded.",
                        "chunk_id": "no_results",
                        "document_id": "user_upload",
                        "score": 0.0,
                        "metadata": {"source": "no_results_message"}
                    }]
                
                logger.info(f"Found {len(session_results)} results from user documents")
                
                # Apply reranking if enabled
                if self.enable_reranking and len(session_results) >= self.reranking_threshold:
                    session_results = await self.reranker.rerank(enhanced_query, session_results)
                
                # Add final ranking
                for i, result in enumerate(session_results[:top_k]):
                    result["final_rank"] = i + 1
                    result["search_type"] = "user_documents_only"
                
                return session_results[:top_k]
            
            else:
                # DEFAULT MODE: Use pre-loaded 10 documents as knowledge base
                logger.info("DEFAULT MODE: No user documents, using pre-loaded knowledge base")
                
                # Perform dense retrieval on main knowledge base
                dense_results = await self._dense_retrieve(enhanced_query, session_id, top_k * 2)
                
                # Perform sparse retrieval on main knowledge base
                sparse_results = await self._sparse_retrieve(enhanced_query, session_id, top_k * 2)
                
                # Combine and rank results
                hybrid_results = self._combine_results(dense_results, sparse_results, top_k * 2)
                
                # Apply reranking if enabled and sufficient results
                if self.enable_reranking and len(hybrid_results) >= self.reranking_threshold:
                    start_rerank = datetime.utcnow()
                    
                    try:
                        # Use advanced reranker if cross-encoder is available, otherwise fast reranker
                        if self.advanced_reranker.cross_encoder:
                            reranked_results = await self.advanced_reranker.rerank_results(
                                enhanced_query, hybrid_results, top_k, None
                            )
                        else:
                            reranked_results = await self.performance_reranker.fast_rerank(
                                enhanced_query, hybrid_results, top_k
                            )
                        
                        rerank_time = (datetime.utcnow() - start_rerank).total_seconds()
                        logger.info(f"Reranking completed in {rerank_time:.3f}s")
                        
                        final_results = reranked_results
                        
                    except Exception as e:
                        logger.error(f"Reranking failed, using original results: {e}")
                        final_results = hybrid_results[:top_k]
                else:
                    final_results = hybrid_results[:top_k]
                
                # Add final ranking
                for i, result in enumerate(final_results[:top_k]):
                    result["final_rank"] = i + 1
                    result["search_type"] = "main_knowledge_base"
                
                logger.info(f"Retrieved {len(final_results)} chunks for query in session {session_id}")
                return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _enhance_query_with_context(
        self, 
        query: str, 
        conversation_history: Optional[List[str]]
    ) -> str:
        """Enhance query with recent conversation context."""
        if not conversation_history:
            return query
        
        # Extract recent questions and keywords
        recent_context = conversation_history[-3:]  # Last 3 user questions
        
        # Simple context enhancement - could be improved with NLP
        if recent_context:
            context_keywords = []
            for question in recent_context:
                # Extract important keywords (simple approach)
                words = question.lower().split()
                keywords = [w for w in words if len(w) > 4 and w.isalpha()]
                context_keywords.extend(keywords[:3])  # Top 3 keywords per question
            
            if context_keywords:
                # Add context keywords to query
                enhanced_query = f"{query} {' '.join(set(context_keywords))}"
                return enhanced_query
        
        return query
    
    @monitor_performance("dense_retrieval")
    async def _dense_retrieve(
        self, 
        query: str, 
        session_id: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform dense vector retrieval. Prioritizes main knowledge base."""
        # Always try main knowledge base first (contains pre-loaded documents)
        vector_store = self.main_vector_store
        
        # Fallback to session-specific store if main store not available
        if vector_store is None and session_id in self.vector_stores:
            vector_store = self.vector_stores[session_id]
        
        if vector_store is None:
            logger.warning("No vector store available for dense retrieval")
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_single_embedding(query)
            
            # Search vector store
            results = await vector_store.search(
                query_embedding, 
                top_k=top_k,
                score_threshold=self.min_score_threshold
            )
            
            # Format results
            dense_results = []
            for result in results:
                dense_results.append({
                    "chunk_id": result["metadata"].get("chunk_id", ""),
                    "document_id": result["metadata"].get("document_id", ""),
                    "text": result["metadata"].get("text", ""),
                    "dense_score": result["score"],
                    "rank": result["rank"],
                    "retrieval_type": "dense",
                    "metadata": result["metadata"]
                })
            
            return dense_results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    @monitor_performance("sparse_retrieval")
    async def _sparse_retrieve(
        self, 
        query: str, 
        session_id: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform sparse BM25 retrieval."""
        if session_id not in self.bm25_retrievers:
            return []
        
        try:
            # Get BM25 results
            bm25_results = self.bm25_retrievers[session_id].search(query, top_k)
            
            # Load session embeddings to get chunk metadata
            session_embeddings = await self.embedding_storage.load_session_embeddings(session_id)
            
            # Create flat list of chunks with indices
            all_chunks = []
            for doc_id, (embeddings, chunks, metadata) in session_embeddings.items():
                for chunk in chunks:
                    all_chunks.append({
                        "document_id": doc_id,
                        "chunk": chunk
                    })
            
            # Format results
            sparse_results = []
            for doc_idx, score in bm25_results:
                if doc_idx < len(all_chunks):
                    chunk_data = all_chunks[doc_idx]
                    chunk = chunk_data["chunk"]
                    
                    sparse_results.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "document_id": chunk_data["document_id"],
                        "text": chunk["text"],
                        "sparse_score": score,
                        "rank": len(sparse_results) + 1,
                        "retrieval_type": "sparse",
                        "metadata": chunk
                    })
            
            return sparse_results
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def _combine_results(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine dense and sparse results using weighted scoring."""
        # Create a map of chunk_id to results
        combined_map = {}
        
        # Add dense results
        for result in dense_results:
            chunk_id = result["chunk_id"]
            combined_map[chunk_id] = {
                **result,
                "dense_score": result["dense_score"],
                "sparse_score": 0.0,
                "combined_score": self.dense_weight * result["dense_score"]
            }
        
        # Add sparse results
        for result in sparse_results:
            chunk_id = result["chunk_id"]
            
            # Normalize BM25 score to 0-1 range (simple approach)
            normalized_sparse_score = min(result["sparse_score"] / 10.0, 1.0)
            
            if chunk_id in combined_map:
                # Update existing result
                combined_map[chunk_id]["sparse_score"] = normalized_sparse_score
                combined_map[chunk_id]["combined_score"] = (
                    self.dense_weight * combined_map[chunk_id]["dense_score"] +
                    self.sparse_weight * normalized_sparse_score
                )
                combined_map[chunk_id]["retrieval_type"] = "hybrid"
            else:
                # Add new result
                combined_map[chunk_id] = {
                    **result,
                    "dense_score": 0.0,
                    "sparse_score": normalized_sparse_score,
                    "combined_score": self.sparse_weight * normalized_sparse_score,
                    "retrieval_type": "sparse"
                }
        
        # Sort by combined score and return top k
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add final ranking
        for i, result in enumerate(combined_results[:top_k]):
            result["final_rank"] = i + 1
        
        return combined_results[:top_k]
    
    async def _retrieve_from_session_docs_only(
        self, 
        query: str, 
        session_id: str,
        session_document_ids: List[str], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ONLY from session documents, ignoring main knowledge base.
        
        Args:
            query: Search query
            session_id: Session identifier
            session_document_ids: List of document IDs to search in
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks from session documents only
        """
        try:
            logger.info(f"Searching exclusively in session documents: {session_document_ids}")
            
            # Get session-specific vector store
            session_vector_store = self.vector_stores.get(session_id)
            if not session_vector_store:
                logger.warning(f"No session vector store found for session {session_id}")
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_single_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in session vector store only
            all_results = await session_vector_store.search(
                query_embedding, 
                top_k=top_k * 3,  # Get more results to filter
                score_threshold=self.min_score_threshold
            )
            
            # Filter results to only include documents from session
            session_results = []
            for result in all_results:
                doc_id = result.get("metadata", {}).get("document_id", "")
                if doc_id in session_document_ids:
                    session_results.append({
                        "chunk_id": result["metadata"].get("chunk_id", ""),
                        "document_id": doc_id,
                        "text": result["metadata"].get("text", ""),
                        "score": result["score"],
                        "rank": result.get("rank", 0),
                        "retrieval_type": "session_only",
                        "metadata": result["metadata"]
                    })
                    
                    if len(session_results) >= top_k:
                        break
            
            logger.info(f"Found {len(session_results)} chunks from session documents")
            return session_results
            
        except Exception as e:
            logger.error(f"Error retrieving from session documents only: {e}")
            return []
    
    async def _retrieve_from_session_docs(
        self, 
        query: str, 
        session_document_ids: List[str], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve specifically from session documents.
        
        Args:
            query: Search query
            session_document_ids: List of document IDs to search in
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks from session documents
        """
        try:
            # Get embedding for query
            query_embedding = await self.embedding_generator.generate_embeddings([query])
            if not query_embedding:
                return []
            
            # Search in vector store with document ID filter
            results = []
            
            # Get all embeddings and filter by document ID
            if hasattr(self.vector_store, 'search_with_filter'):
                # If vector store supports filtering
                results = await self.vector_store.search_with_filter(
                    query_embedding[0], 
                    top_k, 
                    filter_field="document_id",
                    filter_values=session_document_ids
                )
            else:
                # Fallback: search all and filter results
                all_results = await self.vector_store.search(query_embedding[0], top_k * 5)
                
                # Filter by document_id
                for result in all_results:
                    if result.get("metadata", {}).get("document_id") in session_document_ids:
                        results.append(result)
                        if len(results) >= top_k:
                            break
            
            # Add source info
            for result in results:
                result["retrieval_method"] = "session_priority"
                result["score"] = result.get("score", 0) + 0.3  # Boost session docs
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving from session documents: {e}")
            return []
    
    def _combine_with_session_priority(
        self,
        session_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results with session document priority.
        
        Args:
            session_results: Results from session documents
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
            top_k: Number of results to return
            
        Returns:
            Combined and prioritized results
        """
        # Create a combined list starting with session results
        combined_results = []
        seen_chunks = set()
        
        # Add session results first (highest priority)
        for result in session_results:
            chunk_id = result.get("id", result.get("chunk_id", ""))
            if chunk_id and chunk_id not in seen_chunks:
                result["priority"] = "session"
                combined_results.append(result)
                seen_chunks.add(chunk_id)
        
        # Combine dense and sparse results
        all_other_results = dense_results + sparse_results
        
        # Add remaining results, avoiding duplicates
        for result in all_other_results:
            chunk_id = result.get("id", result.get("chunk_id", ""))
            if chunk_id and chunk_id not in seen_chunks:
                result["priority"] = "general"
                combined_results.append(result)
                seen_chunks.add(chunk_id)
        
        # Sort by score, but keep session docs at top
        session_docs = [r for r in combined_results if r.get("priority") == "session"]
        other_docs = [r for r in combined_results if r.get("priority") == "general"]
        
        # Sort each group by score
        session_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        other_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Combine: session docs first, then others
        final_results = session_docs + other_docs
        
        # Add final ranking
        for i, result in enumerate(final_results[:top_k]):
            result["final_rank"] = i + 1
        
        return final_results[:top_k]
    
    async def delete_document_embeddings(self, session_id: str, document_id: str):
        """Delete embeddings for a specific document."""
        try:
            # Delete from embedding storage
            await self.embedding_storage.delete_embeddings(session_id, document_id)
            
            # Update BM25 index (rebuild with remaining documents)
            await self._update_bm25_index(session_id)
            
            # Note: Vector store deletion is complex with FAISS, would need rebuild
            logger.info(f"Deleted embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete document embeddings: {e}")
            raise
    
    async def delete_session_data(self, session_id: str):
        """Delete all data for a session."""
        try:
            # Delete embedding storage
            await self.embedding_storage.delete_session_embeddings(session_id)
            
            # Clear vector store
            if session_id in self.vector_stores:
                del self.vector_stores[session_id]
            
            # Clear BM25 retriever
            if session_id in self.bm25_retrievers:
                del self.bm25_retrievers[session_id]
            
            logger.info(f"Deleted all data for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete session data: {e}")
            raise
    
    def get_stats(self, session_id: str) -> Dict[str, Any]:
        """Get retrieval statistics for a session."""
        stats = {
            "session_id": session_id,
            "has_vector_store": session_id in self.vector_stores,
            "has_bm25_retriever": session_id in self.bm25_retrievers,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight
        }
        
        if session_id in self.vector_stores:
            stats["vector_store_stats"] = self.vector_stores[session_id].get_stats()
        
        if session_id in self.bm25_retrievers:
            stats["bm25_corpus_size"] = self.bm25_retrievers[session_id].N
        
        return stats
