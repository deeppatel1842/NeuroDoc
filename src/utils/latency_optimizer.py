"""
Advanced Latency Optimization Module

This module implements advanced optimization strategies to achieve
consistent <200ms retrieval latency in real-world scenarios.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LatencyOptimizer:
    """
    Advanced latency optimizer that implements multiple strategies
    to achieve consistent sub-200ms retrieval performance.
    """
    
    def __init__(self):
        self.optimization_strategies = {
            "embedding_cache": EmbeddingCache(),
            "result_cache": ResultCache(),
            "batch_optimizer": BatchOptimizer(),
            "parallel_processor": ParallelProcessor(),
            "adaptive_parameters": AdaptiveParameterManager()
        }
        
        # Performance targets
        self.latency_target = 0.2  # 200ms
        self.percentile_target = 95  # 95th percentile must be under target
        
        # Monitoring
        self.recent_latencies = deque(maxlen=100)
        self.optimization_history = []
        
    async def optimize_retrieval_pipeline(
        self, 
        retriever, 
        query: str, 
        session_id: str, 
        top_k: int = 10,
        conversation_history: Optional[List] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimized retrieval pipeline with multiple acceleration strategies.
        
        Returns:
            Tuple of (results, optimization_metadata)
        """
        start_time = time.time()
        optimization_metadata = {"strategies_used": []}
        
        try:
            # Strategy 1: Check result cache first
            cache_key = self._generate_cache_key(query, session_id, top_k)
            cached_result = await self.optimization_strategies["result_cache"].get(cache_key)
            
            if cached_result:
                optimization_metadata["strategies_used"].append("result_cache")
                optimization_metadata["cache_hit"] = True
                return cached_result, optimization_metadata
            
            # Strategy 2: Optimize embedding generation
            query_embedding = await self.optimization_strategies["embedding_cache"].get_or_generate(
                query, retriever.embedding_generator
            )
            if query_embedding is not None:
                optimization_metadata["strategies_used"].append("embedding_cache")
            
            # Strategy 3: Parallel dense and sparse retrieval
            dense_task = asyncio.create_task(
                self._optimized_dense_retrieval(retriever, query_embedding, session_id, top_k)
            )
            sparse_task = asyncio.create_task(
                self._optimized_sparse_retrieval(retriever, query, session_id, top_k)
            )
            
            # Wait for both retrievals with timeout
            try:
                dense_results, sparse_results = await asyncio.wait_for(
                    asyncio.gather(dense_task, sparse_task),
                    timeout=self.latency_target * 0.8  # Use 80% of target for retrieval
                )
                optimization_metadata["strategies_used"].append("parallel_retrieval")
            except asyncio.TimeoutError:
                logger.warning("Retrieval timeout, using partial results")
                dense_results = dense_task.result() if dense_task.done() else []
                sparse_results = sparse_task.result() if sparse_task.done() else []
                optimization_metadata["timeout_occurred"] = True
            
            # Strategy 4: Fast result combination and ranking
            combined_results = await self._fast_combine_results(
                dense_results, sparse_results, top_k
            )
            
            # Strategy 5: Adaptive reranking based on time budget
            elapsed_time = time.time() - start_time
            time_remaining = self.latency_target - elapsed_time
            
            if time_remaining > 0.05:  # At least 50ms left
                final_results = await self._adaptive_reranking(
                    retriever, query, combined_results, time_remaining
                )
                optimization_metadata["strategies_used"].append("adaptive_reranking")
            else:
                final_results = combined_results[:top_k]
                optimization_metadata["strategies_used"].append("skip_reranking")
            
            # Cache the results for future use
            await self.optimization_strategies["result_cache"].set(
                cache_key, final_results, ttl=300  # 5 minutes
            )
            
            total_time = time.time() - start_time
            self.recent_latencies.append(total_time)
            optimization_metadata["total_latency"] = total_time
            optimization_metadata["cache_hit"] = False
            
            return final_results, optimization_metadata
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}")
            # Fallback to standard retrieval
            results = await retriever.retrieve(query, session_id, top_k, conversation_history)
            optimization_metadata["fallback_used"] = True
            return results, optimization_metadata
    
    async def _optimized_dense_retrieval(
        self, retriever, query_embedding, session_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Optimized dense retrieval with pre-computed embeddings."""
        if session_id not in retriever.vector_stores:
            return []
        
        try:
            # Use pre-computed embedding
            if query_embedding is not None:
                results = await retriever.vector_stores[session_id].search(
                    query_embedding, top_k
                )
            else:
                # Fallback to standard method
                results = await retriever._dense_retrieve("", session_id, top_k)
            
            return results
            
        except Exception as e:
            logger.error(f"Optimized dense retrieval failed: {e}")
            return []
    
    async def _optimized_sparse_retrieval(
        self, retriever, query: str, session_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Optimized sparse retrieval with caching."""
        return await retriever._sparse_retrieve(query, session_id, top_k)
    
    async def _fast_combine_results(
        self, dense_results: List, sparse_results: List, top_k: int
    ) -> List[Dict[str, Any]]:
        """Fast result combination using pre-computed weights."""
        combined_scores = {}
        
        # Dense results (higher weight)
        for result in dense_results:
            chunk_id = result.get("chunk_id", result.get("id", ""))
            combined_scores[chunk_id] = {
                "result": result,
                "score": result.get("similarity", 0) * 0.7  # 70% weight
            }
        
        # Sparse results (lower weight) 
        for result in sparse_results:
            chunk_id = result.get("chunk_id", result.get("id", ""))
            if chunk_id in combined_scores:
                combined_scores[chunk_id]["score"] += result.get("score", 0) * 0.3
            else:
                combined_scores[chunk_id] = {
                    "result": result,
                    "score": result.get("score", 0) * 0.3
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["result"] for item in sorted_results[:top_k]]
    
    async def _adaptive_reranking(
        self, retriever, query: str, results: List, time_budget: float
    ) -> List[Dict[str, Any]]:
        """Adaptive reranking based on available time budget."""
        if time_budget < 0.02:  # Less than 20ms
            return results
        
        try:
            if time_budget > 0.1 and hasattr(retriever, 'advanced_reranker'):
                # Use advanced reranker if we have time
                return await retriever.advanced_reranker.rerank_results(
                    query, results, len(results)
                )
            elif hasattr(retriever, 'performance_reranker'):
                # Use fast reranker
                return await retriever.performance_reranker.fast_rerank(
                    query, results, len(results)
                )
            else:
                return results
                
        except Exception as e:
            logger.error(f"Adaptive reranking failed: {e}")
            return results
    
    def _generate_cache_key(self, query: str, session_id: str, top_k: int) -> str:
        """Generate cache key for query results."""
        import hashlib
        content = f"{query}:{session_id}:{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get current latency statistics."""
        if not self.recent_latencies:
            return {}
        
        latencies = list(self.recent_latencies)
        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "count": len(latencies)
        }
    
    def should_optimize(self) -> bool:
        """Check if optimization is needed based on recent performance."""
        stats = self.get_latency_stats()
        if not stats:
            return False
        
        return stats.get("p95", 0) > self.latency_target


class EmbeddingCache:
    """High-performance embedding cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    async def get_or_generate(self, text: str, generator) -> Optional[np.ndarray]:
        """Get embedding from cache or generate if not present."""
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        with self.lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]
        
        # Generate new embedding
        try:
            embedding = await generator.generate_single_embedding(text)
            
            with self.lock:
                # Evict oldest if cache is full
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.access_times.keys(), 
                                   key=lambda k: self.access_times[k])
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
                
                self.cache[cache_key] = embedding
                self.access_times[cache_key] = time.time()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None


class ResultCache:
    """Cache for retrieval results with TTL."""
    
    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.ttl_cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    async def get(self, key: str) -> Optional[List]:
        """Get cached results if not expired."""
        with self.lock:
            if key in self.cache:
                if time.time() - self.ttl_cache[key] < 300:  # 5 minutes TTL
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.ttl_cache[key]
        return None
    
    async def set(self, key: str, value: List, ttl: int = 300):
        """Cache results with TTL."""
        with self.lock:
            # Evict random item if cache is full
            if len(self.cache) >= self.max_size:
                evict_key = next(iter(self.cache))
                del self.cache[evict_key]
                del self.ttl_cache[evict_key]
            
            self.cache[key] = value
            self.ttl_cache[key] = time.time()


class BatchOptimizer:
    """Optimizes batch processing for better throughput."""
    
    def __init__(self):
        self.optimal_batch_sizes = {
            "embedding_generation": 16,
            "vector_search": 1,
            "text_processing": 8
        }
    
    def get_optimal_batch_size(self, operation: str) -> int:
        """Get optimal batch size for operation."""
        return self.optimal_batch_sizes.get(operation, 1)


class ParallelProcessor:
    """Manages parallel processing for improved performance."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_parallel(self, tasks: List[Callable]) -> List[Any]:
        """Process tasks in parallel."""
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(self.executor, task) for task in tasks]
        return await asyncio.gather(*futures)


class AdaptiveParameterManager:
    """Manages adaptive parameter tuning based on performance."""
    
    def __init__(self):
        self.current_parameters = {
            "top_k_multiplier": 2.0,
            "reranking_threshold": 10,
            "cache_ttl": 300,
            "batch_size": 16
        }
        
        self.parameter_history = defaultdict(list)
    
    def adjust_parameters(self, performance_stats: Dict[str, float]):
        """Adjust parameters based on performance feedback."""
        p95_latency = performance_stats.get("p95", 0)
        
        if p95_latency > 0.2:  # Over target
            # Reduce quality for speed
            self.current_parameters["top_k_multiplier"] = max(1.5, 
                self.current_parameters["top_k_multiplier"] * 0.9)
            self.current_parameters["reranking_threshold"] = min(20,
                self.current_parameters["reranking_threshold"] + 2)
        elif p95_latency < 0.15:  # Well under target
            # Increase quality
            self.current_parameters["top_k_multiplier"] = min(3.0,
                self.current_parameters["top_k_multiplier"] * 1.1)
            self.current_parameters["reranking_threshold"] = max(5,
                self.current_parameters["reranking_threshold"] - 1)
    
    def get_parameter(self, name: str) -> Any:
        """Get current parameter value."""
        return self.current_parameters.get(name)
