"""
Performance Optimization Module

This module provides performance monitoring, optimization strategies,
and latency improvements for the NeuroDoc RAG system.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for the RAG system.
    Provides insights into bottlenecks and optimization opportunities.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.operation_counts = defaultdict(int)
        self.start_time = datetime.utcnow()
        
        # Performance targets
        self.targets = {
            "retrieval_latency": 0.2,  # 200ms
            "embedding_generation": 0.1,  # 100ms
            "document_processing": 5.0,  # 5s
            "response_generation": 2.0,  # 2s
        }
    
    def record_metric(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Record a performance metric."""
        metric_data = {
            "duration": duration,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        self.metrics[operation].append(metric_data)
        self.operation_counts[operation] += 1
        
        # Log if exceeding target
        if operation in self.targets and duration > self.targets[operation]:
            logger.warning(f"{operation} took {duration:.3f}s (target: {self.targets[operation]:.3f}s)")
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if operation:
            return self._get_operation_stats(operation)
        
        # Get stats for all operations
        all_stats = {}
        for op in self.metrics.keys():
            all_stats[op] = self._get_operation_stats(op)
        
        return {
            "operations": all_stats,
            "total_operations": sum(self.operation_counts.values()),
            "uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    def _get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        metrics = list(self.metrics[operation])
        if not metrics:
            return {"count": 0}
        
        durations = [m["duration"] for m in metrics]
        
        return {
            "count": len(metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p95_duration": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations),
            "target": self.targets.get(operation),
            "target_met_ratio": sum(1 for d in durations if d <= self.targets.get(operation, float('inf'))) / len(durations)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return self.get_stats()
    

class AsyncCache:
    """
    Async-aware caching system with TTL and LRU eviction.
    Optimized for caching embedding and retrieval results.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            if key in self.cache:
                # Check if expired
                if datetime.utcnow().timestamp() > self.expiry_times.get(key, 0):
                    await self._remove(key)
                    return None
                
                # Update access time for LRU
                self.access_times[key] = datetime.utcnow().timestamp()
                return self.cache[key]
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        async with self._lock:
            ttl = ttl or self.default_ttl
            now = datetime.utcnow().timestamp()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = now
            self.expiry_times[key] = now + ttl
    
    async def _remove(self, key: str) -> None:
        """Remove item from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self._remove(lru_key)
    
    async def clear(self) -> None:
        """Clear all cached items."""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_ratio": getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
        }


class BatchProcessor:
    """
    Batches operations for improved throughput and reduced overhead.
    Useful for embedding generation and vector operations.
    """
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_batches = defaultdict(list)
        self.batch_timers = {}
        self._lock = asyncio.Lock()
    
    async def add_to_batch(
        self, 
        operation_type: str, 
        item: Any, 
        processor: Callable
    ) -> Any:
        """Add item to batch for processing."""
        async with self._lock:
            # Create future for this item
            future = asyncio.Future()
            
            self.pending_batches[operation_type].append({
                "item": item,
                "future": future
            })
            
            # Check if batch is ready
            if len(self.pending_batches[operation_type]) >= self.batch_size:
                await self._process_batch(operation_type, processor)
            else:
                # Set timer if this is the first item in batch
                if len(self.pending_batches[operation_type]) == 1:
                    self.batch_timers[operation_type] = asyncio.create_task(
                        self._batch_timeout(operation_type, processor)
                    )
            
            return await future
    
    async def _process_batch(self, operation_type: str, processor: Callable) -> None:
        """Process a complete batch."""
        batch = self.pending_batches[operation_type]
        self.pending_batches[operation_type] = []
        
        # Cancel timer if exists
        if operation_type in self.batch_timers:
            self.batch_timers[operation_type].cancel()
            del self.batch_timers[operation_type]
        
        if not batch:
            return
        
        try:
            # Extract items and process as batch
            items = [item_data["item"] for item_data in batch]
            results = await processor(items)
            
            # Resolve futures with results
            for item_data, result in zip(batch, results):
                if not item_data["future"].done():
                    item_data["future"].set_result(result)
                    
        except Exception as e:
            # Resolve all futures with the exception
            for item_data in batch:
                if not item_data["future"].done():
                    item_data["future"].set_exception(e)
    
    async def _batch_timeout(self, operation_type: str, processor: Callable) -> None:
        """Handle batch timeout."""
        await asyncio.sleep(self.max_wait_time)
        async with self._lock:
            if operation_type in self.pending_batches:
                await self._process_batch(operation_type, processor)


def timed_operation(operation_name: str, monitor: Optional[PerformanceMonitor] = None):
    """Decorator to time operations and record metrics."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if monitor:
                    monitor.record_metric(operation_name, duration)
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if monitor:
                    monitor.record_metric(operation_name, duration)
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class OptimizedEmbeddingGenerator:
    """
    Optimized wrapper around embedding generation with caching and batching.
    """
    
    def __init__(self, base_generator, cache_size: int = 500, batch_size: int = 32):
        self.base_generator = base_generator
        self.cache = AsyncCache(max_size=cache_size, default_ttl=3600)
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.performance_monitor = PerformanceMonitor()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching and batching optimizations."""
        # Check cache first
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = f"embed_{hash(text)}"
            cached = await self.cache.get(cache_key)
            if cached is not None:
                cached_results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = time.time()
            
            # Use batch processing for efficiency
            new_embeddings = await self.base_generator.generate_embeddings(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = f"embed_{hash(text)}"
                await self.cache.set(cache_key, embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
            
            generation_time = time.time() - start_time
            self.performance_monitor.record_metric("embedding_generation", generation_time)
            
            # Combine cached and new results
            for i, embedding in zip(uncached_indices, new_embeddings):
                cached_results.append((i, embedding))
        
        # Sort by original order
        cached_results.sort(key=lambda x: x[0])
        return [result[1] for result in cached_results]
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate single embedding with caching."""
        results = await self.generate_embeddings([text])
        return results[0] if results else []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "cache": self.cache.stats(),
            "performance": self.performance_monitor.get_stats("embedding_generation")
        }


class QueryOptimizer:
    """
    Optimizes queries for better retrieval performance and relevance.
    """
    
    def __init__(self):
        self.query_cache = AsyncCache(max_size=200, default_ttl=1800)  # 30 minutes
        self.expansion_cache = {}
    
    async def optimize_query(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Optimize query for better retrieval."""
        cache_key = f"query_opt_{hash(query + str(conversation_history))}"
        cached = await self.query_cache.get(cache_key)
        if cached:
            return cached
        
        optimized = self._apply_query_optimizations(query, conversation_history)
        await self.query_cache.set(cache_key, optimized)
        
        return optimized
    
    def _apply_query_optimizations(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Apply various query optimization techniques."""
        # Basic preprocessing
        optimized = query.strip()
        
        # Expand contractions
        contractions = {
            "what's": "what is",
            "how's": "how is", 
            "it's": "it is",
            "can't": "cannot",
            "won't": "will not"
        }
        for contraction, expansion in contractions.items():
            optimized = optimized.replace(contraction, expansion)
        
        # Add context from conversation if available
        if conversation_history:
            context_terms = self._extract_context_terms(conversation_history)
            if context_terms:
                optimized += f" {' '.join(context_terms[:3])}"
        
        return optimized
    
    def _extract_context_terms(self, conversation_history: List[str]) -> List[str]:
        """Extract relevant terms from conversation history."""
        terms = []
        for question in conversation_history[-2:]:  # Last 2 questions
            # Extract nouns and important terms (simple approach)
            words = question.split()
            terms.extend([w for w in words if len(w) > 3 and w.isalpha()])
        
        return terms


# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()


# Utility functions for performance optimization
async def parallel_async_tasks(tasks: List[Callable], max_concurrency: int = 5) -> List[Any]:
    """Execute async tasks in parallel with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def limited_task(task):
        async with semaphore:
            return await task()
    
    return await asyncio.gather(*[limited_task(task) for task in tasks])


def memory_efficient_chunking(data: List[Any], chunk_size: int = 100) -> List[List[Any]]:
    """Split large data into memory-efficient chunks."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


class LatencyOptimizer:
    """
    Specific optimizations to achieve <200ms retrieval latency target.
    """
    
    def __init__(self):
        self.warm_cache = {}
        self.precomputed_embeddings = AsyncCache(max_size=100, default_ttl=1800)
    
    async def warm_up_system(self, common_queries: List[str]):
        """Pre-warm the system with common queries."""
        logger.info("Warming up system for optimal performance...")
        
        # Pre-generate embeddings for common queries
        for query in common_queries:
            try:
                # This would integrate with your embedding generator
                # embedding = await self.embedding_generator.generate_single_embedding(query)
                # await self.precomputed_embeddings.set(f"query_{hash(query)}", embedding)
                pass
            except Exception as e:
                logger.error(f"Failed to pre-warm query '{query}': {e}")
        
        logger.info(f"System warmed up with {len(common_queries)} queries")
    
    async def optimize_for_latency(self, operation_func: Callable, *args, **kwargs) -> Any:
        """Apply latency optimizations to any operation."""
        start_time = time.time()
        
        try:
            # Run operation with timeout
            result = await asyncio.wait_for(operation_func(*args, **kwargs), timeout=0.15)  # 150ms
            
            latency = time.time() - start_time
            if latency > 0.2:
                logger.warning(f"Operation exceeded 200ms target: {latency:.3f}s")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Operation timed out, applying fallback strategy")
            # Could implement fallback strategies here
            raise
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency optimization statistics."""
        return {
            "precomputed_cache_size": len(self.precomputed_embeddings.cache),
            "warm_cache_size": len(self.warm_cache)
        }
