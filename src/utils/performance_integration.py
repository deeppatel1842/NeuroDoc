"""
Performance Integration Module

This module provides easy integration of performance monitoring throughout
the NeuroDoc system with decorators and context managers.
"""

import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager

from .performance import PerformanceMonitor

logger = logging.getLogger(__name__)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str, include_args: bool = False):
    """
    Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation for tracking
        include_args: Whether to include function arguments in metadata
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            metadata = {}
            
            if include_args:
                metadata.update({
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
            
            try:
                result = await func(*args, **kwargs)
                metadata["success"] = True
                return result
            except Exception as e:
                metadata["success"] = False
                metadata["error"] = str(e)
                raise
            finally:
                duration = time.time() - start_time
                performance_monitor.record_metric(operation_name, duration, metadata)
                
                # Log if operation exceeds target
                if operation_name in performance_monitor.targets:
                    target = performance_monitor.targets[operation_name]
                    if duration > target:
                        logger.warning(
                            f"Performance target exceeded for {operation_name}: "
                            f"{duration:.3f}s > {target:.3f}s"
                        )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            metadata = {}
            
            if include_args:
                metadata.update({
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
            
            try:
                result = func(*args, **kwargs)
                metadata["success"] = True
                return result
            except Exception as e:
                metadata["success"] = False
                metadata["error"] = str(e)
                raise
            finally:
                duration = time.time() - start_time
                performance_monitor.record_metric(operation_name, duration, metadata)
                
                # Log if operation exceeds target
                if operation_name in performance_monitor.targets:
                    target = performance_monitor.targets[operation_name]
                    if duration > target:
                        logger.warning(
                            f"Performance target exceeded for {operation_name}: "
                            f"{duration:.3f}s > {target:.3f}s"
                        )
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def performance_context(operation_name: str, metadata: Optional[Dict] = None):
    """
    Context manager for monitoring performance of code blocks.
    
    Args:
        operation_name: Name of the operation for tracking
        metadata: Additional metadata to record
    """
    start_time = time.time()
    context_metadata = metadata or {}
    
    try:
        yield performance_monitor
        context_metadata["success"] = True
    except Exception as e:
        context_metadata["success"] = False
        context_metadata["error"] = str(e)
        raise
    finally:
        duration = time.time() - start_time
        performance_monitor.record_metric(operation_name, duration, context_metadata)


def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of current performance metrics."""
    return performance_monitor.get_summary()


def optimize_batch_size(operation_name: str, max_size: int = 100) -> int:
    """
    Suggest optimal batch size based on historical performance.
    
    Args:
        operation_name: Name of the operation to optimize
        max_size: Maximum allowed batch size
        
    Returns:
        Suggested optimal batch size
    """
    if operation_name not in performance_monitor.metrics:
        return min(10, max_size)  # Conservative default
    
    # Analyze recent performance data
    recent_metrics = list(performance_monitor.metrics[operation_name])[-20:]
    
    if not recent_metrics:
        return min(10, max_size)
    
    # Calculate average duration per item (if metadata includes batch info)
    durations_with_size = []
    for metric in recent_metrics:
        duration = metric["duration"]
        metadata = metric.get("metadata", {})
        batch_size = metadata.get("batch_size", 1)
        
        if batch_size > 0:
            durations_with_size.append((duration / batch_size, batch_size))
    
    if not durations_with_size:
        # Fallback: use average duration to estimate
        avg_duration = sum(m["duration"] for m in recent_metrics) / len(recent_metrics)
        target = performance_monitor.targets.get(operation_name, 0.2)
        
        # Conservative estimate
        suggested_size = max(1, int(target / max(avg_duration, 0.001)))
        return min(suggested_size, max_size)
    
    # Find batch size with best per-item performance
    durations_with_size.sort(key=lambda x: x[0])  # Sort by per-item duration
    best_per_item_duration = durations_with_size[0][0]
    
    # Calculate optimal size based on target latency
    target = performance_monitor.targets.get(operation_name, 0.2)
    optimal_size = int(target / best_per_item_duration)
    
    return min(max(1, optimal_size), max_size)


class PerformanceOptimizer:
    """
    Adaptive performance optimizer that adjusts system parameters
    based on observed performance patterns.
    """
    
    def __init__(self):
        self.optimization_history = {}
        self.adaptive_parameters = {
            "chunk_size": {"min": 100, "max": 2000, "current": 500},
            "overlap_size": {"min": 20, "max": 200, "current": 50},
            "batch_size": {"min": 1, "max": 50, "current": 10},
            "top_k": {"min": 5, "max": 50, "current": 20},
        }
    
    def suggest_optimizations(self) -> Dict[str, Any]:
        """
        Suggest system optimizations based on performance data.
        
        Returns:
            Dictionary of suggested parameter adjustments
        """
        suggestions = {}
        summary = performance_monitor.get_summary()
        
        for operation, stats in summary.items():
            if operation in performance_monitor.targets:
                target = performance_monitor.targets[operation]
                avg_duration = stats.get("avg_duration", 0)
                
                if avg_duration > target * 1.2:  # 20% over target
                    suggestions[operation] = {
                        "status": "needs_optimization",
                        "current_avg": avg_duration,
                        "target": target,
                        "suggestions": self._get_operation_suggestions(operation, stats)
                    }
                elif avg_duration < target * 0.8:  # 20% under target
                    suggestions[operation] = {
                        "status": "can_increase_quality",
                        "current_avg": avg_duration,
                        "target": target,
                        "suggestions": self._get_quality_suggestions(operation, stats)
                    }
                else:
                    suggestions[operation] = {
                        "status": "optimal",
                        "current_avg": avg_duration,
                        "target": target
                    }
        
        return suggestions
    
    def _get_operation_suggestions(self, operation: str, stats: Dict) -> List[str]:
        """Get specific suggestions for optimizing an operation."""
        suggestions = []
        
        if "retrieval" in operation.lower():
            suggestions.extend([
                "Consider reducing top_k parameter",
                "Enable result caching",
                "Optimize chunk size for faster processing",
                "Use performance reranker instead of cross-encoder"
            ])
        elif "embedding" in operation.lower():
            suggestions.extend([
                "Increase batch size for embedding generation",
                "Consider model quantization",
                "Enable embedding caching",
                "Use smaller embedding model if accuracy allows"
            ])
        elif "processing" in operation.lower():
            suggestions.extend([
                "Optimize chunk size",
                "Enable parallel processing",
                "Reduce text cleaning operations",
                "Use faster PDF extraction method"
            ])
        
        return suggestions
    
    def _get_quality_suggestions(self, operation: str, stats: Dict) -> List[str]:
        """Get suggestions for improving quality when performance allows."""
        suggestions = []
        
        if "retrieval" in operation.lower():
            suggestions.extend([
                "Increase top_k parameter for better recall",
                "Enable cross-encoder reranking",
                "Use more sophisticated reranking strategies",
                "Increase context window size"
            ])
        elif "embedding" in operation.lower():
            suggestions.extend([
                "Use larger, more accurate embedding model",
                "Increase embedding dimensions",
                "Enable ensemble embedding approaches"
            ])
        
        return suggestions


# Global optimizer instance
performance_optimizer = PerformanceOptimizer()
