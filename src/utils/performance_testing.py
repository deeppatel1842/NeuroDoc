"""
Performance Testing and Benchmarking Module

This module provides comprehensive performance testing capabilities
to validate that the system meets the <200ms latency targets.
"""

import logging
import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for the NeuroDoc system.
    """
    
    def __init__(self):
        # self.latency_optimizer = LatencyOptimizer()  # Disabled for now
        pass
        self.benchmark_results = {}
        self.test_queries = [
            "What is the main topic of this document?",
            "Can you summarize the key findings?",
            "What are the conclusions mentioned?",
            "Tell me about the methodology used.",
            "What are the recommendations?",
            "How does this relate to previous work?",
            "What are the limitations discussed?",
            "What future work is suggested?",
            "Can you explain the results section?",
            "What is the significance of this research?"
        ]
    
    async def run_comprehensive_benchmark(
        self, 
        retriever, 
        session_id: str,
        num_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark.
        
        Args:
            retriever: Retrieval system to test
            session_id: Test session ID
            num_iterations: Number of test iterations per query
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting comprehensive benchmark with {num_iterations} iterations")
        
        benchmark_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "num_iterations": num_iterations,
                "session_id": session_id,
                "test_queries": len(self.test_queries)
            },
            "results": {}
        }
        
        # Test 1: Standard Retrieval Performance
        standard_results = await self._benchmark_standard_retrieval(
            retriever, session_id, num_iterations
        )
        benchmark_results["results"]["standard_retrieval"] = standard_results
        
        # Test 2: Optimized Retrieval Performance
        optimized_results = await self._benchmark_optimized_retrieval(
            retriever, session_id, num_iterations
        )
        benchmark_results["results"]["optimized_retrieval"] = optimized_results
        
        # Test 3: Concurrent Load Testing
        concurrent_results = await self._benchmark_concurrent_load(
            retriever, session_id, concurrency_levels=[1, 5, 10, 20]
        )
        benchmark_results["results"]["concurrent_load"] = concurrent_results
        
        # Test 4: Cache Performance Testing
        cache_results = await self._benchmark_cache_performance(
            retriever, session_id, num_iterations
        )
        benchmark_results["results"]["cache_performance"] = cache_results
        
        # Test 5: Scalability Testing
        scalability_results = await self._benchmark_scalability(
            retriever, session_id
        )
        benchmark_results["results"]["scalability"] = scalability_results
        
        # Generate performance summary
        summary = self._generate_performance_summary(benchmark_results)
        benchmark_results["summary"] = summary
        
        # Save results
        await self._save_benchmark_results(benchmark_results)
        
        logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    async def _benchmark_standard_retrieval(
        self, retriever, session_id: str, num_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark standard retrieval performance."""
        logger.info("Benchmarking standard retrieval performance")
        
        latencies = []
        errors = 0
        
        for i in range(num_iterations):
            query = self.test_queries[i % len(self.test_queries)]
            
            try:
                start_time = time.time()
                results = await retriever.retrieve(query, session_id, top_k=10)
                latency = time.time() - start_time
                
                latencies.append(latency)
                
                if i % 10 == 0:
                    logger.debug(f"Standard retrieval iteration {i}: {latency:.3f}s")
                    
            except Exception as e:
                logger.error(f"Standard retrieval error in iteration {i}: {e}")
                errors += 1
        
        return self._calculate_latency_stats(latencies, errors, num_iterations)
    
    async def _benchmark_optimized_retrieval(
        self, retriever, session_id: str, num_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark optimized retrieval performance."""
        logger.info("Benchmarking optimized retrieval performance")
        
        latencies = []
        errors = 0
        optimization_stats = []
        
        for i in range(num_iterations):
            query = self.test_queries[i % len(self.test_queries)]
            
            try:
                start_time = time.time()
                results, opt_metadata = await self.latency_optimizer.optimize_retrieval_pipeline(
                    retriever, query, session_id, top_k=10
                )
                latency = time.time() - start_time
                
                latencies.append(latency)
                optimization_stats.append(opt_metadata)
                
                if i % 10 == 0:
                    logger.debug(f"Optimized retrieval iteration {i}: {latency:.3f}s")
                    
            except Exception as e:
                logger.error(f"Optimized retrieval error in iteration {i}: {e}")
                errors += 1
        
        stats = self._calculate_latency_stats(latencies, errors, num_iterations)
        stats["optimization_stats"] = self._analyze_optimization_stats(optimization_stats)
        
        return stats
    
    async def _benchmark_concurrent_load(
        self, retriever, session_id: str, concurrency_levels: List[int]
    ) -> Dict[str, Any]:
        """Benchmark performance under concurrent load."""
        logger.info("Benchmarking concurrent load performance")
        
        concurrent_results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            async def single_request():
                query = self.test_queries[0]  # Use same query for consistency
                start_time = time.time()
                try:
                    results = await retriever.retrieve(query, session_id, top_k=10)
                    return time.time() - start_time, True
                except Exception as e:
                    logger.error(f"Concurrent request failed: {e}")
                    return time.time() - start_time, False
            
            # Create concurrent tasks
            tasks = [single_request() for _ in range(concurrency)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Analyze results
            latencies = []
            errors = 0
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                else:
                    latency, success = result
                    if success:
                        latencies.append(latency)
                    else:
                        errors += 1
            
            concurrent_results[f"concurrency_{concurrency}"] = {
                "latencies": self._calculate_latency_stats(latencies, errors, concurrency),
                "total_time": total_time,
                "throughput": concurrency / total_time if total_time > 0 else 0
            }
        
        return concurrent_results
    
    async def _benchmark_cache_performance(
        self, retriever, session_id: str, num_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark cache performance impact."""
        logger.info("Benchmarking cache performance")
        
        # Test with same query repeated (should hit cache)
        repeated_query = self.test_queries[0]
        cache_latencies = []
        
        for i in range(num_iterations):
            start_time = time.time()
            try:
                results, opt_metadata = await self.latency_optimizer.optimize_retrieval_pipeline(
                    retriever, repeated_query, session_id, top_k=10
                )
                latency = time.time() - start_time
                cache_latencies.append((latency, opt_metadata.get("cache_hit", False)))
            except Exception as e:
                logger.error(f"Cache test error: {e}")
        
        # Separate cache hits and misses
        cache_hits = [lat for lat, hit in cache_latencies if hit]
        cache_misses = [lat for lat, hit in cache_latencies if not hit]
        
        return {
            "cache_hits": {
                "count": len(cache_hits),
                "latencies": self._calculate_basic_stats(cache_hits) if cache_hits else {}
            },
            "cache_misses": {
                "count": len(cache_misses),
                "latencies": self._calculate_basic_stats(cache_misses) if cache_misses else {}
            },
            "hit_rate": len(cache_hits) / len(cache_latencies) if cache_latencies else 0
        }
    
    async def _benchmark_scalability(self, retriever, session_id: str) -> Dict[str, Any]:
        """Test scalability with different result set sizes."""
        logger.info("Benchmarking scalability")
        
        top_k_values = [5, 10, 20, 50]
        scalability_results = {}
        
        for top_k in top_k_values:
            latencies = []
            
            for query in self.test_queries[:5]:  # Use first 5 queries
                try:
                    start_time = time.time()
                    results = await retriever.retrieve(query, session_id, top_k=top_k)
                    latency = time.time() - start_time
                    latencies.append(latency)
                except Exception as e:
                    logger.error(f"Scalability test error for top_k={top_k}: {e}")
            
            scalability_results[f"top_k_{top_k}"] = self._calculate_basic_stats(latencies)
        
        return scalability_results
    
    def _calculate_latency_stats(
        self, latencies: List[float], errors: int, total_requests: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive latency statistics."""
        if not latencies:
            return {
                "error_rate": errors / total_requests if total_requests > 0 else 0,
                "success_rate": 0,
                "message": "No successful requests"
            }
        
        stats = self._calculate_basic_stats(latencies)
        stats.update({
            "error_rate": errors / total_requests if total_requests > 0 else 0,
            "success_rate": len(latencies) / total_requests if total_requests > 0 else 0,
            "target_compliance": {
                "under_200ms": sum(1 for lat in latencies if lat < 0.2) / len(latencies),
                "under_500ms": sum(1 for lat in latencies if lat < 0.5) / len(latencies),
                "under_1s": sum(1 for lat in latencies if lat < 1.0) / len(latencies)
            }
        })
        
        return stats
    
    def _calculate_basic_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        if not latencies:
            return {}
        
        return {
            "count": len(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min": min(latencies),
            "max": max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
    
    def _analyze_optimization_stats(self, optimization_stats: List[Dict]) -> Dict[str, Any]:
        """Analyze optimization strategy effectiveness."""
        if not optimization_stats:
            return {}
        
        strategy_usage = {}
        cache_hit_rate = 0
        timeout_rate = 0
        
        for stat in optimization_stats:
            # Count strategy usage
            for strategy in stat.get("strategies_used", []):
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Track cache hits and timeouts
            if stat.get("cache_hit", False):
                cache_hit_rate += 1
            if stat.get("timeout_occurred", False):
                timeout_rate += 1
        
        total_requests = len(optimization_stats)
        
        return {
            "strategy_usage": {k: v / total_requests for k, v in strategy_usage.items()},
            "cache_hit_rate": cache_hit_rate / total_requests,
            "timeout_rate": timeout_rate / total_requests
        }
    
    def _generate_performance_summary(self, benchmark_results: Dict) -> Dict[str, Any]:
        """Generate overall performance summary."""
        results = benchmark_results["results"]
        
        summary = {
            "overall_status": "unknown",
            "key_metrics": {},
            "recommendations": [],
            "compliance": {}
        }
        
        # Analyze standard vs optimized performance
        standard = results.get("standard_retrieval", {})
        optimized = results.get("optimized_retrieval", {})
        
        if standard and optimized:
            summary["key_metrics"] = {
                "standard_p95": standard.get("p95", 0),
                "optimized_p95": optimized.get("p95", 0),
                "improvement": (standard.get("p95", 0) - optimized.get("p95", 0)) * 1000,  # ms
                "target_compliance_standard": standard.get("target_compliance", {}).get("under_200ms", 0),
                "target_compliance_optimized": optimized.get("target_compliance", {}).get("under_200ms", 0)
            }
            
            # Determine overall status
            opt_p95 = optimized.get("p95", float('inf'))
            opt_compliance = optimized.get("target_compliance", {}).get("under_200ms", 0)
            
            if opt_p95 < 0.2 and opt_compliance > 0.95:
                summary["overall_status"] = "excellent"
            elif opt_p95 < 0.3 and opt_compliance > 0.90:
                summary["overall_status"] = "good"
            elif opt_p95 < 0.5 and opt_compliance > 0.80:
                summary["overall_status"] = "acceptable"
            else:
                summary["overall_status"] = "needs_improvement"
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(results)
        
        return summary
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        optimized = results.get("optimized_retrieval", {})
        cache_perf = results.get("cache_performance", {})
        concurrent = results.get("concurrent_load", {})
        
        # Check latency compliance
        if optimized.get("p95", float('inf')) > 0.2:
            recommendations.append(
                "P95 latency exceeds 200ms target. Consider enabling more aggressive caching."
            )
        
        # Check cache effectiveness
        hit_rate = cache_perf.get("hit_rate", 0)
        if hit_rate < 0.3:
            recommendations.append(
                "Cache hit rate is low. Consider increasing cache size or TTL."
            )
        
        # Check concurrent performance
        for level, data in concurrent.items():
            if "concurrency_10" in level:
                throughput = data.get("throughput", 0)
                if throughput < 5:  # Less than 5 requests/second at concurrency 10
                    recommendations.append(
                        "Low throughput under concurrent load. Consider increasing worker threads."
                    )
        
        # Check error rates
        error_rate = optimized.get("error_rate", 0)
        if error_rate > 0.01:  # More than 1% errors
            recommendations.append(
                "High error rate detected. Check system stability and error handling."
            )
        
        if not recommendations:
            recommendations.append("Performance is within acceptable targets. Consider minor optimizations for further improvements.")
        
        return recommendations
    
    async def _save_benchmark_results(self, results: Dict):
        """Save benchmark results to file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
            # Ensure benchmark directory exists
            benchmark_dir = Path("data/benchmarks")
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = benchmark_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")


class RealTimePerformanceMonitor:
    """Real-time performance monitoring for production systems."""
    
    def __init__(self, alert_threshold: float = 0.25):
        self.alert_threshold = alert_threshold
        self.recent_latencies = []
        self.alerts_sent = []
        
    async def monitor_request(self, latency: float, metadata: Dict = None):
        """Monitor individual request performance."""
        self.recent_latencies.append({
            "latency": latency,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        })
        
        # Keep only recent data (last 100 requests)
        if len(self.recent_latencies) > 100:
            self.recent_latencies.pop(0)
        
        # Check for performance degradation
        if latency > self.alert_threshold:
            await self._handle_performance_alert(latency, metadata)
    
    async def _handle_performance_alert(self, latency: float, metadata: Dict):
        """Handle performance alerts."""
        alert = {
            "timestamp": datetime.utcnow(),
            "latency": latency,
            "threshold": self.alert_threshold,
            "metadata": metadata
        }
        
        self.alerts_sent.append(alert)
        logger.warning(f"Performance alert: {latency:.3f}s > {self.alert_threshold:.3f}s")
        
        # Here you could integrate with monitoring systems like:
        # - Send to Prometheus/Grafana
        # - Send Slack notifications
        # - Trigger auto-scaling
        # - etc.
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.recent_latencies:
            return {}
        
        latencies = [req["latency"] for req in self.recent_latencies]
        
        return {
            "current_p95": np.percentile(latencies, 95),
            "current_mean": np.mean(latencies),
            "request_count": len(latencies),
            "alert_count": len(self.alerts_sent),
            "compliance_rate": sum(1 for lat in latencies if lat < 0.2) / len(latencies)
        }
