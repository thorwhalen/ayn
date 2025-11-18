"""Performance optimization for agents."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


@dataclass
class PerformanceAnalysis:
    """Analysis of agent performance.

    Example:
        >>> analysis = PerformanceAnalysis(
        ...     avg_latency_ms=150.0,
        ...     p95_latency_ms=250.0,
        ...     total_calls=100
        ... )
        >>> analysis.avg_latency_ms
        150.0
    """

    avg_latency_ms: float
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_calls: int = 0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "Performance Analysis:",
            f"  Total Calls: {self.total_calls}",
            f"  Throughput: {self.throughput_rps:.2f} requests/sec",
            f"  Error Rate: {self.error_rate:.2%}",
            "\n  Latency:",
            f"    Min: {self.min_latency_ms:.2f}ms",
            f"    Avg: {self.avg_latency_ms:.2f}ms",
            f"    P50: {self.p50_latency_ms:.2f}ms",
            f"    P95: {self.p95_latency_ms:.2f}ms",
            f"    P99: {self.p99_latency_ms:.2f}ms",
            f"    Max: {self.max_latency_ms:.2f}ms",
        ]

        if self.bottlenecks:
            lines.append("\n  Bottlenecks Detected:")
            for bottleneck in self.bottlenecks:
                lines.append(f"    • {bottleneck}")

        return "\n".join(lines)


class PerformanceOptimizer:
    """Optimizes agent performance through various techniques.

    Techniques:
    - Lazy loading of heavy dependencies
    - Connection pooling
    - Request batching
    - Async execution
    - Prewarming

    Example:
        >>> optimizer = PerformanceOptimizer()
        >>> # Wrap a function for lazy loading
        >>> @optimizer.lazy_load
        ... def expensive_import():
        ...     import numpy as np
        ...     return np
    """

    def __init__(self):
        self._lazy_cache: Dict[str, Any] = {}
        self._prewarm_cache: Dict[str, Any] = {}

    def analyze(self, latencies: List[float], errors: int = 0) -> PerformanceAnalysis:
        """Analyze performance from latency data.

        Args:
            latencies: List of latency measurements in milliseconds
            errors: Number of errors encountered

        Returns:
            PerformanceAnalysis with detailed metrics
        """
        if not latencies:
            return PerformanceAnalysis(
                avg_latency_ms=0.0,
                total_calls=0,
                error_rate=0.0,
            )

        sorted_latencies = sorted(latencies)
        total_calls = len(latencies)

        # Calculate percentiles
        def percentile(data, p):
            index = int(len(data) * p)
            index = min(index, len(data) - 1)
            return data[index]

        analysis = PerformanceAnalysis(
            avg_latency_ms=sum(latencies) / len(latencies),
            p50_latency_ms=percentile(sorted_latencies, 0.50),
            p95_latency_ms=percentile(sorted_latencies, 0.95),
            p99_latency_ms=percentile(sorted_latencies, 0.99),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            total_calls=total_calls,
            error_rate=errors / total_calls if total_calls > 0 else 0.0,
        )

        # Detect bottlenecks
        if analysis.p95_latency_ms > 1000:
            analysis.bottlenecks.append("High P95 latency (>1s) - consider optimization")

        if analysis.max_latency_ms > analysis.avg_latency_ms * 5:
            analysis.bottlenecks.append("High variance in latency - inconsistent performance")

        if analysis.error_rate > 0.05:
            analysis.bottlenecks.append(f"High error rate ({analysis.error_rate:.1%})")

        return analysis

    def lazy_load(self, func: Callable) -> Callable:
        """Decorator for lazy loading expensive resources.

        Args:
            func: Function that returns the resource

        Returns:
            Wrapper that caches the result

        Example:
            >>> optimizer = PerformanceOptimizer()
            >>> @optimizer.lazy_load
            ... def load_model():
            ...     # Expensive model loading
            ...     return "model"
            >>> model = load_model()  # Loads on first call
            >>> model2 = load_model()  # Returns cached version
        """
        cache_key = f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if cache_key not in self._lazy_cache:
                self._lazy_cache[cache_key] = func(*args, **kwargs)
            return self._lazy_cache[cache_key]

        return wrapper

    def prewarm(self, controller: BaseAgentController, test_input: Any):
        """Prewarm agent to reduce first-call latency.

        Args:
            controller: Agent controller to prewarm
            test_input: Test input for warming up
        """
        try:
            # Execute a test call to warm up
            controller.invoke(test_input)
            self._prewarm_cache[controller.__class__.__name__] = True
        except Exception:
            pass  # Ignore errors during prewarming

    def batch_requests(
        self,
        requests: List[Any],
        batch_size: int = 10,
    ) -> List[List[Any]]:
        """Batch requests for efficient processing.

        Args:
            requests: List of requests to batch
            batch_size: Maximum batch size

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(requests), batch_size):
            batches.append(requests[i : i + batch_size])
        return batches

    def recommend_optimizations(
        self,
        analysis: PerformanceAnalysis,
    ) -> List[str]:
        """Recommend performance optimizations.

        Args:
            analysis: PerformanceAnalysis to analyze

        Returns:
            List of recommendations
        """
        recommendations = []

        # High latency
        if analysis.avg_latency_ms > 500:
            recommendations.append(
                "⚡ High average latency detected (>500ms):\n"
                "  - Enable caching for repeated requests\n"
                "  - Use async/await for I/O operations\n"
                "  - Consider model compression or quantization"
            )

        # High P95
        if analysis.p95_latency_ms > analysis.avg_latency_ms * 2:
            recommendations.append(
                "📊 High P95 latency (tail latency):\n"
                "  - Implement request timeout and retry logic\n"
                "  - Add request prioritization\n"
                "  - Consider auto-scaling for peak loads"
            )

        # Low throughput
        if analysis.throughput_rps < 10 and analysis.total_calls > 100:
            recommendations.append(
                "🚀 Low throughput detected:\n"
                "  - Enable parallel processing\n"
                "  - Implement request batching\n"
                "  - Use connection pooling"
            )

        # High error rate
        if analysis.error_rate > 0.01:
            recommendations.append(
                "⚠️ High error rate:\n"
                "  - Add circuit breakers to prevent cascade failures\n"
                "  - Implement exponential backoff for retries\n"
                "  - Add input validation to catch errors early"
            )

        # Variance
        if analysis.max_latency_ms > analysis.avg_latency_ms * 10:
            recommendations.append(
                "📈 High latency variance:\n"
                "  - Implement health checks and failover\n"
                "  - Add monitoring and alerting\n"
                "  - Consider load balancing across multiple instances"
            )

        if not recommendations:
            recommendations.append("✅ Performance looks good! No major issues detected.")

        return recommendations


def lazy_property(func: Callable) -> property:
    """Decorator for lazy-loaded properties.

    Example:
        >>> class MyClass:
        ...     @lazy_property
        ...     def expensive_property(self):
        ...         return "expensive computation"
        >>> obj = MyClass()
        >>> # Property computed on first access, then cached
        >>> obj.expensive_property
        'expensive computation'
    """
    attr_name = f"_lazy_{func.__name__}"

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return wrapper
