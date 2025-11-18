"""Metrics collection for agent performance tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values


@dataclass
class Metric:
    """A single metric measurement."""

    name: str
    value: float
    type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics for agents.

    Tracks:
    - Invocation counts
    - Latencies (avg, p50, p95, p99)
    - Error rates
    - Token usage
    - Costs

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_invocation(latency_ms=150, tokens=100, cost=0.002)
        >>> collector.record_invocation(latency_ms=200, tokens=120, cost=0.003)
        >>> stats = collector.get_stats()
        >>> stats['total_invocations']
        2
        >>> stats['avg_latency_ms']
        175.0
    """

    def __init__(self, agent_name: Optional[str] = None):
        self.agent_name = agent_name
        self.metrics: List[Metric] = []

        # Aggregated stats
        self._invocation_count = 0
        self._error_count = 0
        self._latencies: List[float] = []
        self._tokens: List[int] = []
        self._costs: List[float] = []

    def record_invocation(
        self,
        latency_ms: float,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        success: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Record an agent invocation.

        Args:
            latency_ms: Invocation latency in milliseconds
            tokens: Number of tokens used
            cost: Cost of invocation
            success: Whether invocation succeeded
            tags: Additional tags for the metric
        """
        self._invocation_count += 1

        if not success:
            self._error_count += 1

        # Record latency
        self._latencies.append(latency_ms)
        self.metrics.append(
            Metric(
                name="invocation_latency_ms",
                value=latency_ms,
                type=MetricType.HISTOGRAM,
                tags=tags or {},
            )
        )

        # Record tokens if provided
        if tokens is not None:
            self._tokens.append(tokens)
            self.metrics.append(
                Metric(
                    name="tokens_used",
                    value=tokens,
                    type=MetricType.HISTOGRAM,
                    tags=tags or {},
                )
            )

        # Record cost if provided
        if cost is not None:
            self._costs.append(cost)
            self.metrics.append(
                Metric(
                    name="invocation_cost",
                    value=cost,
                    type=MetricType.HISTOGRAM,
                    tags=tags or {},
                )
            )

        # Record success/failure
        self.metrics.append(
            Metric(
                name="invocation_count",
                value=1,
                type=MetricType.COUNTER,
                tags={**(tags or {}), "success": str(success)},
            )
        )

    def get_stats(self) -> Dict[str, any]:
        """Get aggregated statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_invocations": self._invocation_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._invocation_count
                if self._invocation_count > 0
                else 0.0
            ),
        }

        # Latency stats
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            stats.update(
                {
                    "avg_latency_ms": sum(self._latencies) / len(self._latencies),
                    "min_latency_ms": min(self._latencies),
                    "max_latency_ms": max(self._latencies),
                    "p50_latency_ms": self._percentile(sorted_latencies, 0.50),
                    "p95_latency_ms": self._percentile(sorted_latencies, 0.95),
                    "p99_latency_ms": self._percentile(sorted_latencies, 0.99),
                }
            )

        # Token stats
        if self._tokens:
            stats.update(
                {
                    "total_tokens": sum(self._tokens),
                    "avg_tokens": sum(self._tokens) / len(self._tokens),
                    "min_tokens": min(self._tokens),
                    "max_tokens": max(self._tokens),
                }
            )

        # Cost stats
        if self._costs:
            stats.update(
                {
                    "total_cost": sum(self._costs),
                    "avg_cost": sum(self._costs) / len(self._costs),
                    "min_cost": min(self._costs),
                    "max_cost": max(self._costs),
                }
            )

        return stats

    def reset(self):
        """Reset all metrics and stats."""
        self.metrics.clear()
        self._invocation_count = 0
        self._error_count = 0
        self._latencies.clear()
        self._tokens.clear()
        self._costs.clear()

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values.

        Args:
            sorted_values: Sorted list of values
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0

        index = int(len(sorted_values) * percentile)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def __str__(self) -> str:
        """String representation of metrics."""
        stats = self.get_stats()
        lines = []

        if self.agent_name:
            lines.append(f"Metrics for: {self.agent_name}")

        lines.append(f"Total Invocations: {stats['total_invocations']}")
        lines.append(f"Error Rate: {stats['error_rate']:.2%}")

        if "avg_latency_ms" in stats:
            lines.append(f"\nLatency:")
            lines.append(f"  Avg: {stats['avg_latency_ms']:.2f}ms")
            lines.append(f"  P50: {stats['p50_latency_ms']:.2f}ms")
            lines.append(f"  P95: {stats['p95_latency_ms']:.2f}ms")
            lines.append(f"  P99: {stats['p99_latency_ms']:.2f}ms")

        if "total_tokens" in stats:
            lines.append(f"\nTokens:")
            lines.append(f"  Total: {stats['total_tokens']}")
            lines.append(f"  Avg: {stats['avg_tokens']:.1f}")

        if "total_cost" in stats:
            lines.append(f"\nCost:")
            lines.append(f"  Total: ${stats['total_cost']:.4f}")
            lines.append(f"  Avg: ${stats['avg_cost']:.4f}")

        return "\n".join(lines)
