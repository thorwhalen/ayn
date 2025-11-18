"""Cost optimization for agent operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


# Model pricing (per 1M tokens) - can be updated
MODEL_PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class CostAnalysis:
    """Analysis of agent costs.

    Example:
        >>> analysis = CostAnalysis(
        ...     total_cost=1.50,
        ...     avg_cost_per_call=0.015,
        ...     total_calls=100
        ... )
        >>> analysis.total_cost
        1.5
    """

    total_cost: float
    avg_cost_per_call: float
    total_calls: int
    total_tokens: int = 0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    projections: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "Cost Analysis:",
            f"  Total Cost: ${self.total_cost:.4f}",
            f"  Average per Call: ${self.avg_cost_per_call:.4f}",
            f"  Total Calls: {self.total_calls}",
        ]

        if self.total_tokens:
            lines.append(f"  Total Tokens: {self.total_tokens:,}")
            cost_per_1k = (self.total_cost / self.total_tokens) * 1000
            lines.append(f"  Cost per 1K tokens: ${cost_per_1k:.4f}")

        if self.cost_breakdown:
            lines.append("\n  Cost Breakdown:")
            for item, cost in self.cost_breakdown.items():
                percentage = (cost / self.total_cost * 100) if self.total_cost > 0 else 0
                lines.append(f"    {item}: ${cost:.4f} ({percentage:.1f}%)")

        if self.projections:
            lines.append("\n  Projections:")
            for period, cost in self.projections.items():
                lines.append(f"    {period}: ${cost:.2f}")

        return "\n".join(lines)


@dataclass
class OptimizationRecommendation:
    """Recommendation for cost optimization.

    Example:
        >>> rec = OptimizationRecommendation(
        ...     recommendation="Use cheaper model",
        ...     potential_savings=0.50,
        ...     impact="Low impact on quality"
        ... )
        >>> rec.potential_savings
        0.5
    """

    recommendation: str
    potential_savings: float
    impact: str
    confidence: float = 0.8  # 0-1
    implementation_effort: str = "Low"  # Low, Medium, High

    def __str__(self) -> str:
        return (
            f"💡 {self.recommendation}\n"
            f"   Savings: ${self.potential_savings:.4f}\n"
            f"   Impact: {self.impact}\n"
            f"   Effort: {self.implementation_effort}\n"
            f"   Confidence: {self.confidence:.0%}"
        )


class CostOptimizer:
    """Optimizes agent costs through various strategies.

    Strategies:
    - Model selection (cheaper models for simpler tasks)
    - Caching (reduce redundant calls)
    - Batching (combine requests)
    - Token optimization (shorter prompts)
    - Provider selection (route to cheaper providers)

    Example:
        >>> from ayn.core import ControllerConfig
        >>> config = ControllerConfig()
        >>> optimizer = CostOptimizer(config)
        >>> # analysis = optimizer.analyze(call_history)
        >>> # recommendations = optimizer.recommend(analysis)
    """

    def __init__(self, config: Optional[any] = None):
        """Initialize cost optimizer.

        Args:
            config: Optional configuration with pricing info
        """
        self.config = config
        self.pricing = MODEL_PRICING.copy()

    def analyze(
        self,
        calls: List[Dict],
        model: str = "gpt-4-turbo",
    ) -> CostAnalysis:
        """Analyze costs from call history.

        Args:
            calls: List of call records with tokens/costs
            model: Model name for pricing

        Returns:
            CostAnalysis with detailed breakdown
        """
        total_cost = 0.0
        total_tokens = 0
        cost_breakdown = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "api_overhead": 0.0,
        }

        for call in calls:
            # Extract token usage
            input_tokens = call.get("input_tokens", 0)
            output_tokens = call.get("output_tokens", 0)

            # Calculate costs
            if model in self.pricing:
                pricing = self.pricing[model]
                input_cost = (input_tokens / 1_000_000) * pricing["input"]
                output_cost = (output_tokens / 1_000_000) * pricing["output"]

                total_cost += input_cost + output_cost
                total_tokens += input_tokens + output_tokens

                cost_breakdown["input_tokens"] += input_cost
                cost_breakdown["output_tokens"] += output_cost
            else:
                # Use provided cost if available
                total_cost += call.get("cost", 0)
                total_tokens += input_tokens + output_tokens

        total_calls = len(calls)
        avg_cost = total_cost / total_calls if total_calls > 0 else 0.0

        # Project costs
        projections = {
            "Daily (100 calls)": avg_cost * 100,
            "Weekly (700 calls)": avg_cost * 700,
            "Monthly (3000 calls)": avg_cost * 3000,
            "Yearly (36500 calls)": avg_cost * 36500,
        }

        return CostAnalysis(
            total_cost=total_cost,
            avg_cost_per_call=avg_cost,
            total_calls=total_calls,
            total_tokens=total_tokens,
            cost_breakdown=cost_breakdown,
            projections=projections,
        )

    def recommend(
        self,
        analysis: CostAnalysis,
        current_model: str = "gpt-4-turbo",
    ) -> List[OptimizationRecommendation]:
        """Generate cost optimization recommendations.

        Args:
            analysis: CostAnalysis from analyze()
            current_model: Currently used model

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # 1. Model downgrade recommendation
        if current_model in self.pricing:
            cheaper_models = self._find_cheaper_models(current_model)
            if cheaper_models:
                model_name, savings_pct = cheaper_models[0]
                potential_savings = analysis.total_cost * savings_pct

                recommendations.append(
                    OptimizationRecommendation(
                        recommendation=f"Switch to {model_name} for cost savings",
                        potential_savings=potential_savings,
                        impact="May reduce quality for complex tasks",
                        confidence=0.7,
                        implementation_effort="Low",
                    )
                )

        # 2. Caching recommendation
        if analysis.total_calls > 50:
            # Assume 20% cache hit rate
            cache_savings = analysis.total_cost * 0.20

            recommendations.append(
                OptimizationRecommendation(
                    recommendation="Implement semantic caching to reduce redundant calls",
                    potential_savings=cache_savings,
                    impact="No impact on quality",
                    confidence=0.8,
                    implementation_effort="Low",
                )
            )

        # 3. Token optimization
        if analysis.total_tokens > 1_000_000:
            # Assume 15% reduction through prompt optimization
            token_savings = analysis.total_cost * 0.15

            recommendations.append(
                OptimizationRecommendation(
                    recommendation="Optimize prompts to reduce token usage",
                    potential_savings=token_savings,
                    impact="Minimal if done carefully",
                    confidence=0.6,
                    implementation_effort="Medium",
                )
            )

        # 4. Batching recommendation
        if analysis.avg_cost_per_call < 0.001:
            # Small calls benefit from batching
            batch_savings = analysis.total_cost * 0.10

            recommendations.append(
                OptimizationRecommendation(
                    recommendation="Batch similar requests together",
                    potential_savings=batch_savings,
                    impact="Added latency for batched requests",
                    confidence=0.5,
                    implementation_effort="Medium",
                )
            )

        # 5. Request filtering
        if analysis.total_calls > 100:
            filter_savings = analysis.total_cost * 0.05

            recommendations.append(
                OptimizationRecommendation(
                    recommendation="Add input validation to filter unnecessary calls",
                    potential_savings=filter_savings,
                    impact="Better user experience",
                    confidence=0.7,
                    implementation_effort="Low",
                )
            )

        # Sort by potential savings
        recommendations.sort(key=lambda r: r.potential_savings, reverse=True)

        return recommendations

    def _find_cheaper_models(
        self,
        current_model: str,
    ) -> List[tuple[str, float]]:
        """Find cheaper alternative models.

        Args:
            current_model: Current model name

        Returns:
            List of (model_name, savings_percentage) tuples
        """
        if current_model not in self.pricing:
            return []

        current_pricing = self.pricing[current_model]
        current_avg_cost = (current_pricing["input"] + current_pricing["output"]) / 2

        cheaper = []
        for model, pricing in self.pricing.items():
            if model == current_model:
                continue

            avg_cost = (pricing["input"] + pricing["output"]) / 2
            if avg_cost < current_avg_cost:
                savings_pct = (current_avg_cost - avg_cost) / current_avg_cost
                cheaper.append((model, savings_pct))

        # Sort by savings
        cheaper.sort(key=lambda x: x[1], reverse=True)

        return cheaper

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4-turbo",
    ) -> float:
        """Estimate cost for a given token count.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            Estimated cost in dollars
        """
        if model not in self.pricing:
            return 0.0

        pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def add_custom_pricing(self, model: str, input_price: float, output_price: float):
        """Add custom model pricing.

        Args:
            model: Model name
            input_price: Price per 1M input tokens
            output_price: Price per 1M output tokens
        """
        self.pricing[model] = {
            "input": input_price,
            "output": output_price,
        }
