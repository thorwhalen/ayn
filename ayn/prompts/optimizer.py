"""Prompt optimization based on performance feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import PromptTemplate, PromptVersion


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    recommended_version: str
    confidence: float  # 0-1
    reason: str
    alternative_versions: List[str]


class PromptOptimizer:
    """Optimizes prompt selection based on performance data.

    Analyzes performance metrics to recommend the best prompt version.

    Example:
        >>> from ayn.prompts import PromptTemplate, PromptVersion
        >>> template = PromptTemplate(name="test")
        >>> v1 = PromptVersion(version="1.0", template="Template 1")
        >>> v1.performance_metrics = {"accuracy": 0.85, "latency_ms": 200}
        >>> v2 = PromptVersion(version="2.0", template="Template 2")
        >>> v2.performance_metrics = {"accuracy": 0.90, "latency_ms": 250}
        >>> template.add_version(v1)
        >>> template.add_version(v2)
        >>>
        >>> optimizer = PromptOptimizer(primary_metric="accuracy")
        >>> result = optimizer.recommend(template)
        >>> result.recommended_version
        '2.0'
    """

    def __init__(
        self,
        primary_metric: str = "accuracy",
        secondary_metrics: Optional[List[str]] = None,
        min_samples: int = 10,
    ):
        """Initialize optimizer.

        Args:
            primary_metric: Main metric to optimize for
            secondary_metrics: Additional metrics to consider
            min_samples: Minimum samples required for reliable recommendation
        """
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or []
        self.min_samples = min_samples

    def recommend(self, template: PromptTemplate) -> OptimizationResult:
        """Recommend best prompt version.

        Args:
            template: PromptTemplate to optimize

        Returns:
            OptimizationResult with recommendation
        """
        if not template.versions:
            return OptimizationResult(
                recommended_version="",
                confidence=0.0,
                reason="No versions available",
                alternative_versions=[],
            )

        # Filter versions with primary metric
        versions_with_metric = [
            v
            for v in template.versions
            if self.primary_metric in v.performance_metrics
        ]

        if not versions_with_metric:
            # No metrics available, use active version
            active = template.get_active()
            return OptimizationResult(
                recommended_version=active.version if active else "",
                confidence=0.5,
                reason="No performance data available, using active version",
                alternative_versions=[],
            )

        # Sort by primary metric (descending)
        sorted_versions = sorted(
            versions_with_metric,
            key=lambda v: v.performance_metrics[self.primary_metric],
            reverse=True,
        )

        best = sorted_versions[0]
        alternatives = [v.version for v in sorted_versions[1:3]]  # Top 3 alternatives

        # Calculate confidence based on metric difference
        if len(sorted_versions) > 1:
            best_score = best.performance_metrics[self.primary_metric]
            second_score = sorted_versions[1].performance_metrics[self.primary_metric]
            diff = best_score - second_score

            # Higher difference = higher confidence
            confidence = min(0.5 + diff, 1.0)
        else:
            confidence = 0.7  # Only one version with metrics

        reason = f"Highest {self.primary_metric}: {best.performance_metrics[self.primary_metric]:.3f}"

        # Check secondary metrics
        if self.secondary_metrics:
            secondary_info = []
            for metric in self.secondary_metrics:
                if metric in best.performance_metrics:
                    secondary_info.append(
                        f"{metric}={best.performance_metrics[metric]:.3f}"
                    )

            if secondary_info:
                reason += f" ({', '.join(secondary_info)})"

        return OptimizationResult(
            recommended_version=best.version,
            confidence=confidence,
            reason=reason,
            alternative_versions=alternatives,
        )

    def suggest_improvements(self, template: PromptTemplate) -> List[str]:
        """Suggest improvements based on performance data.

        Args:
            template: PromptTemplate to analyze

        Returns:
            List of suggestions
        """
        suggestions = []

        if not template.versions:
            suggestions.append("No versions available - create initial version")
            return suggestions

        # Check if versions have metrics
        versions_with_metrics = [
            v for v in template.versions if v.performance_metrics
        ]

        if not versions_with_metrics:
            suggestions.append(
                "No performance metrics tracked - start tracking to enable optimization"
            )

        # Check for stagnant performance
        if len(versions_with_metrics) >= 3:
            recent_versions = sorted(
                versions_with_metrics, key=lambda v: v.created_at, reverse=True
            )[:3]

            if all(
                self.primary_metric in v.performance_metrics for v in recent_versions
            ):
                scores = [
                    v.performance_metrics[self.primary_metric] for v in recent_versions
                ]
                variation = max(scores) - min(scores)

                if variation < 0.05:  # Less than 5% variation
                    suggestions.append(
                        "Recent versions show little variation - consider trying different approaches"
                    )

        # Check for active version optimality
        active = template.get_active()
        if active:
            recommendation = self.recommend(template)
            if (
                recommendation.recommended_version
                and recommendation.recommended_version != active.version
            ):
                suggestions.append(
                    f"Consider switching to version {recommendation.recommended_version} "
                    f"(better {self.primary_metric})"
                )

        return suggestions
