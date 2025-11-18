"""Prompt selection strategies."""

from __future__ import annotations

from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import PromptTemplate, PromptVersion


class SelectionStrategy(str, Enum):
    """Strategies for selecting prompt versions."""

    ACTIVE = "active"  # Use active version
    LATEST = "latest"  # Use latest version
    BEST_PERFORMING = "best_performing"  # Use version with best metrics
    A_B_TEST = "a_b_test"  # Randomly select for A/B testing


class PromptSelector:
    """Selects prompt versions based on strategy.

    Example:
        >>> from ayn.prompts import PromptTemplate, PromptVersion
        >>> template = PromptTemplate(name="test")
        >>> v1 = PromptVersion(version="1.0", template="Hello {name}")
        >>> v1.performance_metrics["accuracy"] = 0.8
        >>> v2 = PromptVersion(version="2.0", template="Hi {name}!")
        >>> v2.performance_metrics["accuracy"] = 0.9
        >>> template.add_version(v1)
        >>> template.add_version(v2)
        >>>
        >>> selector = PromptSelector(strategy=SelectionStrategy.BEST_PERFORMING)
        >>> selected = selector.select(template, metric="accuracy")
        >>> selected.version
        '2.0'
    """

    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.ACTIVE):
        """Initialize selector.

        Args:
            strategy: Selection strategy to use
        """
        self.strategy = strategy

    def select(
        self,
        template: PromptTemplate,
        metric: str = "accuracy",
    ) -> Optional[PromptVersion]:
        """Select a prompt version based on strategy.

        Args:
            template: PromptTemplate to select from
            metric: Metric name for BEST_PERFORMING strategy

        Returns:
            Selected PromptVersion or None
        """
        if not template.versions:
            return None

        if self.strategy == SelectionStrategy.ACTIVE:
            return template.get_active()

        elif self.strategy == SelectionStrategy.LATEST:
            # Return most recently created
            return max(template.versions, key=lambda v: v.created_at)

        elif self.strategy == SelectionStrategy.BEST_PERFORMING:
            # Return version with best metric
            versions_with_metric = [
                v for v in template.versions if metric in v.performance_metrics
            ]

            if not versions_with_metric:
                # Fall back to active if no metrics
                return template.get_active()

            return max(
                versions_with_metric,
                key=lambda v: v.performance_metrics[metric],
            )

        elif self.strategy == SelectionStrategy.A_B_TEST:
            # Random selection for A/B testing
            import random

            return random.choice(template.versions)

        return template.get_active()
