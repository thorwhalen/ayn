"""Agent ensemble for combining multiple agent outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


class EnsembleStrategy(str, Enum):
    """Strategies for combining ensemble outputs."""

    AVERAGE = "average"  # Average numeric outputs
    VOTING = "voting"  # Vote on categorical outputs
    STACKING = "stacking"  # Use meta-learner to combine
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by performance
    BEST_PERFORMER = "best_performer"  # Use best performing agent


@dataclass
class EnsembleResult:
    """Result from agent ensemble.

    Example:
        >>> result = EnsembleResult(
        ...     output="Final answer",
        ...     individual_outputs={"agent1": "Answer 1", "agent2": "Answer 2"},
        ...     confidence=0.9
        ... )
        >>> result.output
        'Final answer'
    """

    output: Any
    individual_outputs: Dict[str, Any]
    confidence: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Ensemble Result (Confidence: {self.confidence:.0%}):",
            f"  Output: {self.output}",
            "\n  Individual Outputs:",
        ]

        for agent, output in self.individual_outputs.items():
            weight_str = f"(weight: {self.weights[agent]:.2f})" if agent in self.weights else ""
            lines.append(f"    {agent}: {output} {weight_str}")

        return "\n".join(lines)


class AgentEnsemble:
    """Ensemble of agents for improved predictions.

    Combines outputs from multiple agents using various strategies.

    Example:
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> from ayn import GenericController
        >>>
        >>> meta1 = AgentMetadata(name="agent1", description="test", framework=AgentFramework.CUSTOM)
        >>> meta2 = AgentMetadata(name="agent2", description="test", framework=AgentFramework.CUSTOM)
        >>> agent1 = GenericController(meta1, ControllerConfig())
        >>> agent2 = GenericController(meta2, ControllerConfig())
        >>>
        >>> ensemble = AgentEnsemble([agent1, agent2])
        >>> # result = ensemble.predict({"input": "test"})
    """

    def __init__(
        self,
        agents: List[BaseAgentController],
        strategy: EnsembleStrategy = EnsembleStrategy.VOTING,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize ensemble.

        Args:
            agents: List of agents in ensemble
            strategy: Strategy for combining outputs
            weights: Optional weights for each agent
        """
        self.agents = agents
        self.strategy = strategy
        self.weights = weights or {}

    def predict(self, input_data: Any) -> EnsembleResult:
        """Make prediction using ensemble.

        Args:
            input_data: Input for all agents

        Returns:
            EnsembleResult with combined output
        """
        # Collect outputs from all agents
        individual_outputs = {}

        for i, agent in enumerate(self.agents):
            agent_name = getattr(agent, "metadata", None)
            agent_name = agent_name.name if agent_name else f"agent_{i}"

            try:
                output = agent.invoke(input_data)

                # Extract main output
                if isinstance(output, dict):
                    individual_outputs[agent_name] = output.get(
                        "result", output.get("answer", output)
                    )
                else:
                    individual_outputs[agent_name] = output

            except Exception as e:
                # Skip failed agents
                individual_outputs[agent_name] = None

        # Filter out None outputs
        valid_outputs = {k: v for k, v in individual_outputs.items() if v is not None}

        # Combine outputs using strategy
        combined_output, confidence = self._combine_outputs(valid_outputs)

        return EnsembleResult(
            output=combined_output,
            individual_outputs=individual_outputs,
            confidence=confidence,
            weights=self.weights,
        )

    def _combine_outputs(
        self,
        outputs: Dict[str, Any],
    ) -> tuple[Any, float]:
        """Combine outputs using selected strategy.

        Args:
            outputs: Valid outputs from agents

        Returns:
            (combined_output, confidence) tuple
        """
        if not outputs:
            return None, 0.0

        if self.strategy == EnsembleStrategy.VOTING:
            return self._voting(outputs)

        elif self.strategy == EnsembleStrategy.AVERAGE:
            return self._average(outputs)

        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average(outputs)

        elif self.strategy == EnsembleStrategy.BEST_PERFORMER:
            return self._best_performer(outputs)

        else:
            # Default to voting
            return self._voting(outputs)

    def _voting(self, outputs: Dict[str, Any]) -> tuple[Any, float]:
        """Majority voting.

        Args:
            outputs: Agent outputs

        Returns:
            (consensus, confidence) tuple
        """
        # Count votes
        vote_counts: Dict[Any, int] = {}
        for output in outputs.values():
            output_str = str(output)
            vote_counts[output_str] = vote_counts.get(output_str, 0) + 1

        # Find majority
        max_votes = max(vote_counts.values())
        consensus = max(vote_counts.keys(), key=lambda k: vote_counts[k])

        # Calculate confidence
        confidence = max_votes / len(outputs)

        # Convert back to original type
        for output in outputs.values():
            if str(output) == consensus:
                consensus = output
                break

        return consensus, confidence

    def _average(self, outputs: Dict[str, Any]) -> tuple[Any, float]:
        """Average numeric outputs.

        Args:
            outputs: Agent outputs

        Returns:
            (average, confidence) tuple
        """
        numeric_outputs = []

        for output in outputs.values():
            try:
                # Try to convert to float
                if isinstance(output, (int, float)):
                    numeric_outputs.append(float(output))
                elif isinstance(output, str):
                    numeric_outputs.append(float(output))
            except (ValueError, TypeError):
                # Skip non-numeric outputs
                continue

        if not numeric_outputs:
            # Fall back to voting
            return self._voting(outputs)

        # Calculate average
        average = sum(numeric_outputs) / len(numeric_outputs)

        # Calculate confidence based on variance
        variance = sum((x - average) ** 2 for x in numeric_outputs) / len(numeric_outputs)
        std_dev = variance ** 0.5

        # Lower variance = higher confidence
        # Normalize std_dev to confidence (0-1)
        confidence = max(0.0, 1.0 - (std_dev / (average + 1e-6)))

        return average, confidence

    def _weighted_average(self, outputs: Dict[str, Any]) -> tuple[Any, float]:
        """Weighted average using agent weights.

        Args:
            outputs: Agent outputs

        Returns:
            (weighted_average, confidence) tuple
        """
        if not self.weights:
            # Fall back to regular average
            return self._average(outputs)

        weighted_sum = 0.0
        total_weight = 0.0
        numeric_outputs = []

        for agent_name, output in outputs.items():
            weight = self.weights.get(agent_name, 1.0)

            try:
                if isinstance(output, (int, float)):
                    value = float(output)
                elif isinstance(output, str):
                    value = float(output)
                else:
                    continue

                weighted_sum += value * weight
                total_weight += weight
                numeric_outputs.append(value)

            except (ValueError, TypeError):
                continue

        if total_weight == 0 or not numeric_outputs:
            # Fall back to voting
            return self._voting(outputs)

        weighted_avg = weighted_sum / total_weight

        # Confidence based on weight distribution
        confidence = min(1.0, total_weight / len(outputs))

        return weighted_avg, confidence

    def _best_performer(self, outputs: Dict[str, Any]) -> tuple[Any, float]:
        """Use output from best performing agent.

        Args:
            outputs: Agent outputs

        Returns:
            (best_output, confidence) tuple
        """
        if not self.weights:
            # Fall back to first agent
            first_output = next(iter(outputs.values()))
            return first_output, 0.5

        # Find agent with highest weight
        best_agent = max(self.weights.keys(), key=lambda k: self.weights[k])

        if best_agent in outputs:
            return outputs[best_agent], self.weights[best_agent]
        else:
            # Fall back to first agent
            first_output = next(iter(outputs.values()))
            return first_output, 0.5

    def set_weights(self, weights: Dict[str, float]):
        """Set agent weights.

        Args:
            weights: Dict mapping agent names to weights
        """
        self.weights = weights

    def update_weights_from_performance(
        self,
        performance_metrics: Dict[str, float],
    ):
        """Update weights based on performance metrics.

        Args:
            performance_metrics: Dict mapping agent names to performance scores
        """
        # Normalize performance metrics to weights
        total = sum(performance_metrics.values())
        if total > 0:
            self.weights = {
                agent: score / total
                for agent, score in performance_metrics.items()
            }
