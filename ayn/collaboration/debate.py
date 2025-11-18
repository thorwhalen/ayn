"""Multi-agent debate system for improved decision making."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


class ConsensusStrategy(str, Enum):
    """Strategies for reaching consensus."""

    MAJORITY_VOTE = "majority_vote"  # Simple majority wins
    WEIGHTED_VOTE = "weighted_vote"  # Weighted by confidence
    UNANIMOUS = "unanimous"  # All agents must agree
    BEST_CONFIDENCE = "best_confidence"  # Highest confidence wins
    JUDGE = "judge"  # Separate judge agent decides


@dataclass
class DebateResult:
    """Result of a multi-agent debate.

    Example:
        >>> result = DebateResult(
        ...     consensus="Option A",
        ...     confidence=0.85,
        ...     votes={"agent1": "Option A", "agent2": "Option A", "agent3": "Option B"}
        ... )
        >>> result.consensus
        'Option A'
    """

    consensus: Any
    confidence: float  # 0-1
    votes: Dict[str, Any]
    reasoning: Dict[str, str] = field(default_factory=dict)
    rounds: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Debate Result (Confidence: {self.confidence:.0%}):",
            f"  Consensus: {self.consensus}",
            f"  Rounds: {self.rounds}",
            "\n  Votes:",
        ]

        for agent, vote in self.votes.items():
            match = "✓" if vote == self.consensus else "✗"
            lines.append(f"    {match} {agent}: {vote}")

        if self.reasoning:
            lines.append("\n  Reasoning:")
            for agent, reason in self.reasoning.items():
                lines.append(f"    {agent}: {reason[:100]}...")

        return "\n".join(lines)


class AgentDebate:
    """Multi-agent debate system for collaborative decision making.

    Runs multiple agents on the same input and combines their outputs
    using various consensus strategies.

    Example:
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> from ayn import GenericController
        >>>
        >>> meta1 = AgentMetadata(name="agent1", description="test", framework=AgentFramework.CUSTOM)
        >>> meta2 = AgentMetadata(name="agent2", description="test", framework=AgentFramework.CUSTOM)
        >>> agent1 = GenericController(meta1, ControllerConfig())
        >>> agent2 = GenericController(meta2, ControllerConfig())
        >>>
        >>> debate = AgentDebate([agent1, agent2])
        >>> # result = debate.run({"question": "What is 2+2?"})
    """

    def __init__(
        self,
        agents: List[BaseAgentController],
        strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY_VOTE,
        max_rounds: int = 3,
        judge_agent: Optional[BaseAgentController] = None,
    ):
        """Initialize debate system.

        Args:
            agents: List of agents to participate
            strategy: Consensus strategy to use
            max_rounds: Maximum debate rounds
            judge_agent: Optional judge agent for JUDGE strategy
        """
        self.agents = agents
        self.strategy = strategy
        self.max_rounds = max_rounds
        self.judge_agent = judge_agent

    def run(self, input_data: Any) -> DebateResult:
        """Run a debate with all agents.

        Args:
            input_data: Input for all agents

        Returns:
            DebateResult with consensus
        """
        votes = {}
        reasoning = {}

        # Round 1: Initial responses
        for i, agent in enumerate(self.agents):
            agent_name = getattr(agent, "metadata", None)
            agent_name = agent_name.name if agent_name else f"agent_{i}"

            try:
                response = agent.invoke(input_data)

                # Extract vote and reasoning
                if isinstance(response, dict):
                    votes[agent_name] = response.get("answer", response.get("result", response))
                    reasoning[agent_name] = response.get("reasoning", "")
                else:
                    votes[agent_name] = response
                    reasoning[agent_name] = ""

            except Exception as e:
                votes[agent_name] = None
                reasoning[agent_name] = f"Error: {e}"

        # Apply consensus strategy
        consensus, confidence = self._reach_consensus(votes, reasoning)

        return DebateResult(
            consensus=consensus,
            confidence=confidence,
            votes=votes,
            reasoning=reasoning,
            rounds=1,
        )

    def _reach_consensus(
        self,
        votes: Dict[str, Any],
        reasoning: Dict[str, str],
    ) -> tuple[Any, float]:
        """Reach consensus from votes.

        Args:
            votes: Agent votes
            reasoning: Agent reasoning

        Returns:
            (consensus, confidence) tuple
        """
        # Filter out None votes (errors)
        valid_votes = {k: v for k, v in votes.items() if v is not None}

        if not valid_votes:
            return None, 0.0

        if self.strategy == ConsensusStrategy.MAJORITY_VOTE:
            return self._majority_vote(valid_votes)

        elif self.strategy == ConsensusStrategy.UNANIMOUS:
            return self._unanimous(valid_votes)

        elif self.strategy == ConsensusStrategy.JUDGE and self.judge_agent:
            return self._judge_decision(valid_votes, reasoning)

        else:
            # Default to majority vote
            return self._majority_vote(valid_votes)

    def _majority_vote(self, votes: Dict[str, Any]) -> tuple[Any, float]:
        """Simple majority voting.

        Args:
            votes: Valid votes

        Returns:
            (consensus, confidence) tuple
        """
        # Count votes
        vote_counts: Dict[Any, int] = {}
        for vote in votes.values():
            vote_str = str(vote)  # Convert to string for hashing
            vote_counts[vote_str] = vote_counts.get(vote_str, 0) + 1

        # Find majority
        if not vote_counts:
            return None, 0.0

        max_votes = max(vote_counts.values())
        consensus = max(vote_counts.keys(), key=lambda k: vote_counts[k])

        # Calculate confidence (proportion of votes for consensus)
        confidence = max_votes / len(votes)

        # Convert back to original type if possible
        for vote in votes.values():
            if str(vote) == consensus:
                consensus = vote
                break

        return consensus, confidence

    def _unanimous(self, votes: Dict[str, Any]) -> tuple[Any, float]:
        """Unanimous consensus required.

        Args:
            votes: Valid votes

        Returns:
            (consensus, confidence) tuple
        """
        # Check if all votes are the same
        unique_votes = set(str(v) for v in votes.values())

        if len(unique_votes) == 1:
            # Unanimous
            consensus = next(iter(votes.values()))
            return consensus, 1.0
        else:
            # No consensus
            return None, 0.0

    def _judge_decision(
        self,
        votes: Dict[str, Any],
        reasoning: Dict[str, str],
    ) -> tuple[Any, float]:
        """Use judge agent to decide.

        Args:
            votes: Valid votes
            reasoning: Agent reasoning

        Returns:
            (consensus, confidence) tuple
        """
        if not self.judge_agent:
            # Fall back to majority vote
            return self._majority_vote(votes)

        # Prepare judge input
        judge_input = {
            "question": "Review the following agent responses and select the best answer",
            "responses": [
                {
                    "agent": agent,
                    "answer": answer,
                    "reasoning": reasoning.get(agent, ""),
                }
                for agent, answer in votes.items()
            ],
        }

        try:
            decision = self.judge_agent.invoke(judge_input)

            # Extract consensus from judge decision
            if isinstance(decision, dict):
                consensus = decision.get("selected_answer", decision.get("result"))
                confidence = decision.get("confidence", 0.8)
            else:
                consensus = decision
                confidence = 0.8

            return consensus, confidence

        except Exception:
            # Fall back to majority vote
            return self._majority_vote(votes)

    def multi_round_debate(
        self,
        input_data: Any,
        convergence_threshold: float = 0.8,
    ) -> DebateResult:
        """Run multi-round debate until convergence.

        Args:
            input_data: Input for debate
            convergence_threshold: Confidence threshold for stopping

        Returns:
            DebateResult from all rounds
        """
        for round_num in range(1, self.max_rounds + 1):
            result = self.run(input_data)
            result.rounds = round_num

            # Check convergence
            if result.confidence >= convergence_threshold:
                return result

            # Prepare input for next round with previous votes
            input_data = {
                "original_question": input_data,
                "previous_round": {
                    "votes": result.votes,
                    "reasoning": result.reasoning,
                },
                "instruction": "Consider the previous round's responses and provide your updated answer",
            }

        return result
