"""Multi-agent collaboration features."""

from .debate import (
    AgentDebate,
    DebateResult,
    ConsensusStrategy,
)
from .ensemble import (
    AgentEnsemble,
    EnsembleStrategy,
    EnsembleResult,
)

__all__ = [
    # Debate
    "AgentDebate",
    "DebateResult",
    "ConsensusStrategy",
    # Ensemble
    "AgentEnsemble",
    "EnsembleStrategy",
    "EnsembleResult",
]
