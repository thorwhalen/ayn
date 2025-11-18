"""Configuration for agent controllers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ControllerConfig:
    """Configuration for agent controllers

    Example:
        >>> config = ControllerConfig(
        ...     model="gpt-4",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> config.model
        'gpt-4'
    """

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 60
    retry_count: int = 3
    api_key: Optional[str] = None
    additional_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
