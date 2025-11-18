"""Core validation framework for agents."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Must be fixed before deployment
    WARNING = "warning"  # Should be addressed
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: ValidationSeverity
    message: str
    rule: str
    context: Optional[dict] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.message}"]
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        if self.context:
            parts.append(f"  Context: {self.context}")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """Result of agent validation."""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not self.has_errors

    def __str__(self) -> str:
        if not self.issues:
            return "✓ Validation passed - no issues found"

        lines = [f"Found {len(self.issues)} issue(s):"]
        for issue in self.issues:
            lines.append(str(issue))
        return "\n\n".join(lines)


class AgentValidator:
    """Validates agent configurations and implementations.

    Checks for:
    - Missing dependencies
    - Invalid configurations
    - Security issues
    - Performance anti-patterns
    - Type mismatches
    - Required attributes

    Example:
        >>> from ayn import AgentValidator, GenericController
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>>
        >>> metadata = AgentMetadata(
        ...     name="test",
        ...     description="test agent",
        ...     framework=AgentFramework.CUSTOM
        ... )
        >>> controller = GenericController(metadata, ControllerConfig())
        >>> validator = AgentValidator()
        >>> result = validator.validate(controller)
        >>> result.is_valid
        True
    """

    def __init__(self):
        self.rules = []
        self._load_default_rules()

    def _load_default_rules(self):
        """Load all default validation rules."""
        from .rules import (
            ConfigValidationRule,
            DependencyValidationRule,
            SecurityValidationRule,
            PerformanceValidationRule,
        )

        self.rules = [
            ConfigValidationRule(),
            DependencyValidationRule(),
            SecurityValidationRule(),
            PerformanceValidationRule(),
        ]

    def validate(self, controller: BaseAgentController) -> ValidationResult:
        """Validate an agent controller.

        Args:
            controller: The controller to validate

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult()

        # Run all validation rules
        for rule in self.rules:
            try:
                issues = rule.validate(controller)
                result.issues.extend(issues)
            except Exception as e:
                # If a rule fails, add it as an error
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule '{rule.__class__.__name__}' failed: {e}",
                        rule=rule.__class__.__name__,
                    )
                )

        return result

    def add_rule(self, rule: Any):
        """Add a custom validation rule.

        Args:
            rule: A rule implementing validate(controller) -> List[ValidationIssue]
        """
        self.rules.append(rule)
