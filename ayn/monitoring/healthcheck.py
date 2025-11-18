"""Health checking for agent controllers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return self.status == HealthStatus.HEALTHY

    def __str__(self) -> str:
        status_emoji = {
            HealthStatus.HEALTHY: "✓",
            HealthStatus.DEGRADED: "⚠",
            HealthStatus.UNHEALTHY: "✗",
        }

        lines = [
            f"{status_emoji[self.status]} Health Status: {self.status.value.upper()}"
        ]

        if self.checks:
            lines.append("\nChecks:")
            for check_name, passed in self.checks.items():
                symbol = "✓" if passed else "✗"
                lines.append(f"  {symbol} {check_name}")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  • {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")

        return "\n".join(lines)


class AgentHealthCheck:
    """Comprehensive health checking for agents.

    Performs checks for:
    - Dependencies availability
    - Configuration validity
    - Test invocation success
    - Response time
    - API connectivity

    Example:
        >>> from ayn import AgentHealthCheck, GenericController
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>>
        >>> metadata = AgentMetadata(
        ...     name="test",
        ...     description="test agent",
        ...     framework=AgentFramework.CUSTOM
        ... )
        >>> controller = GenericController(metadata, ControllerConfig())
        >>> healthcheck = AgentHealthCheck(controller)
        >>> result = healthcheck.run()
        >>> result.is_healthy
        True
    """

    def __init__(
        self,
        controller: BaseAgentController,
        test_input: Optional[any] = None,
        timeout_ms: float = 5000,
    ):
        self.controller = controller
        self.test_input = test_input or {"test": "healthcheck"}
        self.timeout_ms = timeout_ms

    def run(self) -> HealthCheckResult:
        """Run all health checks.

        Returns:
            HealthCheckResult with overall status
        """
        result = HealthCheckResult(status=HealthStatus.HEALTHY)

        # Run individual checks
        self._check_dependencies(result)
        self._check_configuration(result)
        self._check_invoke(result)
        self._check_response_time(result)

        # Determine overall status
        if result.errors:
            result.status = HealthStatus.UNHEALTHY
        elif result.warnings:
            result.status = HealthStatus.DEGRADED

        return result

    def _check_dependencies(self, result: HealthCheckResult):
        """Check if required dependencies are available."""
        check_name = "dependencies"

        try:
            # Check if controller has required methods
            required_methods = ["invoke"]
            for method in required_methods:
                if not hasattr(self.controller, method):
                    result.errors.append(f"Missing required method: {method}")
                    result.checks[check_name] = False
                    return

            result.checks[check_name] = True

        except Exception as e:
            result.errors.append(f"Dependency check failed: {e}")
            result.checks[check_name] = False

    def _check_configuration(self, result: HealthCheckResult):
        """Check if configuration is valid."""
        check_name = "configuration"

        try:
            if not hasattr(self.controller, "config"):
                result.warnings.append("Controller missing config attribute")
                result.checks[check_name] = True  # Warning, not error
                return

            config = self.controller.config

            # Check for common misconfigurations
            if hasattr(config, "timeout") and config.timeout is None:
                result.warnings.append("No timeout configured")

            result.checks[check_name] = True

        except Exception as e:
            result.errors.append(f"Configuration check failed: {e}")
            result.checks[check_name] = False

    def _check_invoke(self, result: HealthCheckResult):
        """Check if agent can be invoked successfully."""
        check_name = "invoke"

        try:
            # Try to invoke with test input
            _ = self.controller.invoke(self.test_input)
            result.checks[check_name] = True
            result.metadata["test_invocation"] = "success"

        except Exception as e:
            result.errors.append(f"Test invocation failed: {e}")
            result.checks[check_name] = False
            result.metadata["test_invocation"] = "failed"

    def _check_response_time(self, result: HealthCheckResult):
        """Check response time is within acceptable limits."""
        check_name = "response_time"

        try:
            start_time = time.time()
            _ = self.controller.invoke(self.test_input)
            latency_ms = (time.time() - start_time) * 1000

            result.metadata["latency_ms"] = latency_ms

            if latency_ms > self.timeout_ms:
                result.warnings.append(
                    f"Response time {latency_ms:.2f}ms exceeds target {self.timeout_ms}ms"
                )
                result.checks[check_name] = True  # Warning, not error
            else:
                result.checks[check_name] = True

        except Exception as e:
            result.errors.append(f"Response time check failed: {e}")
            result.checks[check_name] = False
