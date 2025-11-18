"""Validation rules for agent controllers."""

from __future__ import annotations

import importlib.util
import inspect
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from .validator import ValidationIssue, ValidationSeverity

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController


class ValidationRule(ABC):
    """Base class for validation rules."""

    @abstractmethod
    def validate(self, controller: BaseAgentController) -> List[ValidationIssue]:
        """Validate a controller and return any issues found.

        Args:
            controller: The controller to validate

        Returns:
            List of validation issues
        """
        pass


class ConfigValidationRule(ValidationRule):
    """Validates controller configuration."""

    def validate(self, controller: BaseAgentController) -> List[ValidationIssue]:
        issues = []

        # Check if config exists
        if not hasattr(controller, "config"):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Controller missing 'config' attribute",
                    rule=self.__class__.__name__,
                    suggestion="Ensure controller has a config attribute",
                )
            )
            return issues

        config = controller.config

        # Check for common misconfigurations
        if hasattr(config, "temperature"):
            temp = config.temperature
            if temp < 0 or temp > 2:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Temperature {temp} is outside typical range [0, 2]",
                        rule=self.__class__.__name__,
                        context={"temperature": temp},
                        suggestion="Use temperature between 0 and 2",
                    )
                )

        if hasattr(config, "max_tokens"):
            max_tokens = config.max_tokens
            if max_tokens and max_tokens > 100000:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"max_tokens={max_tokens} is very high and may be costly",
                        rule=self.__class__.__name__,
                        context={"max_tokens": max_tokens},
                        suggestion="Consider using a lower max_tokens value",
                    )
                )

        # Check for required timeout
        if hasattr(config, "timeout"):
            if config.timeout is None:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="No timeout configured - agent may hang indefinitely",
                        rule=self.__class__.__name__,
                        suggestion="Set a timeout value in the config",
                    )
                )

        return issues


class DependencyValidationRule(ValidationRule):
    """Validates that required dependencies are available."""

    FRAMEWORK_DEPENDENCIES = {
        "CREWAI": ["crewai"],
        "LANGCHAIN": ["langchain"],
        "AUTOGEN": ["autogen", "pyautogen"],
        "SMOLAGENTS": ["smolagents"],
    }

    def validate(self, controller: BaseAgentController) -> List[ValidationIssue]:
        issues = []

        # Check if controller has metadata
        if hasattr(controller, "metadata"):
            framework = controller.metadata.framework
            framework_name = framework.value if hasattr(framework, "value") else str(framework)

            # Check framework dependencies
            if framework_name.upper() in self.FRAMEWORK_DEPENDENCIES:
                required_deps = self.FRAMEWORK_DEPENDENCIES[framework_name.upper()]

                for dep in required_deps:
                    if not self._is_package_available(dep):
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                message=f"Required dependency '{dep}' not found for {framework_name}",
                                rule=self.__class__.__name__,
                                context={"framework": framework_name, "dependency": dep},
                                suggestion=f"Install with: pip install {dep}",
                            )
                        )

        # Check for invoke method
        if not hasattr(controller, "invoke") or not callable(controller.invoke):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Controller missing required 'invoke' method",
                    rule=self.__class__.__name__,
                    suggestion="Implement the invoke(input_data, **kwargs) method",
                )
            )

        return issues

    def _is_package_available(self, package_name: str) -> bool:
        """Check if a package is available."""
        spec = importlib.util.find_spec(package_name)
        return spec is not None


class SecurityValidationRule(ValidationRule):
    """Validates security-related concerns."""

    SENSITIVE_PATTERNS = [
        "api_key",
        "secret",
        "password",
        "token",
        "credential",
        "private_key",
    ]

    def validate(self, controller: BaseAgentController) -> List[ValidationIssue]:
        issues = []

        # Check config for hardcoded secrets
        if hasattr(controller, "config"):
            config = controller.config
            config_dict = vars(config) if hasattr(config, "__dict__") else {}

            for key, value in config_dict.items():
                if any(pattern in key.lower() for pattern in self.SENSITIVE_PATTERNS):
                    if value and isinstance(value, str) and len(value) > 0:
                        # Check if it's not an environment variable reference
                        if not value.startswith("${") and not value.startswith("$"):
                            issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.ERROR,
                                    message=f"Potential hardcoded secret in config: '{key}'",
                                    rule=self.__class__.__name__,
                                    context={"field": key},
                                    suggestion="Use environment variables or secure vaults for secrets",
                                )
                            )

        # Check for eval/exec usage (basic code inspection)
        source = None
        try:
            source = inspect.getsource(controller.invoke)
        except (TypeError, OSError):
            pass

        if source:
            dangerous_calls = ["eval(", "exec(", "compile(", "__import__"]
            for call in dangerous_calls:
                if call in source:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Potentially dangerous call '{call}' found in invoke method",
                            rule=self.__class__.__name__,
                            context={"call": call},
                            suggestion="Avoid using eval/exec for security reasons",
                        )
                    )

        return issues


class PerformanceValidationRule(ValidationRule):
    """Validates performance-related concerns."""

    def validate(self, controller: BaseAgentController) -> List[ValidationIssue]:
        issues = []

        # Check for streaming support
        has_stream = hasattr(controller, "stream") and callable(controller.stream)
        has_ainvoke = hasattr(controller, "ainvoke") and callable(controller.ainvoke)

        if not has_stream:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Controller does not implement streaming (stream method)",
                    rule=self.__class__.__name__,
                    suggestion="Implement stream() for better UX with long-running operations",
                )
            )

        if not has_ainvoke:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Controller does not implement async invoke (ainvoke method)",
                    rule=self.__class__.__name__,
                    suggestion="Implement ainvoke() for better performance in async contexts",
                )
            )

        # Check for caching
        has_caching = (
            hasattr(controller, "_cache")
            or hasattr(controller, "cache")
            or "cache" in dir(controller)
        )

        if not has_caching:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Controller does not appear to implement caching",
                    rule=self.__class__.__name__,
                    suggestion="Consider adding caching to reduce costs and improve performance",
                )
            )

        return issues
