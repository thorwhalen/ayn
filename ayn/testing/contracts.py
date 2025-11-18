"""Contract-based testing for agents."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, Union


@dataclass
class InputSchema:
    """Schema definition for agent inputs."""

    schema: Union[Type, Dict[str, Type]]
    required: bool = True
    description: Optional[str] = None

    def validate(self, data: Any) -> tuple[bool, Optional[str]]:
        """Validate input data against schema.

        Returns:
            (is_valid, error_message)
        """
        if isinstance(self.schema, type):
            # Simple type check
            if not isinstance(data, self.schema):
                return False, f"Expected {self.schema.__name__}, got {type(data).__name__}"
            return True, None

        elif isinstance(self.schema, dict):
            # Dict schema validation
            if not isinstance(data, dict):
                return False, f"Expected dict, got {type(data).__name__}"

            for key, expected_type in self.schema.items():
                if key not in data:
                    if self.required:
                        return False, f"Missing required field: {key}"
                elif not isinstance(data[key], expected_type):
                    return (
                        False,
                        f"Field '{key}' should be {expected_type.__name__}, "
                        f"got {type(data[key]).__name__}",
                    )

            return True, None

        return True, None


@dataclass
class OutputSchema:
    """Schema definition for agent outputs."""

    schema: Union[Type, Dict[str, Type]]
    description: Optional[str] = None

    def validate(self, data: Any) -> tuple[bool, Optional[str]]:
        """Validate output data against schema.

        Returns:
            (is_valid, error_message)
        """
        if isinstance(self.schema, type):
            if not isinstance(data, self.schema):
                return False, f"Expected {self.schema.__name__}, got {type(data).__name__}"
            return True, None

        elif isinstance(self.schema, dict):
            if not isinstance(data, dict):
                return False, f"Expected dict, got {type(data).__name__}"

            for key, expected_type in self.schema.items():
                if key in data and not isinstance(data[key], expected_type):
                    return (
                        False,
                        f"Field '{key}' should be {expected_type.__name__}, "
                        f"got {type(data[key]).__name__}",
                    )

            return True, None

        return True, None


@dataclass
class PerformanceConstraint:
    """Performance constraints for agent execution."""

    max_latency_ms: Optional[float] = None  # Maximum latency in milliseconds
    max_latency_p95_ms: Optional[float] = None  # 95th percentile latency
    max_latency_p99_ms: Optional[float] = None  # 99th percentile latency
    min_throughput_rps: Optional[float] = None  # Minimum requests per second

    def check_latency(self, latency_ms: float) -> tuple[bool, Optional[str]]:
        """Check if latency meets constraint.

        Returns:
            (is_valid, error_message)
        """
        if self.max_latency_ms and latency_ms > self.max_latency_ms:
            return False, f"Latency {latency_ms:.2f}ms exceeds limit of {self.max_latency_ms}ms"
        return True, None


@dataclass
class CostConstraint:
    """Cost constraints for agent execution."""

    max_cost_per_call: Optional[float] = None  # Maximum cost per invocation
    max_tokens_per_call: Optional[int] = None  # Maximum tokens per invocation

    def check_cost(self, cost: float) -> tuple[bool, Optional[str]]:
        """Check if cost meets constraint.

        Returns:
            (is_valid, error_message)
        """
        if self.max_cost_per_call and cost > self.max_cost_per_call:
            return False, f"Cost ${cost:.4f} exceeds limit of ${self.max_cost_per_call:.4f}"
        return True, None

    def check_tokens(self, tokens: int) -> tuple[bool, Optional[str]]:
        """Check if token usage meets constraint.

        Returns:
            (is_valid, error_message)
        """
        if self.max_tokens_per_call and tokens > self.max_tokens_per_call:
            return False, f"Tokens {tokens} exceeds limit of {self.max_tokens_per_call}"
        return True, None


@dataclass
class Contract:
    """Contract definition for an agent.

    Defines input/output schemas and performance/cost constraints.

    Example:
        >>> contract_def = Contract(
        ...     input_schema=InputSchema(schema=dict),
        ...     output_schema=OutputSchema(schema=str),
        ...     performance=PerformanceConstraint(max_latency_ms=1000),
        ...     cost=CostConstraint(max_cost_per_call=0.01)
        ... )
    """

    input_schema: Optional[InputSchema] = None
    output_schema: Optional[OutputSchema] = None
    performance: Optional[PerformanceConstraint] = None
    cost: Optional[CostConstraint] = None
    description: Optional[str] = None

    def validate_input(self, data: Any) -> tuple[bool, Optional[str]]:
        """Validate input data."""
        if self.input_schema:
            return self.input_schema.validate(data)
        return True, None

    def validate_output(self, data: Any) -> tuple[bool, Optional[str]]:
        """Validate output data."""
        if self.output_schema:
            return self.output_schema.validate(data)
        return True, None

    def check_performance(self, latency_ms: float) -> tuple[bool, Optional[str]]:
        """Check performance constraints."""
        if self.performance:
            return self.performance.check_latency(latency_ms)
        return True, None

    def check_cost(self, cost: float, tokens: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """Check cost constraints."""
        if self.cost:
            is_valid, msg = self.cost.check_cost(cost)
            if not is_valid:
                return is_valid, msg

            if tokens is not None:
                return self.cost.check_tokens(tokens)

        return True, None


def contract(
    input_schema: Optional[Union[Type, Dict[str, Type]]] = None,
    output_schema: Optional[Union[Type, Dict[str, Type]]] = None,
    max_latency_ms: Optional[float] = None,
    max_cost_per_call: Optional[float] = None,
    max_tokens_per_call: Optional[int] = None,
) -> Callable:
    """Decorator to enforce contracts on agent methods.

    Args:
        input_schema: Schema for input validation
        output_schema: Schema for output validation
        max_latency_ms: Maximum allowed latency in milliseconds
        max_cost_per_call: Maximum allowed cost per call
        max_tokens_per_call: Maximum allowed tokens per call

    Example:
        >>> @contract(
        ...     input_schema={"text": str},
        ...     output_schema=str,
        ...     max_latency_ms=500
        ... )
        ... def process_text(text):
        ...     return text.upper()
        >>> result = process_text({"text": "hello"})
        >>> result
        'HELLO'
    """
    contract_def = Contract(
        input_schema=InputSchema(schema=input_schema) if input_schema else None,
        output_schema=OutputSchema(schema=output_schema) if output_schema else None,
        performance=PerformanceConstraint(max_latency_ms=max_latency_ms)
        if max_latency_ms
        else None,
        cost=CostConstraint(
            max_cost_per_call=max_cost_per_call,
            max_tokens_per_call=max_tokens_per_call,
        )
        if (max_cost_per_call or max_tokens_per_call)
        else None,
    )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get input data (first arg or first kwarg)
            input_data = args[0] if args else (kwargs.get("input_data") or kwargs)

            # Validate input
            is_valid, error = contract_def.validate_input(input_data)
            if not is_valid:
                raise ValueError(f"Input validation failed: {error}")

            # Measure execution time
            start_time = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Validate output
            is_valid, error = contract_def.validate_output(result)
            if not is_valid:
                raise ValueError(f"Output validation failed: {error}")

            # Check performance
            is_valid, error = contract_def.check_performance(latency_ms)
            if not is_valid:
                raise RuntimeError(f"Performance constraint violated: {error}")

            # TODO: Check cost constraints if metadata available

            return result

        # Attach contract for inspection
        wrapper._contract = contract_def  # type: ignore

        return wrapper

    return decorator
