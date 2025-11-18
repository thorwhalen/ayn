"""Integration with i2 package for signature adaptation."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Dict

# i2 integration is optional
try:
    from i2 import Sig, wrap
    HAS_I2 = True
except ImportError:
    HAS_I2 = False


def adapt_signature(
    input_mapper: Optional[Callable] = None,
    output_mapper: Optional[Callable] = None,
) -> Callable:
    """Adapt agent signature using i2.

    Automatically transforms inputs and outputs to match expected schemas.

    Args:
        input_mapper: Function to transform input
        output_mapper: Function to transform output

    Example:
        >>> if HAS_I2:
        ...     @adapt_signature(
        ...         input_mapper=lambda text: {"input": text},
        ...         output_mapper=lambda result: result["output"]
        ...     )
        ...     def process(data: dict) -> dict:
        ...         return {"output": data["input"].upper()}
        ...
        ...     result = process("hello")
        ...     result
        ...     'HELLO'
    """
    if not HAS_I2:
        # Fallback to manual wrapping if i2 not available
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Apply input mapper
                if input_mapper and args:
                    args = (input_mapper(args[0]),) + args[1:]

                result = func(*args, **kwargs)

                # Apply output mapper
                if output_mapper:
                    result = output_mapper(result)

                return result

            return wrapper

        return decorator

    # Use i2's wrap if available
    def decorator(func: Callable) -> Callable:
        return wrap(
            func,
            input_trans=input_mapper,
            output_trans=output_mapper,
        )

    return decorator


def auto_convert(
    input_schema: Optional[Dict[str, type]] = None,
    output_schema: Optional[type] = None,
) -> Callable:
    """Automatically convert inputs/outputs to match schema.

    Args:
        input_schema: Expected input schema (dict of field -> type)
        output_schema: Expected output type

    Example:
        >>> @auto_convert(
        ...     input_schema={"count": int, "message": str},
        ...     output_schema=str
        ... )
        ... def repeat_message(count: int, message: str) -> str:
        ...     return message * count
        >>>
        >>> result = repeat_message(count="3", message="Hi")  # Auto-converts count to int
        >>> result
        'HiHiHi'
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Convert input types if schema provided
            if input_schema and kwargs:
                converted_kwargs = {}
                for key, value in kwargs.items():
                    if key in input_schema:
                        expected_type = input_schema[key]
                        try:
                            converted_kwargs[key] = expected_type(value)
                        except (TypeError, ValueError):
                            converted_kwargs[key] = value
                    else:
                        converted_kwargs[key] = value
                kwargs = converted_kwargs

            result = func(*args, **kwargs)

            # Convert output type if schema provided
            if output_schema and result is not None:
                try:
                    result = output_schema(result)
                except (TypeError, ValueError):
                    pass  # Keep original if conversion fails

            return result

        return wrapper

    return decorator


class SignatureAdapter:
    """Adapter for converting between different agent signatures.

    Useful when integrating agents with different calling conventions.

    Example:
        >>> adapter = SignatureAdapter()
        >>> adapter.register_conversion(
        ...     "text_input",
        ...     lambda text: {"messages": [{"role": "user", "content": text}]}
        ... )
        >>> converted = adapter.convert("text_input", "Hello")
        >>> converted
        {'messages': [{'role': 'user', 'content': 'Hello'}]}
    """

    def __init__(self):
        self.conversions: Dict[str, Callable] = {}

    def register_conversion(self, name: str, converter: Callable):
        """Register a conversion function.

        Args:
            name: Name of the conversion
            converter: Function to perform conversion
        """
        self.conversions[name] = converter

    def convert(self, conversion_name: str, data: Any) -> Any:
        """Apply a registered conversion.

        Args:
            conversion_name: Name of conversion to apply
            data: Data to convert

        Returns:
            Converted data
        """
        if conversion_name not in self.conversions:
            raise ValueError(f"Unknown conversion: {conversion_name}")

        return self.conversions[conversion_name](data)

    def adapt_controller(self, controller: Any, conversion_name: str):
        """Wrap a controller's invoke method with signature adaptation.

        Args:
            controller: Controller to adapt
            conversion_name: Name of conversion to apply

        Returns:
            Adapted controller
        """
        original_invoke = controller.invoke

        def adapted_invoke(input_data: Any, **kwargs):
            converted_input = self.convert(conversion_name, input_data)
            return original_invoke(converted_input, **kwargs)

        controller.invoke = adapted_invoke
        return controller
