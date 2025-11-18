"""Caching decorators for agent methods."""

from __future__ import annotations

import functools
import json
from typing import Any, Callable, Optional

from .cache import Cache, InMemoryCache, SemanticCache, CacheConfig


def cached(
    cache: Optional[Cache] = None,
    key_fn: Optional[Callable[[Any], str]] = None,
) -> Callable:
    """Decorator to cache function results.

    Args:
        cache: Cache instance to use (defaults to InMemoryCache)
        key_fn: Function to generate cache key from arguments

    Example:
        >>> cache_instance = InMemoryCache()
        >>> @cached(cache=cache_instance)
        ... def expensive_operation(x):
        ...     return x * 2
        >>> expensive_operation(5)
        10
        >>> expensive_operation(5)  # Returns from cache
        10
    """
    _cache = cache or InMemoryCache()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default key generation
                key = _make_default_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = _cache.get(key)
            if cached_result is not None:
                return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            _cache.set(key, result)

            return result

        # Attach cache instance for inspection
        wrapper._cache = _cache  # type: ignore
        return wrapper

    return decorator


def semantic_cached(
    embed_fn: Callable[[str], list],
    cache: Optional[SemanticCache] = None,
    input_extractor: Optional[Callable] = None,
) -> Callable:
    """Decorator to cache using semantic similarity.

    Args:
        embed_fn: Function to generate embeddings from text
        cache: SemanticCache instance to use
        input_extractor: Function to extract text from function arguments

    Example:
        >>> def dummy_embed(text):
        ...     return [hash(text) % 100 for _ in range(3)]
        >>> @semantic_cached(embed_fn=dummy_embed)
        ... def answer_question(question: str) -> str:
        ...     return f"Answer to: {question}"
        >>> answer_question("What is AI?")
        'Answer to: What is AI?'
    """
    _cache = cache or SemanticCache(embed_fn=embed_fn)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract input text for semantic matching
            if input_extractor:
                text_input = input_extractor(*args, **kwargs)
            else:
                # Default: use first string argument or convert first arg to string
                if args and isinstance(args[0], str):
                    text_input = args[0]
                elif args:
                    text_input = str(args[0])
                else:
                    text_input = json.dumps(kwargs, sort_keys=True)

            # Try to get from cache
            cached_result = _cache.get(text_input)
            if cached_result is not None:
                return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            _cache.set(text_input, result)

            return result

        # Attach cache instance for inspection
        wrapper._cache = _cache  # type: ignore
        return wrapper

    return decorator


def _make_default_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a default cache key from function name and arguments."""
    # Skip 'self' argument if present
    args_to_hash = args[1:] if args and hasattr(args[0], func_name) else args

    # Create a deterministic representation
    key_parts = [func_name]

    # Add positional args
    for arg in args_to_hash:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, dict):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))

    # Add keyword args (sorted for consistency)
    if kwargs:
        key_parts.append(json.dumps(kwargs, sort_keys=True))

    return ":".join(key_parts)
