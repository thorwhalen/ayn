"""Cache implementations for agent results."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""

    ttl: Optional[int] = 3600  # Time to live in seconds (1 hour default)
    max_size: Optional[int] = 1000  # Maximum cache entries
    similarity_threshold: float = 0.95  # For semantic caching (0-1)
    enabled: bool = True


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Cache Stats:\n"
            f"  Hits: {self.hits}\n"
            f"  Misses: {self.misses}\n"
            f"  Hit Rate: {self.hit_rate:.2%}\n"
            f"  Evictions: {self.evictions}\n"
            f"  Total Size: {self.total_size}"
        )


class Cache(ABC):
    """Abstract base class for cache implementations."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    def _make_key(self, obj: Any) -> str:
        """Create a cache key from an object."""
        if isinstance(obj, str):
            return hashlib.sha256(obj.encode()).hexdigest()[:16]
        elif isinstance(obj, dict):
            # Sort keys for consistent hashing
            serialized = json.dumps(obj, sort_keys=True)
            return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        else:
            # Convert to string and hash
            return hashlib.sha256(str(obj).encode()).hexdigest()[:16]


class InMemoryCache(Cache):
    """Simple in-memory cache with TTL support.

    Example:
        >>> cache = InMemoryCache()
        >>> cache.set("key1", "value1")
        >>> cache.get("key1")
        'value1'
        >>> cache.delete("key1")
        >>> cache.get("key1") is None
        True
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        super().__init__(config)
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired."""
        if not self.config.enabled:
            return None

        if key in self._cache:
            value, timestamp = self._cache[key]

            # Check if expired
            if self.config.ttl and (time.time() - timestamp > self.config.ttl):
                del self._cache[key]
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            return value

        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with timestamp."""
        if not self.config.enabled:
            return

        # Evict oldest entries if at max size
        if self.config.max_size and len(self._cache) >= self.config.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats.evictions += 1

        self._cache[key] = (value, time.time())
        self.stats.total_size = len(self._cache)

    def delete(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            self.stats.total_size = len(self._cache)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.stats.total_size = 0


class SemanticCache(Cache):
    """Semantic cache using embeddings for similarity matching.

    Caches similar inputs together using embedding similarity.
    Requires an embedding function to compute text embeddings.

    Example:
        >>> def dummy_embed(text):
        ...     # In real usage, use OpenAI, HuggingFace, etc.
        ...     return [hash(text) % 100 for _ in range(3)]
        >>> cache = SemanticCache(embed_fn=dummy_embed)
        >>> cache.set("What is AI?", "Artificial Intelligence")
        >>> result = cache.get("What is AI?")  # Exact match
        >>> result == "Artificial Intelligence"
        True
    """

    def __init__(
        self,
        embed_fn: Optional[callable] = None,
        config: Optional[CacheConfig] = None,
    ):
        super().__init__(config)
        self.embed_fn = embed_fn
        self._cache: Dict[str, Tuple[Any, list, float]] = {}  # key -> (value, embedding, timestamp)

    def _get_embedding(self, text: str) -> Optional[list]:
        """Get embedding for text."""
        if not self.embed_fn:
            return None

        try:
            return self.embed_fn(text)
        except Exception:
            return None

    def _cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value using semantic similarity."""
        if not self.config.enabled or not self.embed_fn:
            return None

        # Get embedding for query
        query_embedding = self._get_embedding(key)
        if not query_embedding:
            self.stats.misses += 1
            return None

        # Find most similar cached entry
        best_match = None
        best_similarity = 0.0

        for cached_key, (value, embedding, timestamp) in self._cache.items():
            # Check if expired
            if self.config.ttl and (time.time() - timestamp > self.config.ttl):
                continue

            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = value

        # Return if above threshold
        if best_similarity >= self.config.similarity_threshold:
            self.stats.hits += 1
            return best_match

        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value with its embedding."""
        if not self.config.enabled or not self.embed_fn:
            return

        embedding = self._get_embedding(key)
        if not embedding:
            return

        # Evict if at max size
        if self.config.max_size and len(self._cache) >= self.config.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats.evictions += 1

        cache_key = self._make_key(key)
        self._cache[cache_key] = (value, embedding, time.time())
        self.stats.total_size = len(self._cache)

    def delete(self, key: str) -> None:
        """Remove entry from cache."""
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            self.stats.total_size = len(self._cache)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.stats.total_size = 0
