"""Agent registry implementation."""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from typing import Dict, List, Optional

from ..core.models import AgentMetadata
from .searchers import (
    AgentSearcher,
    GitHubAgentSearcher,
    HuggingFaceAgentSearcher,
    AwesomeListSearcher,
)


class AgentRegistry(MutableMapping[str, AgentMetadata]):
    """Unified registry for searching and managing agents

    Aggregates results from multiple sources and provides a dict-like interface.

    Example:
        >>> registry = AgentRegistry()
        >>> # Add a local agent
        >>> from ayn.core import AgentMetadata, AgentFramework
        >>> meta = AgentMetadata(name="test", description="test", framework=AgentFramework.CUSTOM)
        >>> registry['test'] = meta
        >>> registry['test'].name
        'test'
        >>> 'test' in registry
        True
    """

    def __init__(
        self,
        searchers: Optional[List[AgentSearcher]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.searchers = searchers or [
            GitHubAgentSearcher(),
            HuggingFaceAgentSearcher(),
            AwesomeListSearcher(),
        ]
        self.cache_dir = cache_dir or os.path.expanduser("~/.ayn/registry")
        self._local_store: Dict[str, AgentMetadata] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached registry from disk"""
        if os.path.exists(self.cache_dir):
            cache_file = os.path.join(self.cache_dir, "registry.json")
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    data = json.load(f)
                    for key, meta_dict in data.items():
                        self._local_store[key] = AgentMetadata.from_dict(meta_dict)

    def _save_cache(self):
        """Save registry cache to disk"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "registry.json")
        data = {k: v.to_dict() for k, v in self._local_store.items()}
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        source: Optional[str] = None,
        **filters,
    ) -> List[AgentMetadata]:
        """Search across all sources

        Args:
            query: Search query
            limit: Max results
            source: Specific source to search (github, huggingface, awesome)
            **filters: Additional filters

        Returns:
            List of agent metadata

        Example:
            >>> registry = AgentRegistry(searchers=[])
            >>> # Add some test data
            >>> from ayn.core import AgentMetadata, AgentFramework
            >>> registry['agent1'] = AgentMetadata(
            ...     name="agent1", description="test agent",
            ...     framework=AgentFramework.CREWAI, tags=['data']
            ... )
            >>> results = registry.search('agent')
            >>> len(results) >= 1
            True
        """
        results = []

        # Search local store first
        for name, meta in self._local_store.items():
            if (
                query.lower() in name.lower()
                or query.lower() in meta.description.lower()
                or any(query.lower() in tag.lower() for tag in meta.tags)
            ):
                results.append(meta)

        # Search remote sources if not enough results
        if len(results) < limit:
            for searcher in self.searchers:
                if source and source not in searcher.__class__.__name__.lower():
                    continue
                try:
                    remote_results = list(
                        searcher.search(query, limit=limit - len(results), **filters)
                    )
                    results.extend(remote_results)
                    if len(results) >= limit:
                        break
                except Exception as e:
                    # Continue on error
                    continue

        return results[:limit]

    # MutableMapping interface
    def __getitem__(self, key: str) -> AgentMetadata:
        return self._local_store[key]

    def __setitem__(self, key: str, value: AgentMetadata):
        self._local_store[key] = value
        self._save_cache()

    def __delitem__(self, key: str):
        del self._local_store[key]
        self._save_cache()

    def __iter__(self):
        return iter(self._local_store)

    def __len__(self):
        return len(self._local_store)
