"""Integration with dol package for flexible storage backends."""

from __future__ import annotations

from typing import Any, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models import AgentMetadata
    from ..monitoring.logger import LogEntry

# dol integration is optional
try:
    from dol import Pipe, wrap_kvs, TextFiles, PickleFiles
    HAS_DOL = True
except ImportError:
    HAS_DOL = False


class DolBackedRegistry:
    """Agent registry backed by dol storage.

    Allows storing agents in any dol-compatible backend:
    - Local files (JSON, pickle)
    - S3
    - Database
    - MongoDB
    - Redis
    - etc.

    Example:
        >>> if HAS_DOL:
        ...     from dol import PickleFiles
        ...     import tempfile
        ...     import os
        ...     temp_dir = tempfile.mkdtemp()
        ...     store = PickleFiles(temp_dir)
        ...     registry = DolBackedRegistry(store)
        ...     # Use registry like a dict
        ...     import shutil
        ...     shutil.rmtree(temp_dir)
    """

    def __init__(self, store: Any):
        if not HAS_DOL:
            raise ImportError(
                "dol package required. Install with: pip install dol"
            )
        self.store = store

    def __getitem__(self, key: str) -> AgentMetadata:
        """Get agent metadata by key."""
        from ..core.models import AgentMetadata
        data = self.store[key]
        if isinstance(data, dict):
            return AgentMetadata.from_dict(data)
        return data

    def __setitem__(self, key: str, value: AgentMetadata):
        """Store agent metadata."""
        if hasattr(value, 'to_dict'):
            self.store[key] = value.to_dict()
        else:
            self.store[key] = value

    def __delitem__(self, key: str):
        """Delete agent metadata."""
        del self.store[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over agent keys."""
        return iter(self.store)

    def __len__(self) -> int:
        """Get number of agents."""
        return len(self.store)

    def __contains__(self, key: str) -> bool:
        """Check if agent exists."""
        return key in self.store

    def keys(self):
        """Get all agent keys."""
        return self.store.keys()

    def values(self):
        """Get all agent metadata."""
        return (self[k] for k in self.keys())

    def items(self):
        """Get all (key, metadata) pairs."""
        return ((k, self[k]) for k in self.keys())


class DolLogger:
    """Agent logger using dol for flexible storage.

    Store logs in any dol-compatible backend.

    Example:
        >>> if HAS_DOL:
        ...     from dol import TextFiles
        ...     import tempfile
        ...     import os
        ...     temp_dir = tempfile.mkdtemp()
        ...     store = TextFiles(temp_dir)
        ...     logger = DolLogger(store, agent_name="test")
        ...     import shutil
        ...     shutil.rmtree(temp_dir)
    """

    def __init__(self, store: Any, agent_name: str):
        if not HAS_DOL:
            raise ImportError(
                "dol package required. Install with: pip install dol"
            )
        self.store = store
        self.agent_name = agent_name

    def log(self, entry: LogEntry):
        """Log an entry.

        Args:
            entry: LogEntry to store
        """
        import json
        key = f"{self.agent_name}_{entry.datetime.strftime('%Y%m%d_%H%M%S_%f')}"
        self.store[key] = entry.to_json()

    def get_logs(self, limit: Optional[int] = None) -> list:
        """Get recent logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        import json
        from ..monitoring.logger import LogEntry

        keys = sorted(self.store.keys(), reverse=True)
        if limit:
            keys = keys[:limit]

        logs = []
        for key in keys:
            try:
                data = json.loads(self.store[key])
                # Reconstruct LogEntry from dict
                logs.append(data)
            except Exception:
                continue

        return logs
