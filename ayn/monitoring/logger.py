"""Execution logging for agents."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LogEntry:
    """A single log entry for an agent invocation."""

    timestamp: float = field(default_factory=time.time)
    input_data: Any = None
    output: Any = None
    latency_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @property
    def datetime(self) -> datetime:
        """Get datetime from timestamp."""
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class ExecutionLog:
    """Collection of log entries."""

    agent_name: str
    entries: List[LogEntry] = field(default_factory=list)

    def add(self, entry: LogEntry):
        """Add a log entry."""
        self.entries.append(entry)

    def filter(
        self,
        success: Optional[bool] = None,
        min_latency_ms: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
    ) -> List[LogEntry]:
        """Filter log entries.

        Args:
            success: Filter by success status
            min_latency_ms: Minimum latency threshold
            max_latency_ms: Maximum latency threshold

        Returns:
            Filtered list of log entries
        """
        filtered = self.entries

        if success is not None:
            filtered = [e for e in filtered if e.success == success]

        if min_latency_ms is not None:
            filtered = [
                e
                for e in filtered
                if e.latency_ms and e.latency_ms >= min_latency_ms
            ]

        if max_latency_ms is not None:
            filtered = [
                e
                for e in filtered
                if e.latency_ms and e.latency_ms <= max_latency_ms
            ]

        return filtered

    def get_errors(self) -> List[LogEntry]:
        """Get all error entries."""
        return self.filter(success=False)

    def get_slow_requests(self, threshold_ms: float = 1000) -> List[LogEntry]:
        """Get requests slower than threshold."""
        return self.filter(min_latency_ms=threshold_ms)


class AgentLogger:
    """Logs agent executions to disk.

    Can be integrated with dol for flexible storage backends.

    Example:
        >>> import tempfile
        >>> import os
        >>> temp_dir = tempfile.mkdtemp()
        >>> logger = AgentLogger("test_agent", log_dir=temp_dir)
        >>> logger.log_invocation(
        ...     input_data={"test": "input"},
        ...     output={"result": "output"},
        ...     latency_ms=150.5,
        ...     success=True
        ... )
        >>> # Clean up
        >>> import shutil
        >>> shutil.rmtree(temp_dir)
    """

    def __init__(
        self,
        agent_name: str,
        log_dir: Optional[str] = None,
        auto_flush: bool = True,
    ):
        self.agent_name = agent_name
        self.log_dir = Path(log_dir or os.path.expanduser("~/.ayn/logs"))
        self.auto_flush = auto_flush

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory log
        self.execution_log = ExecutionLog(agent_name=agent_name)

    def log_invocation(
        self,
        input_data: Any,
        output: Any,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an agent invocation.

        Args:
            input_data: Input to the agent
            output: Output from the agent
            latency_ms: Execution latency in milliseconds
            success: Whether invocation succeeded
            error: Error message if failed
            metadata: Additional metadata
        """
        entry = LogEntry(
            input_data=input_data,
            output=output,
            latency_ms=latency_ms,
            success=success,
            error=error,
            metadata=metadata or {},
        )

        self.execution_log.add(entry)

        if self.auto_flush:
            self._write_entry(entry)

    def _write_entry(self, entry: LogEntry):
        """Write a log entry to disk."""
        # Create filename with timestamp
        filename = f"{entry.datetime.strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = self.log_dir / self.agent_name / filename

        # Create agent directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(filepath, "w") as f:
            f.write(entry.to_json())

    def get_logs(
        self,
        success: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[LogEntry]:
        """Get logs from memory.

        Args:
            success: Filter by success status
            limit: Maximum number of entries to return

        Returns:
            List of log entries
        """
        if success is not None:
            entries = self.execution_log.filter(success=success)
        else:
            entries = self.execution_log.entries

        if limit:
            entries = entries[-limit:]

        return entries

    def clear(self):
        """Clear in-memory logs."""
        self.execution_log.entries.clear()
