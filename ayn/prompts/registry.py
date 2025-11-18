"""Prompt registry for managing and versioning prompts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PromptVersion:
    """A specific version of a prompt.

    Example:
        >>> v1 = PromptVersion(
        ...     version="1.0",
        ...     template="You are a helpful assistant. {instruction}",
        ...     description="Initial version"
        ... )
        >>> v1.version
        '1.0'
    """

    version: str
    template: str
    description: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """Render the template with variables.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered prompt
        """
        return self.template.format(**kwargs)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PromptVersion:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PromptTemplate:
    """A prompt template with multiple versions.

    Example:
        >>> template = PromptTemplate(
        ...     name="greeting",
        ...     description="Greeting prompt"
        ... )
        >>> v1 = PromptVersion(version="1.0", template="Hello {name}!")
        >>> template.add_version(v1)
        >>> template.get_version("1.0").render(name="Alice")
        'Hello Alice!'
    """

    name: str
    description: Optional[str] = None
    versions: List[PromptVersion] = field(default_factory=list)
    active_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def add_version(self, version: PromptVersion):
        """Add a new version.

        Args:
            version: PromptVersion to add
        """
        # Check if version already exists
        existing = self.get_version(version.version)
        if existing:
            # Replace existing version
            self.versions = [v for v in self.versions if v.version != version.version]

        self.versions.append(version)

        # Set as active if first version or no active version set
        if not self.active_version or len(self.versions) == 1:
            self.active_version = version.version

    def get_version(self, version: str) -> Optional[PromptVersion]:
        """Get a specific version.

        Args:
            version: Version identifier

        Returns:
            PromptVersion or None
        """
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_active(self) -> Optional[PromptVersion]:
        """Get the active version.

        Returns:
            Active PromptVersion or None
        """
        if self.active_version:
            return self.get_version(self.active_version)
        return None

    def set_active(self, version: str):
        """Set the active version.

        Args:
            version: Version to set as active
        """
        if self.get_version(version):
            self.active_version = version
        else:
            raise ValueError(f"Version {version} not found")

    def list_versions(self) -> List[str]:
        """List all version identifiers.

        Returns:
            List of version strings
        """
        return [v.version for v in self.versions]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "versions": [v.to_dict() for v in self.versions],
            "active_version": self.active_version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PromptTemplate:
        """Create from dictionary."""
        versions = [PromptVersion.from_dict(v) for v in data.get("versions", [])]
        return cls(
            name=data["name"],
            description=data.get("description"),
            versions=versions,
            active_version=data.get("active_version"),
            tags=data.get("tags", []),
        )


class PromptRegistry:
    """Registry for managing prompt templates and versions.

    Stores prompts with versioning, A/B testing, and performance tracking.

    Example:
        >>> import tempfile
        >>> import os
        >>> temp_dir = tempfile.mkdtemp()
        >>> registry = PromptRegistry(storage_dir=temp_dir)
        >>>
        >>> # Register a prompt
        >>> template = PromptTemplate(name="greeting", description="Greet users")
        >>> v1 = PromptVersion(version="1.0", template="Hello {name}!")
        >>> template.add_version(v1)
        >>> registry.register(template)
        >>>
        >>> # Retrieve and render
        >>> prompt = registry.get("greeting")
        >>> prompt.get_active().render(name="Bob")
        'Hello Bob!'
        >>>
        >>> # Clean up
        >>> import shutil
        >>> shutil.rmtree(temp_dir)
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize prompt registry.

        Args:
            storage_dir: Directory to store prompts (default: ~/.ayn/prompts)
        """
        self.storage_dir = Path(storage_dir or os.path.expanduser("~/.ayn/prompts"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._templates: Dict[str, PromptTemplate] = {}

        # Load existing prompts
        self._load_all()

    def register(self, template: PromptTemplate):
        """Register a prompt template.

        Args:
            template: PromptTemplate to register
        """
        self._templates[template.name] = template
        self._save(template)

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name.

        Args:
            name: Template name

        Returns:
            PromptTemplate or None
        """
        return self._templates.get(name)

    def list(self, tag: Optional[str] = None) -> List[str]:
        """List all registered prompts.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of prompt names
        """
        if tag:
            return [
                name
                for name, template in self._templates.items()
                if tag in template.tags
            ]
        return list(self._templates.keys())

    def delete(self, name: str):
        """Delete a prompt template.

        Args:
            name: Template name
        """
        if name in self._templates:
            del self._templates[name]

            # Delete from storage
            filepath = self.storage_dir / f"{name}.json"
            if filepath.exists():
                filepath.unlink()

    def update_performance(
        self,
        name: str,
        version: str,
        metrics: Dict[str, float],
    ):
        """Update performance metrics for a prompt version.

        Args:
            name: Template name
            version: Version identifier
            metrics: Performance metrics to update
        """
        template = self.get(name)
        if template:
            prompt_version = template.get_version(version)
            if prompt_version:
                prompt_version.performance_metrics.update(metrics)
                self._save(template)

    def _save(self, template: PromptTemplate):
        """Save template to disk.

        Args:
            template: PromptTemplate to save
        """
        filepath = self.storage_dir / f"{template.name}.json"

        with open(filepath, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

    def _load_all(self):
        """Load all prompts from storage."""
        if not self.storage_dir.exists():
            return

        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    template = PromptTemplate.from_dict(data)
                    self._templates[template.name] = template
            except Exception:
                # Skip invalid files
                continue
