"""Permission system for agent access control."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Set


class Permission(str, Enum):
    """Standard agent permissions."""

    # Read permissions
    READ_DOCUMENTS = "read:documents"
    READ_DATABASE = "read:database"
    READ_FILES = "read:files"

    # Write permissions
    WRITE_DOCUMENTS = "write:documents"
    WRITE_DATABASE = "write:database"
    WRITE_FILES = "write:files"

    # Execution permissions
    EXECUTE_CODE = "execute:code"
    EXECUTE_COMMANDS = "execute:commands"

    # Network permissions
    NETWORK_HTTP = "network:http"
    NETWORK_EXTERNAL_API = "network:external_api"

    # Special permissions
    ADMIN = "admin"
    IMPERSONATE = "impersonate"


@dataclass
class PermissionSet:
    """Set of permissions with grant/deny logic.

    Example:
        >>> perms = PermissionSet()
        >>> perms.grant(Permission.READ_DOCUMENTS)
        >>> perms.has_permission(Permission.READ_DOCUMENTS)
        True
        >>> perms.deny(Permission.WRITE_DATABASE)
        >>> perms.has_permission(Permission.WRITE_DATABASE)
        False
    """

    granted: Set[Permission] = field(default_factory=set)
    denied: Set[Permission] = field(default_factory=set)

    def grant(self, permission: Permission):
        """Grant a permission.

        Args:
            permission: Permission to grant
        """
        self.granted.add(permission)
        # Remove from denied if present
        self.denied.discard(permission)

    def deny(self, permission: Permission):
        """Explicitly deny a permission.

        Args:
            permission: Permission to deny
        """
        self.denied.add(permission)
        # Remove from granted if present
        self.granted.discard(permission)

    def revoke(self, permission: Permission):
        """Revoke a permission (remove from both granted and denied).

        Args:
            permission: Permission to revoke
        """
        self.granted.discard(permission)
        self.denied.discard(permission)

    def has_permission(self, permission: Permission) -> bool:
        """Check if permission is granted.

        Args:
            permission: Permission to check

        Returns:
            True if granted and not explicitly denied
        """
        # Explicit deny takes precedence
        if permission in self.denied:
            return False

        # Check for admin permission (grants all)
        if Permission.ADMIN in self.granted and Permission.ADMIN not in self.denied:
            return True

        # Check if explicitly granted
        return permission in self.granted

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if any of the permissions are granted.

        Args:
            permissions: List of permissions to check

        Returns:
            True if any permission is granted
        """
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if all permissions are granted.

        Args:
            permissions: List of permissions to check

        Returns:
            True if all permissions are granted
        """
        return all(self.has_permission(p) for p in permissions)

    def __str__(self) -> str:
        """String representation."""
        lines = []

        if self.granted:
            lines.append("Granted:")
            for perm in sorted(self.granted, key=lambda p: p.value):
                lines.append(f"  ✓ {perm.value}")

        if self.denied:
            lines.append("Denied:")
            for perm in sorted(self.denied, key=lambda p: p.value):
                lines.append(f"  ✗ {perm.value}")

        return "\n".join(lines) if lines else "No permissions set"


class PermissionManager:
    """Manages permissions for multiple agents.

    Example:
        >>> manager = PermissionManager()
        >>> manager.create_agent("agent1")
        >>> manager.grant("agent1", Permission.READ_DOCUMENTS)
        >>> manager.check_permission("agent1", Permission.READ_DOCUMENTS)
        True
    """

    def __init__(self):
        self.agents: dict[str, PermissionSet] = {}

    def create_agent(self, agent_id: str, permissions: Optional[PermissionSet] = None):
        """Create or update an agent's permissions.

        Args:
            agent_id: Agent identifier
            permissions: Initial permissions (default: empty)
        """
        self.agents[agent_id] = permissions or PermissionSet()

    def grant(self, agent_id: str, permission: Permission):
        """Grant permission to an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission to grant
        """
        if agent_id not in self.agents:
            self.create_agent(agent_id)

        self.agents[agent_id].grant(permission)

    def deny(self, agent_id: str, permission: Permission):
        """Deny permission to an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission to deny
        """
        if agent_id not in self.agents:
            self.create_agent(agent_id)

        self.agents[agent_id].deny(permission)

    def check_permission(self, agent_id: str, permission: Permission) -> bool:
        """Check if agent has permission.

        Args:
            agent_id: Agent identifier
            permission: Permission to check

        Returns:
            True if permission is granted
        """
        if agent_id not in self.agents:
            return False

        return self.agents[agent_id].has_permission(permission)

    def get_permissions(self, agent_id: str) -> Optional[PermissionSet]:
        """Get agent's permission set.

        Args:
            agent_id: Agent identifier

        Returns:
            PermissionSet or None if agent not found
        """
        return self.agents.get(agent_id)


def require_permission(permission: Permission) -> Callable:
    """Decorator to require permission for method execution.

    Args:
        permission: Required permission

    Example:
        >>> perms = PermissionSet()
        >>> perms.grant(Permission.READ_DOCUMENTS)
        >>>
        >>> @require_permission(Permission.READ_DOCUMENTS)
        ... def read_doc(permissions=perms):
        ...     return "document content"
        >>>
        >>> read_doc(permissions=perms)
        'document content'
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get permissions from various sources
            permissions = None

            # Check kwargs
            if "permissions" in kwargs:
                permissions = kwargs.get("permissions")
            # Check if first arg is a PermissionSet
            elif args and isinstance(args[0], PermissionSet):
                permissions = args[0]
            # Check if object has permissions attribute
            elif args and hasattr(args[0], "permissions"):
                permissions = args[0].permissions

            # Verify permission
            if permissions and isinstance(permissions, PermissionSet):
                if not permissions.has_permission(permission):
                    raise PermissionError(
                        f"Permission denied: {permission.value} required"
                    )
            else:
                # No permissions found - be strict and deny
                raise PermissionError(
                    f"No permissions provided for {func.__name__}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_permission(*permissions: Permission) -> Callable:
    """Decorator to require at least one of the specified permissions.

    Args:
        *permissions: Required permissions (at least one)

    Example:
        >>> perms = PermissionSet()
        >>> perms.grant(Permission.READ_DOCUMENTS)
        >>>
        >>> @require_any_permission(Permission.READ_DOCUMENTS, Permission.READ_FILES)
        ... def read_data(permissions=perms):
        ...     return "data"
        >>>
        >>> read_data(permissions=perms)
        'data'
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get permissions
            perm_set = None

            if "permissions" in kwargs:
                perm_set = kwargs.get("permissions")
            elif args and isinstance(args[0], PermissionSet):
                perm_set = args[0]
            elif args and hasattr(args[0], "permissions"):
                perm_set = args[0].permissions

            # Verify at least one permission
            if perm_set and isinstance(perm_set, PermissionSet):
                if not perm_set.has_any_permission(list(permissions)):
                    perm_names = ", ".join(p.value for p in permissions)
                    raise PermissionError(
                        f"Permission denied: one of [{perm_names}] required"
                    )
            else:
                raise PermissionError(
                    f"No permissions provided for {func.__name__}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
