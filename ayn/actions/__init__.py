"""Custom actions for ChatGPT and Claude."""

from .chatgpt import generate_chatgpt_action
from .claude import generate_claude_mcp_config

__all__ = [
    "generate_chatgpt_action",
    "generate_claude_mcp_config",
]
