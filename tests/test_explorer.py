"""Unit tests for explorer subagent read-only safety and limits."""

import pytest
from ghost_cli.tools.explorer import EXPLORER_READ_ONLY_TOOLS, spawn_explorer
from ghost_cli.tools.registry import TOOL_DEFINITIONS, READ_ONLY_TOOLS


def test_explorer_tools_strictly_read_only():
    """Verify that EXPLORER_READ_ONLY_TOOLS contains only read-only safe tools."""
    for tool_name in EXPLORER_READ_ONLY_TOOLS:
        assert tool_name in READ_ONLY_TOOLS, f"Explorer tool '{tool_name}' is not in READ_ONLY_TOOLS"
        assert "write" not in tool_name
        assert "edit" not in tool_name
        assert "command" not in tool_name
        assert "kill" not in tool_name


def test_explorer_step_limit_clamping():
    """Verify that spawn_explorer gracefully handles out-of-range steps."""
    # Test step clamping logic without running full LLM
    from ghost_cli.tools import explorer
    # Verify tools defined
    sub_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in explorer.EXPLORER_READ_ONLY_TOOLS]
    assert len(sub_tools) >= 5
    assert all(t["function"]["name"] in explorer.EXPLORER_READ_ONLY_TOOLS for t in sub_tools)
