"""Unit tests for Phase 2: Terminal UI, status bar, diff viewer, error diagnostics, and autocomplete."""

import re
import pytest
from types import SimpleNamespace
from rich.console import Console

from ghost_cli.ui import format_status_bar, diagnose_error, render_diff_viewer, UI
from ghost_cli.sessions import TodoManager
from ghost_cli.cli import SlashCommandCompleter, COMMAND_REGISTRY


def test_format_status_bar():
    """Verify status bar format strictly matches Phase 2.1 specification."""
    status_text = format_status_bar(
        model="gpt-4o",
        mode="normal",
        used_tokens=1500,
        budget=64000,
        pct=2.34,
        git_branch="main",
        level=5,
        title="Operative",
    )
    expected = "[Model: gpt-4o] [Mode: normal] [Tokens: 1500/64000 (2.3%)] [Branch: main] [XP: Lv.5 Operative]"
    assert status_text == expected

    # Verify regex pattern covers all fields
    pattern = r"\[Model:\s+[^\]]+\]\s+\[Mode:\s+[^\]]+\]\s+\[Tokens:\s+\d+/\d+\s+\([^\]]+%\)\]\s+\[Branch:\s+[^\]]+\]\s+\[XP:\s+Lv\.\d+\s+[^\]]+\]"
    assert re.match(pattern, status_text)


def test_diagnose_error_api_key():
    """Verify authentication errors return What, Why, and Fix."""
    what, why, fix = diagnose_error("Incorrect API key provided: sk-proj-123456789. 401 Unauthorized")
    assert "authentication failed" in what.lower()
    assert "api key" in why.lower()
    assert "OPENAI_API_KEY" in fix or "/config" in fix or "/doctor" in fix


def test_diagnose_error_rate_limit():
    """Verify rate limit errors return quota explanation and action."""
    what, why, fix = diagnose_error("Error code: 429 - Rate limit reached for requests per min")
    assert "rate limit" in what.lower() or "quota" in what.lower()
    assert "throttled" in why.lower() or "limits" in why.lower()
    assert "/model" in fix or "wait" in fix.lower()


def test_diagnose_error_network_timeout():
    """Verify network connection timeout errors return network diagnosis."""
    what, why, fix = diagnose_error("HTTPSConnectionPool: Read timed out after 30 seconds.")
    assert "timed out" in what.lower() or "network" in what.lower()
    assert "connection" in why.lower() or "internet" in why.lower()
    assert "/doctor" in fix


def test_diagnose_error_file_not_found():
    """Verify missing file errors suggest search tools."""
    what, why, fix = diagnose_error("FileNotFoundError: [Errno 2] No such file or directory: 'src/missing.py'")
    assert "not found" in what.lower()
    assert "exist" in why.lower() or "path" in why.lower()
    assert "list_dir" in fix or "search_code" in fix


def test_diagnose_error_permission_denied():
    """Verify permission and sandbox violations return clear guidance."""
    what, why, fix = diagnose_error("PermissionError: [Errno 13] Permission denied: '/etc/shadow'")
    assert "access was denied" in what.lower() or "permission" in what.lower()
    assert "permissions" in why.lower() or "root" in why.lower()
    assert "sandbox" in fix.lower() or "permissions" in fix.lower()


def test_diagnose_error_syntax():
    """Verify syntax error reports point to edit_file fix."""
    what, why, fix = diagnose_error("SyntaxError: invalid syntax at line 42 in module.py")
    assert "syntax validation" in what.lower() or "syntax" in what.lower()
    assert "syntax" in why.lower() or "parentheses" in why.lower()
    assert "edit_file" in fix


def test_render_diff_viewer_unified():
    """Verify unified diff rendering on standard width terminal."""
    diff_sample = (
        "--- old.py\n"
        "+++ new.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    test_console = Console(width=80, force_terminal=True, color_system="truecolor")
    with test_console.capture() as capture:
        render_diff_viewer(diff_sample, filepath="foo.py", console_instance=test_console)
    output = capture.get()
    assert "Diff Preview" in output
    assert "-    return 1" in output
    assert "+    return 2" in output


def test_render_diff_viewer_side_by_side():
    """Verify side-by-side diff rendering on wide terminal (>120 cols)."""
    diff_sample = (
        "--- old.py\n"
        "+++ new.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    test_console = Console(width=140, force_terminal=True, color_system="truecolor")
    with test_console.capture() as capture:
        render_diff_viewer(diff_sample, filepath="foo.py", console_instance=test_console)
    output = capture.get()
    assert "Diff Preview" in output
    assert "Original" in output
    assert "Modified" in output


def test_todo_manager_checklist_symbols():
    """Verify TodoManager renders and formats with [x], [>], [ ]."""
    mgr = TodoManager()
    mgr.update([
        {"id": "1", "content": "Analyze architecture", "status": "completed"},
        {"id": "2", "content": "Implement feature", "status": "in_progress"},
        {"id": "3", "content": "Write unit tests", "status": "pending"},
    ])

    text = mgr.format_text()
    assert "[x] #1: Analyze architecture (completed)" in text
    assert "[>] #2: Implement feature (in_progress)" in text
    assert "[ ] #3: Write unit tests (pending)" in text

    # Verify update_status dynamically shifts items
    mgr.update_status("2", "completed")
    mgr.update_status("3", "in_progress")
    text_updated = mgr.format_text()
    assert "[x] #2: Implement feature (completed)" in text_updated
    assert "[>] #3: Write unit tests (in_progress)" in text_updated


def test_slash_command_completer():
    """Verify slash command autocomplete provides all registered commands and descriptions."""
    completer = SlashCommandCompleter()
    doc = SimpleNamespace(text_before_cursor="/")
    completions = list(completer.get_completions(doc, None))
    assert len(completions) >= len(COMMAND_REGISTRY)

    # Check that /help and /status are present with valid descriptions
    cmd_names = [getattr(c, "text", str(c)) for c in completions]
    assert "/help" in cmd_names
    assert "/status" in cmd_names
    assert "/mode" in cmd_names
    assert "/test" in cmd_names

    # Check prefix filtering
    doc_prefix = SimpleNamespace(text_before_cursor="/mod")
    prefix_completions = list(completer.get_completions(doc_prefix, None))
    prefix_names = [getattr(c, "text", str(c)) for c in prefix_completions]
    assert "/model" in prefix_names
    assert "/mode" in prefix_names
    assert "/status" not in prefix_names
