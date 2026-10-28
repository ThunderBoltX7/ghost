"""Comprehensive unit tests for Ghost CLI refactored modules."""

import tempfile
from pathlib import Path
import pytest

from ghost_cli.planning import parse_plan_response
from ghost_cli.permissions import is_within_project_root, classify_command, RISK_LOW, RISK_RISKY, RISK_CRITICAL
from ghost_cli.utils import truncate_output, _count_tokens
from ghost_cli.context import rewind_messages


def test_parse_plan_response_json():
    json_text = """
    Here is the plan:
    ```json
    {
      "objective": "Refactor codebase",
      "affected_files": ["main.py", "utils.py"],
      "risks": "None",
      "tasks": [
        {
          "id": 1,
          "description": "Extract main.py",
          "affected_files": ["main.py"],
          "dependencies": [],
          "expected_changes": "Split file",
          "verification_methods": ["syntax_check"],
          "risks": "Low"
        }
      ]
    }
    ```
    """
    plan = parse_plan_response(json_text, default_task="Default task")
    assert plan.objective == "Refactor codebase"
    assert "main.py" in plan.affected_files
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Extract main.py"


def test_parse_plan_response_markdown_fallback():
    md_text = """
    Objective: Fix critical bug in auth.py
    Step 1. Read auth.py and inspect tokens.
    Step 2. Update token validation logic in auth.py.
    """
    plan = parse_plan_response(md_text, default_task="Default task")
    assert plan.objective == "Fix critical bug in auth.py"
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == 1
    assert "auth.py" in plan.affected_files


def test_parse_plan_response_malformed():
    malformed = "Not a json response at all, just plain gibberish without steps."
    plan = parse_plan_response(malformed, default_task="Fallback task")
    assert plan.objective == "Fallback task"
    assert len(plan.tasks) >= 1


def test_path_sandbox():
    cwd = Path.cwd().resolve()
    inside_path = cwd / "src" / "main.py"
    outside_path = Path(tempfile.gettempdir()) / "malicious.py"

    is_in, _ = is_within_project_root(str(inside_path))
    assert is_in is True

    is_out, _ = is_within_project_root(str(outside_path))
    assert is_out is False


def test_command_classification():
    assert classify_command("git status") == RISK_LOW
    assert classify_command("npm install") == RISK_RISKY
    assert classify_command("rm -rf /") == RISK_CRITICAL
    assert classify_command("git push origin main --force") == RISK_CRITICAL
    assert classify_command("python -m pytest") == RISK_LOW


def test_truncate_output():
    short_text = "hello"
    assert truncate_output(short_text, max_chars=100) == "hello"

    long_text = "a" * 2000
    truncated = truncate_output(long_text, max_chars=100)
    assert len(truncated) < 200
    assert "truncated" in truncated


def test_token_estimation():
    assert _count_tokens("hello world") > 0


def test_rewind_messages():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task 1"},
        {"role": "assistant", "content": "response 1"},
        {"role": "user", "content": "task 2"},
        {"role": "assistant", "content": "response 2"},
    ]
    rewound = rewind_messages(messages, n_turns=1)
    assert len(rewound) == 3  # system, task 1, response 1
    assert rewound[-1]["content"] == "response 1"
