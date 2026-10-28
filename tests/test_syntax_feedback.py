"""Unit tests for post-edit syntax validation feedback loop."""

import pytest
from ghost_cli.tools import file_ops
from ghost_cli.tools.file_ops import write_file, edit_file


@pytest.fixture
def sandboxed_dir(tmp_path, monkeypatch):
    """Bypass external sandbox prompt for isolated test temp directory."""
    monkeypatch.setattr(file_ops, "check_path_sandbox", lambda fp: True)
    return tmp_path


def test_write_file_syntax_error_feedback(sandboxed_dir):
    """Verify write_file detects Python syntax errors and returns clear ERROR feedback."""
    test_py = sandboxed_dir / "broken.py"
    broken_code = "def foo(\n   print('missing parenthesis'\n"
    res = write_file(str(test_py), broken_code)
    assert "ERROR: Syntax validation failed after write:" in res
    assert "SyntaxError" in res


def test_write_file_valid_syntax(sandboxed_dir):
    """Verify write_file succeeds with valid Python syntax."""
    test_py = sandboxed_dir / "valid.py"
    valid_code = "def foo():\n    return 42\n"
    res = write_file(str(test_py), valid_code)
    assert "Created" in res or "Overwrote" in res
    assert "ERROR" not in res


def test_edit_file_syntax_error_feedback(sandboxed_dir):
    """Verify edit_file detects Python syntax errors and returns clear ERROR feedback."""
    test_py = sandboxed_dir / "target.py"
    test_py.write_text("def foo():\n    return 42\n", encoding="utf-8")

    res = edit_file(str(test_py), old_text="return 42", new_text="return 42 + (")
    assert "ERROR: Syntax validation failed after edit:" in res
    assert "SyntaxError" in res
