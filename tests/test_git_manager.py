import subprocess
import pytest
from pathlib import Path

import ghost_cli.git_manager as git_manager
from ghost_cli.git_manager import (
    _is_git_repo,
    create_shadow_checkpoint,
    history_push,
    history_undo,
)


@pytest.fixture
def temp_git_repo(tmp_path, monkeypatch):
    """Set up an isolated temporary git repository."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(git_manager, "CHECKPOINT_FILE", tmp_path / ".ghost" / "checkpoints.json")
    monkeypatch.setattr(git_manager, "HISTORY_FILE", tmp_path / ".ghost" / "history.json")

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@ghost.ai"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Ghost Tester"], cwd=tmp_path, check=True, capture_output=True)

    test_file = tmp_path / "hello.txt"
    test_file.write_text("initial content", encoding="utf-8")
    subprocess.run(["git", "add", "hello.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True, capture_output=True)

    return tmp_path


def test_is_git_repo(temp_git_repo):
    assert _is_git_repo() is True


def test_is_git_repo_false(tmp_path, monkeypatch):
    # Non-git directory
    non_git_dir = tmp_path / "not_git"
    non_git_dir.mkdir()
    monkeypatch.chdir(non_git_dir)
    assert _is_git_repo() is False


def test_create_shadow_checkpoint(temp_git_repo):
    ref = create_shadow_checkpoint(label="test checkpoint")
    assert ref is not None
    assert isinstance(ref, str)
    assert len(ref) >= 7


def test_history_push_and_undo_roundtrip(temp_git_repo):
    test_file = temp_git_repo / "hello.txt"
    initial_content = test_file.read_text(encoding="utf-8")
    assert initial_content == "initial content"

    # Push to history stack before editing
    history_push("edit", str(test_file), initial_content)

    # Edit the file
    test_file.write_text("completely modified content", encoding="utf-8")
    assert test_file.read_text(encoding="utf-8") == "completely modified content"

    # Undo
    reverted, err = history_undo(1)
    assert err is None
    assert str(test_file) in reverted
    assert test_file.read_text(encoding="utf-8") == "initial content"
