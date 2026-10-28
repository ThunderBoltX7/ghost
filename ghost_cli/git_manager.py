"""Git operations, shadow checkpoints, and undo history stack for Ghost CLI."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .config import CHECKPOINT_FILE, HISTORY_FILE, ensure_data_dir

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}


def _git_available() -> bool:
    return shutil.which("git") is not None


def _is_git_repo() -> bool:
    if not _git_available():
        return False
    try:
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, timeout=5)
        return res.returncode == 0 and res.stdout.strip() == "true"
    except Exception:
        return False


def _run_git(args, timeout=30) -> str:
    if not _git_available():
        return "Error: git is not installed or not on PATH."
    try:
        res = subprocess.run(["git"] + args, capture_output=True, text=True, timeout=timeout)
        out = res.stdout.strip()
        err = res.stderr.strip()
        if res.returncode != 0:
            return err or out or f"git {' '.join(args)} failed (exit {res.returncode})."
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out."
    except Exception as e:
        return f"Git error: {e}"


def git_status() -> str:
    return _run_git(["status", "--short", "--branch"])


def git_diff(path: str = None) -> str:
    args = ["diff", "--color=never"]
    if path:
        args.append(path)
    out = _run_git(args)
    return out[:6000] if out else "No changes."


def git_log(n: int = 10) -> str:
    try:
        n = max(1, int(n))
    except (TypeError, ValueError):
        n = 10
    return _run_git(["log", f"-{n}", "--oneline", "--decorate"])


def git_commit(message: str) -> str:
    if not message or not message.strip():
        return "Error: a non-empty commit message is required."
    add_out = _run_git(["add", "-A"])
    if add_out.lower().startswith("error"):
        return add_out
    return _run_git(["commit", "-m", message])


def git_push(remote: str = "origin", branch: str = None) -> str:
    args = ["push", remote or "origin"]
    if branch:
        args.append(branch)
    return _run_git(args)


# Shadow Git Checkpoints
def _load_checkpoints() -> List[Dict[str, Any]]:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_checkpoints(stack: List[Dict[str, Any]]):
    try:
        ensure_data_dir()
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(stack[-25:], f)
    except Exception:
        pass


def create_shadow_checkpoint(label: str = "turn") -> Optional[str]:
    if not _is_git_repo():
        return None
    try:
        head_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        head_sha = head_res.stdout.strip() if head_res.returncode == 0 else ""
        stash_res = subprocess.run(["git", "stash", "create"], capture_output=True, text=True, timeout=10)
        stash_sha = stash_res.stdout.strip() if stash_res.returncode == 0 else ""
        ref = stash_sha or head_sha
        if not ref:
            return None
        stack = _load_checkpoints()
        stack.append({
            "ref": ref,
            "head": head_sha,
            "is_stash": bool(stash_sha),
            "label": label[:80],
            "cwd": os.getcwd(),
            "ts": time.time(),
        })
        _save_checkpoints(stack)
        return ref
    except Exception:
        return None


def restore_shadow_checkpoint() -> Tuple[bool, str]:
    if not _is_git_repo():
        return False, "Not inside a git repository."
    stack = _load_checkpoints()
    cwd = os.getcwd()
    matching = [i for i, c in enumerate(stack) if c.get("cwd") == cwd]
    if not matching:
        return False, "No git checkpoints recorded for this workspace."
    idx = matching[-1]
    entry = stack.pop(idx)
    _save_checkpoints(stack)
    ref = entry["ref"]
    head = entry.get("head") or "HEAD"
    try:
        subprocess.run(["git", "reset", "--hard", head], capture_output=True, text=True, timeout=15)
        if entry.get("is_stash"):
            subprocess.run(["git", "stash", "apply", "--index", ref], capture_output=True, text=True, timeout=15)
        from .tools.search import INDEX, invalidate_cache
        INDEX.build_index()
        invalidate_cache(".")
        return True, f"Restored workspace to checkpoint ({entry.get('label', 'turn')}) [{ref[:8]}]."
    except Exception as e:
        return False, f"Failed to restore git checkpoint: {e}"


# Undo / history
MAX_HISTORY = 25


def _load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(stack):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(stack[-MAX_HISTORY:], f)
    except Exception:
        pass


def history_push(action: str, filepath: str, previous_content: str):
    stack = _load_history()
    stack.append({
        "action": action,
        "filepath": filepath,
        "previous_content": previous_content,
        "ts": time.time(),
    })
    _save_history(stack)


def history_undo(n: int = 1):
    stack = _load_history()
    if not stack:
        ok, msg = restore_shadow_checkpoint()
        if ok:
            return [msg], None
        return [], "Nothing to undo."
    reverted = []
    n = max(1, min(n, len(stack)))
    for _ in range(n):
        entry = stack.pop()
        try:
            with open(entry["filepath"], "w", encoding="utf-8") as f:
                f.write(entry["previous_content"])
            reverted.append(entry["filepath"])
            from .tools.search import INDEX, invalidate_cache
            INDEX.invalidate_file(entry["filepath"])
        except Exception as e:
            _save_history(stack)
            return reverted, f"Undo failed on {entry['filepath']}: {e}"
    _save_history(stack)
    from .tools.search import invalidate_cache
    invalidate_cache(".")
    return reverted, None


def history_list_recent(limit: int = 10):
    stack = _load_history()
    return list(reversed(stack[-limit:]))


def history_clear():
    _save_history([])


def history_count():
    return len(_load_history())


def checkpoint_create(label: str = "manual") -> Optional[str]:
    return create_shadow_checkpoint(label)


def checkpoint_restore() -> Tuple[bool, str]:
    return restore_shadow_checkpoint()


def checkpoint_list() -> List[Dict[str, Any]]:
    return _load_checkpoints()


def undo_last_edit() -> Tuple[bool, str]:
    stack = _load_history()
    if not stack:
        return restore_shadow_checkpoint()
    reverted, err = history_undo(1)
    if err:
        return False, err
    if reverted:
        return True, f"Reverted edit to: {', '.join(reverted)}"
    return False, "Nothing to undo."


def get_git_branch() -> str:
    """Return active git branch name, 'detached', or 'no-git'."""
    if not _is_git_repo():
        return "no-git"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            branch = res.stdout.strip()
            return branch if branch else "detached"
    except Exception:
        pass
    return "no-git"


