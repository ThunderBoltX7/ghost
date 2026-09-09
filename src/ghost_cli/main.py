#!/usr/bin/env python3
"""Ghost CLI — an AI pair-programming agent for the terminal.

Single-file version of the Ghost CLI v2. This preserves the original modular
behavior while keeping the entire application in one entrypoint.
"""

__version__ = "2.1.1"

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table


# ---------------------------------------------------------------------------
# Configuration / filesystem
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / ".ghost"
CONFIG_FILE = DATA_DIR / "ghost_config.json"
STATS_FILE = DATA_DIR / "ghost_stats.json"
UNDO_FILE = DATA_DIR / ".ghost_undo.json"
HISTORY_FILE = DATA_DIR / "ghost_history.json"
CACHE_FILE = DATA_DIR / "ghost_cache.json"

MODEL_PRESETS = {
    "1": ("deepseek/deepseek-v3.2", "DeepSeek V3.2 — Fast, precise coding"),
    "2": ("openai/gpt-4o", "GPT-4o — Deep multi-step reasoning"),
    "3": ("deepseek-ai/deepseek-v4-pro-0813", "DeepSeek V4 Pro — NVIDIA NIM"),
    "4": ("nvidia/nemotron-3.5-lightning:free", "Nemotron 3.5 Lightning — Free NIM"),
    "5": ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B — Fast open-weights model"),
}

DEFAULT_MODEL = "deepseek/deepseek-v3.2"
PERMISSION_MODES = ("safe", "ask", "strict")
DEFAULT_PERMISSION_MODE = "ask"
GITHUB_REPO_URL = "git+https://github.com/ThunderBoltX7/ghost"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg):
    try:
        ensure_data_dir()
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)
        return True
    except Exception:
        return False


def interactive_setup(cfg):
    """Handles onboarding, provider selection, and permanent saving."""
    console.clear()
    console.print(
        Panel(
            "[bold cyan]Ghost Configuration[/bold cyan]\n\n"
            "Configure your AI provider. This is saved permanently to ~/.ghost/ghost_config.json.",
            border_style="cyan",
        )
    )

    console.print("Select your API Provider:")
    console.print("[cyan]1.[/cyan] OpenRouter (Recommended)")
    console.print("[cyan]2.[/cyan] OpenAI")
    console.print("[cyan]3.[/cyan] Groq")
    console.print("[cyan]4.[/cyan] DeepSeek")
    console.print("[cyan]5.[/cyan] NVIDIA NIM")
    console.print("[cyan]6.[/cyan] Custom (Any OpenAI-compatible endpoint)")

    choice = Prompt.ask("\n[bold white]Enter choice (1-6)[/bold white]", choices=["1", "2", "3", "4", "5", "6"], default="1")

    if choice == "6":
        provider_name = "Custom"
        base_url = Prompt.ask("[bold white]Enter Base URL (e.g., http://localhost:11434/v1)[/bold white]").strip()
        default_model = Prompt.ask("[bold white]Enter default model name[/bold white]", default="gpt-4").strip()
    else:
        providers = {
            "1": ("OpenRouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v3.2"),
            "2": ("OpenAI", "https://api.openai.com/v1", "gpt-4o"),
            "3": ("Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            "4": ("DeepSeek", "https://api.deepseek.com", "deepseek-chat"),
            "5": ("NVIDIA", "https://integrate.api.nvidia.com/v1", "deepseek-ai/deepseek-v4-pro-0813"),
        }
        provider_name, base_url, default_model = providers[choice]

    while True:
        new_key = Prompt.ask(f"\n[bold white]Enter your {provider_name} API Key[/bold white]", password=True).strip()

        if len(new_key) > 5:
            cfg["provider"] = provider_name
            cfg["base_url"] = base_url
            cfg["api_key"] = new_key
            cfg["model"] = default_model
            cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
            cfg.setdefault("auto_test", False)

            save_config(cfg)

            console.print(f"\n[bold green]✔ {provider_name} configured and saved securely to {CONFIG_FILE}[/bold green]")
            console.print("[dim]Booting system...[/dim]\n")
            return cfg
        else:
            console.print("[red]Invalid key length. Try again.[/red]")


def get_api_config(cfg):
    env_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    if env_key and "api_key" not in cfg:
        cfg["api_key"] = env_key
        if os.environ.get("NVIDIA_API_KEY") and not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            cfg["base_url"] = "https://integrate.api.nvidia.com/v1"
            cfg["model"] = "deepseek-ai/deepseek-v4-pro-0813"
            cfg["provider"] = "NVIDIA NIM"
        else:
            cfg["base_url"] = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            cfg["model"] = os.environ.get("GHOST_MODEL", DEFAULT_MODEL)
            cfg["provider"] = "Environment Variable"

    if not cfg.get("api_key"):
        cfg = interactive_setup(cfg)

    cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
    cfg.setdefault("auto_test", False)
    return cfg


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------
console = Console()


# ---------------------------------------------------------------------------
# Auto-upgrader
# ---------------------------------------------------------------------------
def upgrade_ghost():
    """Auto-upgrades Ghost CLI from GitHub using the active Python executable."""
    console.print(f"\n[bold cyan]▲ Connecting to repository to upgrade Ghost...[/bold cyan]")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        GITHUB_REPO_URL,
    ]

    status = console.status("[cyan]Running pip install --upgrade...[/cyan]", spinner="dots")
    status.start()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        status.stop()
        console.print("[red]Upgrade timed out after 180s. Check your internet connection.[/red]")
        return
    except KeyboardInterrupt:
        status.stop()
        console.print("\n[yellow]Upgrade interrupted by user.[/yellow]")
        return
    except Exception as e:
        status.stop()
        console.print(f"[red]Upgrade failed to execute:[/red] {e}")
        return
    finally:
        status.stop()

    if res.returncode == 0:
        console.print("\n[bold green]✔ Ghost upgraded successfully![/bold green]")
        console.print("[dim]Type /exit and run `ghost` again to use the new version.[/dim]\n")
    else:
        err_msg = res.stderr.strip() or res.stdout.strip()
        console.print(f"\n[bold red]Upgrade failed (exit code {res.returncode}):[/bold red]\n{err_msg}\n")


# ---------------------------------------------------------------------------
# Stats / XP / leveling
# ---------------------------------------------------------------------------
_stats_cache = None
_stats_dirty = False


def _read_stats_from_disk():
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"level": 1, "xp": 0, "edits": 0, "commands": 0}


def load_stats():
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = _read_stats_from_disk()
    return _stats_cache


def mark_dirty():
    global _stats_dirty
    _stats_dirty = True


def flush_stats():
    global _stats_dirty
    if _stats_dirty and _stats_cache is not None:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_FILE, "w") as f:
            json.dump(_stats_cache, f)
        _stats_dirty = False


def add_xp(amount, reason=""):
    global _stats_dirty
    stats = load_stats()
    stats["xp"] += amount
    threshold = stats["level"] * 100
    leveled = False
    if stats["xp"] >= threshold:
        stats["level"] += 1
        stats["xp"] -= threshold
        leveled = True
    _stats_dirty = True
    if leveled:
        console.print(f"[bold yellow]▲ Level up. Ghost is now Level {stats['level']}.[/bold yellow]")
    console.print(f"[dim green]+{amount} xp — {reason}[/dim green]")
    return stats


def bump(field, amount=1):
    """Increment a counter field (e.g. 'edits', 'commands') and mark dirty."""
    global _stats_dirty
    stats = load_stats()
    stats[field] = stats.get(field, 0) + amount
    _stats_dirty = True
    return stats


def get_rank(level):
    if level < 5:
        return "Rookie"
    if level < 15:
        return "Operative"
    if level < 30:
        return "Phantom"
    if level < 50:
        return "Wraith"
    return "Ghost Prime"


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------
def unified_diff_text(old_content: str, new_content: str, filepath: str, max_lines: int = 200) -> str:
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=filepath,
        tofile=filepath,
        lineterm="",
    )
    lines = list(diff)
    if not lines:
        return ""
    truncated = lines[:max_lines]
    text = "".join(line if line.endswith("\n") else line + "\n" for line in truncated)
    if len(lines) > max_lines:
        text += f"... ({len(lines) - max_lines} more lines omitted)\n"
    return text


# ---------------------------------------------------------------------------
# Project scan / cache
# ---------------------------------------------------------------------------
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

_EXT_LANG = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".rb": "Ruby",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ header",
    ".cs": "C#",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
}


def _detect_languages(files):
    counts = {}
    for f in files:
        lang = _EXT_LANG.get(Path(f).suffix)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def scan_project(root=".", max_depth=3, max_files=2000):
    root_path = Path(root)
    files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        rel = Path(dirpath).relative_to(root_path)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            files.append(str((rel / fn)))
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break

    languages = _detect_languages(files)
    markers = {
        "git": (root_path / ".git").exists(),
        "package.json": (root_path / "package.json").exists(),
        "pyproject.toml": (root_path / "pyproject.toml").exists(),
        "requirements.txt": (root_path / "requirements.txt").exists(),
        "Dockerfile": (root_path / "Dockerfile").exists(),
    }
    top_level = sorted({f.split(os.sep)[0] for f in files if f})[:50]

    return {
        "root": str(root_path.resolve()),
        "file_count": len(files),
        "languages": languages,
        "markers": markers,
        "top_level": top_level,
        "scanned_at": time.time(),
    }


CACHE_TTL_SECONDS = 300


def _load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(data):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_cached_scan(root, scanner_fn, force=False):
    """Returns (result_dict, was_cached: bool)."""
    cache = _load_cache()
    key = str(Path(root).resolve())
    entry = cache.get(key)
    now = time.time()

    if not force and entry and (now - entry.get("scanned_at", 0)) < CACHE_TTL_SECONDS:
        return entry, True

    fresh = scanner_fn(root)
    cache[key] = fresh
    _save_cache(cache)
    return fresh, False


def invalidate_cache(root):
    cache = _load_cache()
    key = str(Path(root).resolve())
    if key in cache:
        del cache[key]
        _save_cache(cache)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def _git_available() -> bool:
    return shutil.which("git") is not None


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


# ---------------------------------------------------------------------------
# Undo / history
# ---------------------------------------------------------------------------
MAX_HISTORY = 25


def _load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(stack):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(stack[-MAX_HISTORY:], f)
    except Exception:
        pass


def history_push(action: str, filepath: str, previous_content: str):
    """Record the pre-change content of a file so it can be restored later."""
    stack = _load_history()
    stack.append({
        "action": action,
        "filepath": filepath,
        "previous_content": previous_content,
        "ts": time.time(),
    })
    _save_history(stack)


def history_undo(n: int = 1):
    """Revert the last `n` recorded edits, most recent first.

    Returns (reverted_filepaths, error_message_or_None).
    """
    stack = _load_history()
    if not stack:
        return [], "Nothing to undo."
    reverted = []
    n = max(1, min(n, len(stack)))
    for _ in range(n):
        entry = stack.pop()
        try:
            with open(entry["filepath"], "w", encoding="utf-8") as f:
                f.write(entry["previous_content"])
            reverted.append(entry["filepath"])
        except Exception as e:
            _save_history(stack)
            return reverted, f"Undo failed on {entry['filepath']}: {e}"
    _save_history(stack)
    return reverted, None


def history_list_recent(limit: int = 10):
    stack = _load_history()
    return list(reversed(stack[-limit:]))


def history_clear():
    _save_history([])


def history_count():
    return len(_load_history())


# ---------------------------------------------------------------------------
# Planning mode
# ---------------------------------------------------------------------------
READ_ONLY_TOOLS = {
    "read_file",
    "list_dir",
    "search_code",
    "git_status",
    "git_diff",
    "git_log",
    "scan_project",
}

PLANNING_SYSTEM_SUFFIX = """

PLANNING MODE is active. You may only use read-only tools to investigate
the project: read_file, list_dir, search_code, git_status, git_diff,
git_log, scan_project. Do NOT call write_file, edit_file, run_command,
git_commit, or git_push — they are unavailable right now.

Investigate as needed, then respond with a concise, numbered, step-by-step
plan for the task. Do not perform the work yet — only propose the plan.
End your response once the plan is complete; do not ask the user to
confirm inline, the interface will handle that.
"""


class PlanState:
    def __init__(self):
        self.active = False
        self.task = None
        self.plan_text = None

    def start(self, task):
        self.active = True
        self.task = task
        self.plan_text = None

    def finish_drafting(self, plan_text):
        self.active = False
        self.plan_text = plan_text

    def clear(self):
        self.active = False
        self.task = None
        self.plan_text = None

    def has_plan(self):
        return bool(self.plan_text)


PLAN_STATE = PlanState()


# ---------------------------------------------------------------------------
# Safety system
# ---------------------------------------------------------------------------
RISK_LOW = "low"
RISK_RISKY = "risky"
RISK_CRITICAL = "critical"

CRITICAL_COMMAND_PATTERNS = [
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/(\s|$)",
    r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r?\s+/(\s|$)",
    r"rm\s+-rf\s+~",
    r"rm\s+-rf\s+\*",
    r"rm\s+-rf\s+\.\s*$",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"mkfs(\.\w+)?\s",
    r"dd\s+.*of=/dev/",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+-R\s+000",
    r"chmod\s+-R\s+777\s+/",
    r"git\s+push\s+.*--force",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[a-zA-Z]*f",
    r"drop\s+(table|database)",
    r"\bshutdown\b",
    r"\breboot\b",
    r"format\s+[a-zA-Z]:",
    r"del\s+/[sf]\s+/[qs]",
]

RISKY_COMMAND_PATTERNS = [
    r"\brm\b",
    r"\bmv\b",
    r"\bsudo\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"pip\s+(uninstall|install)",
    r"npm\s+(uninstall|install|ci)\b",
    r"yarn\s+(remove|add)\b",
    r"git\s+(checkout|reset|rebase|merge|stash\s+drop)",
    r"curl\s+.*\|\s*(sh|bash)",
    r"wget\s+.*\|\s*(sh|bash)",
    r"\bkill\b",
    r"docker\s+(rm|rmi|system\s+prune)",
    r">\s*[^&|]",
]

STATIC_TOOL_RISK = {
    "read_file": RISK_LOW,
    "list_dir": RISK_LOW,
    "search_code": RISK_LOW,
    "git_status": RISK_LOW,
    "git_diff": RISK_LOW,
    "git_log": RISK_LOW,
    "run_tests": RISK_LOW,
    "scan_project": RISK_LOW,
    "write_file": RISK_RISKY,
    "edit_file": RISK_RISKY,
    "git_commit": RISK_CRITICAL,
    "git_push": RISK_CRITICAL,
}


def classify_command(command: str) -> str:
    """Classify a raw shell command string into low / risky / critical."""
    for pat in CRITICAL_COMMAND_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return RISK_CRITICAL
    for pat in RISKY_COMMAND_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return RISK_RISKY
    return RISK_LOW


def assess_tool_call(fn_name: str, args: dict) -> tuple:
    """Returns (risk, short_description) for a tool call."""
    if fn_name == "run_command":
        cmd = str(args.get("command", ""))
        return classify_command(cmd), f"run_command: {cmd}"
    if fn_name == "write_file":
        return RISK_RISKY, f"write_file → {args.get('filepath', '?')}"
    if fn_name == "edit_file":
        return RISK_RISKY, f"edit_file → {args.get('filepath', '?')}"
    if fn_name == "git_commit":
        return RISK_CRITICAL, "git commit"
    if fn_name == "git_push":
        remote = args.get("remote", "origin")
        branch = args.get("branch") or "(current branch)"
        return RISK_CRITICAL, f"git push {remote} {branch}"
    risk = STATIC_TOOL_RISK.get(fn_name, RISK_LOW)
    return risk, fn_name


def needs_confirmation(mode: str, risk: str) -> bool:
    if risk == RISK_CRITICAL:
        return True
    if mode == "safe":
        return False
    if mode == "strict":
        return True
    return risk == RISK_RISKY


def _risk_tag(risk: str) -> str:
    return {
        RISK_CRITICAL: "[bold red]CRITICAL[/bold red]",
        RISK_RISKY: "[yellow]RISKY[/yellow]",
        RISK_LOW: "[dim]routine[/dim]",
    }[risk]


def _ask(risk: str) -> bool:
    try:
        if risk == RISK_CRITICAL:
            ans = Prompt.ask("[bold red]Type 'yes' to proceed, anything else cancels[/bold red]", default="no")
            return ans.strip().lower() == "yes"
        ans = Prompt.ask("Proceed? [y/N]", default="n")
        return ans.strip().lower() in ("y", "yes")
    except KeyboardInterrupt:
        console.print("\n[yellow]Action cancelled by user.[/yellow]")
        return False


def prompt_confirm(description: str, risk: str) -> bool:
    """Ask the user to approve an action. Returns True if approved."""
    console.print(f"\n[bold]⚠ confirmation required[/bold] ({_risk_tag(risk)}) — {description}")
    return _ask(risk)


def confirm_with_preview(description: str, risk: str, show_preview) -> bool:
    """Prints preview first and then invokes confirmation prompt."""
    console.print(f"\n[bold]⚠ confirmation required[/bold] ({_risk_tag(risk)}) — {description}")
    if show_preview is not None:
        show_preview()
    return _ask(risk)


# ---------------------------------------------------------------------------
# Tool functions exposed to the model
# ---------------------------------------------------------------------------
RIPGREP_AVAILABLE = shutil.which("rg") is not None
MAX_READ_BYTES = 300_000


def run_command(command: str) -> str:
    console.print(f"[bold magenta]▶ exec[/bold magenta] [dim]{command}[/dim]")
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = res.stdout if res.returncode == 0 else res.stderr
        if res.returncode == 0:
            add_xp(15, "clean execution")
        else:
            add_xp(5, "hit a bug")
        return output.strip() if output else "Command executed successfully."
    except subprocess.TimeoutExpired:
        return "Execution error: command timed out after 60s."
    except KeyboardInterrupt:
        return "Execution error: command interrupted by user (SIGINT)."
    except Exception as e:
        return f"Execution error: {str(e)}"


def read_file(filepath: str) -> str:
    try:
        path = Path(filepath)
        if not path.exists():
            return f"File read error: '{filepath}' not found."
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_READ_BYTES)
            add_xp(5, "scanned file (large — truncated)")
            console.print(f"[yellow]Note: {filepath} is {size:,} bytes; showing the first {MAX_READ_BYTES:,}.[/yellow]")
            return content + f"\n... [truncated, {size - MAX_READ_BYTES:,} more bytes omitted]"
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        add_xp(5, "scanned file")
        try:
            lexer = Syntax.guess_lexer(filepath, code=content)
            console.print(Syntax(content, lexer, theme="monokai", line_numbers=True, word_wrap=True))
        except Exception:
            pass
        return content
    except UnicodeDecodeError:
        return f"File read error: '{filepath}' does not look like a text file."
    except Exception as e:
        return f"File read error: {str(e)}"


def list_dir(path: str = ".") -> str:
    try:
        entries = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".venv")]
            depth = root.replace(path, "").count(os.sep)
            if depth > 2:
                continue
            for f in files:
                entries.append(os.path.relpath(os.path.join(root, f), path))
        listing = "\n".join(sorted(entries)[:200])
        add_xp(3, "surveyed the terrain")
        return listing or "(empty)"
    except Exception as e:
        return f"List error: {str(e)}"


def search_code(pattern: str, path: str = ".") -> str:
    try:
        if RIPGREP_AVAILABLE:
            cmd = ["rg", "-n", "--no-heading", "--color=never", "-g", "!.git", "-g", "!__pycache__", "-g", "!node_modules", pattern, path]
        else:
            cmd = ["grep", "-rn", "--exclude-dir=.git", "--exclude-dir=__pycache__", "--exclude-dir=node_modules", pattern, path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = res.stdout.strip()
        add_xp(5, "swept the codebase")
        return out[:4000] if out else "No matches."
    except FileNotFoundError:
        return "Search error: neither ripgrep nor grep is available on this system."
    except subprocess.TimeoutExpired:
        return "Search error: search timed out."
    except KeyboardInterrupt:
        return "Search error: search cancelled by user."
    except Exception as e:
        return f"Search error: {str(e)}"


def _print_diff(diff_text: str):
    if not diff_text:
        console.print("[dim](no textual difference)[/dim]")
        return
    console.print(Syntax(diff_text, "diff", theme="monokai", word_wrap=True))


def write_file(filepath: str, content: str, confirm=None) -> str:
    try:
        previous = ""
        existed = os.path.exists(filepath)
        if existed:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    previous = f.read()
            except Exception:
                previous = ""

        diff_text = unified_diff_text(previous, content, filepath)
        if confirm is not None and not confirm(diff_text):
            return "write_file cancelled by user."

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        history_push("write_file", filepath, previous if existed else "")
        _print_diff(diff_text)
        add_xp(20, "forged a new file" if not existed else "overwrote a file")
        return f"{'Created' if not existed else 'Overwrote'} {filepath} ({len(content)} bytes)."
    except Exception as e:
        return str(e)


def edit_file(filepath: str, old_text: str, new_text: str, confirm=None) -> str:
    try:
        if not os.path.exists(filepath):
            return f"Error: '{filepath}' not found."
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return "Error: exact old_text not found. Read the file again for an exact match."

        new_content = content.replace(old_text, new_text)
        diff_text = unified_diff_text(content, new_content, filepath)

        if confirm is not None and not confirm(diff_text):
            return "edit_file cancelled by user."

        history_push("edit_file", filepath, content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        _print_diff(diff_text)
        add_xp(25, "surgical edit")
        return f"Successfully updated {filepath}."
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# UI / rendering
# ---------------------------------------------------------------------------
GHOST_MASCOT = r"""[bold white]  .▄▄▄▄▄▄▄▄▄.
 ▐█  ◕   ◕  █▌
 ▐█     ▾    █▌
 ▐█▄▄▄▄▄▄▄▄▄█▌
  ╲╱ ╲╱ ╲╱ ╲╱[/bold white]"""


def render_header(stats, model_name):
    console.clear()
    rank = get_rank(stats["level"])
    threshold = stats["level"] * 100
    body = (
        f"{GHOST_MASCOT}\n\n"
        f"[bold white]G H O S T[/bold white]\n"
        f"[dim]{rank} · Level {stats['level']} · {stats['xp']}/{threshold} xp[/dim]\n"
        f"[dim]Model: {model_name} · type a task, or /help for commands[/dim]\n"
        f"[dim italic](Press Ctrl+C to halt AI responses cleanly)[/dim italic]"
    )
    console.print(Panel(Align.center(body), border_style="grey50", padding=(1, 4)))


def render_stats(stats, model_name, permission_mode, auto_test):
    table = Table(title="Ghost — Status", border_style="grey50", show_header=False)
    table.add_row("Rank", get_rank(stats["level"]))
    table.add_row("Level", str(stats["level"]))
    table.add_row("XP", f"{stats['xp']} / {stats['level']*100}")
    table.add_row("Edits made", str(stats.get("edits", 0)))
    table.add_row("Commands run", str(stats.get("commands", 0)))
    table.add_row("Active Model", str(model_name))
    table.add_row("Permission mode", permission_mode)
    table.add_row("Auto-test", "on" if auto_test else "off")
    console.print(table)


def print_help():
    console.print(
        Panel(
            "\n".join(
                [
                    "[cyan]/stats[/cyan]              — show level, xp, rank, and active model",
                    "[cyan]/rank[/cyan]               — show current rank only",
                    "[cyan]/model[/cyan]              — switch model menu, or use: [cyan]/model <id>[/cyan]",
                    "[cyan]/config[/cyan]             — reconfigure provider / API keys",
                    "[cyan]/upgrade[/cyan]            — auto-upgrade Ghost CLI from GitHub",
                    "[cyan]/mode[/cyan]               — show or set permission mode: [cyan]/mode safe|ask|strict[/cyan]",
                    "",
                    "[cyan]/plan <task>[/cyan]        — analyze the project and draft a step-by-step plan (no edits)",
                    "[cyan]/execute[/cyan]            — carry out the currently drafted plan",
                    "[cyan]/cancelplan[/cyan]         — discard the current plan",
                    "",
                    "[cyan]/git status[/cyan]         — show working tree status",
                    "[cyan]/git diff [path][/cyan]    — show unstaged/staged changes",
                    "[cyan]/git log [n][/cyan]        — show recent commits",
                    "[cyan]/git commit <msg>[/cyan]   — stage & commit (always asks to confirm)",
                    "[cyan]/git push [remote branch][/cyan] — push (always asks to confirm)",
                    "",
                    "[cyan]/test[/cyan]               — detect and run the project's test suite",
                    "[cyan]/testfix[/cyan]            — run tests, then loop analyze→fix→retest (max 3 tries)",
                    "[cyan]/autotest on|off[/cyan]    — toggle running tests automatically after edits",
                    "",
                    "[cyan]/scan[/cyan]               — quick project scan (languages, markers, file count)",
                    "[cyan]/undo [n][/cyan]           — revert the last (or last n) file edits",
                    "[cyan]/history[/cyan]            — list recent undoable edits",
                    "",
                    "[cyan]/clear[/cyan]              — clear the screen",
                    "[cyan]/help[/cyan]               — this menu",
                    "[cyan]/exit[/cyan]               — quit",
                    "[dim]Tip: Press Ctrl+C at any time to interrupt streaming responses.[/dim]",
                ]
            ),
            title="Commands",
            border_style="grey50",
        )
    )


# ---------------------------------------------------------------------------
# Testing detection / execution
# ---------------------------------------------------------------------------
def detect_python_test_cmd(root="."):
    root = Path(root)
    has_pytest = shutil.which("pytest") is not None
    markers = (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() or (root / "setup.cfg").exists()
    has_test_files = any(root.rglob("test_*.py")) or any(root.rglob("*_test.py"))
    if not markers and not has_test_files:
        return None
    if has_pytest:
        return ["pytest", "-q"]
    if has_test_files:
        return ["python3", "-m", "unittest", "discover", "-q"]
    return None


def detect_js_test_cmd(root="."):
    pkg = Path(root) / "package.json"
    if not pkg.exists():
        return None
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except Exception:
        return None
    scripts = data.get("scripts", {})
    test_script = scripts.get("test", "")
    if not test_script or "no test specified" in test_script.lower():
        return None
    if (Path(root) / "yarn.lock").exists() and shutil.which("yarn"):
        return ["yarn", "test"]
    if (Path(root) / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
        return ["pnpm", "test"]
    if shutil.which("npm"):
        return ["npm", "test", "--silent"]
    return None


def detect_test_commands(root="."):
    cmds = []
    py = detect_python_test_cmd(root)
    if py:
        cmds.append(("python", py))
    js = detect_js_test_cmd(root)
    if js:
        cmds.append(("javascript", js))
    return cmds


def _run_one(lang, cmd, root, timeout):
    try:
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
        ok = res.returncode == 0
        output = (res.stdout + "\n" + res.stderr).strip()
        return {"lang": lang, "cmd": cmd, "success": ok, "output": output[-4000:]}
    except FileNotFoundError:
        return {"lang": lang, "cmd": cmd, "success": False, "output": f"{cmd[0]} not found on PATH."}
    except subprocess.TimeoutExpired:
        return {"lang": lang, "cmd": cmd, "success": False, "output": f"Test run timed out after {timeout}s."}
    except KeyboardInterrupt:
        return {"lang": lang, "cmd": cmd, "success": False, "output": "Test run interrupted by user."}
    except Exception as e:
        return {"lang": lang, "cmd": cmd, "success": False, "output": f"Error running tests: {e}"}


def run_tests(root=".", timeout=120):
    cmds = detect_test_commands(root)
    if not cmds:
        return {"ran": False, "success": None, "results": [], "output": "No Python or JavaScript test framework detected."}

    results = []
    if len(cmds) > 1:
        with ThreadPoolExecutor(max_workers=len(cmds)) as pool:
            futures = [pool.submit(_run_one, lang, cmd, root, timeout) for lang, cmd in cmds]
            results = [f.result() for f in futures]
    else:
        lang, cmd = cmds[0]
        results = [_run_one(lang, cmd, root, timeout)]

    overall_ok = all(r["success"] for r in results)
    return {"ran": True, "success": overall_ok, "results": results}


def format_results(result: dict) -> str:
    if not result["ran"]:
        return result["output"]
    lines = []
    for r in result["results"]:
        status = "PASS" if r["success"] else "FAIL"
        lines.append(f"[{r['lang']}] {status} — {' '.join(r['cmd'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI-compatible client holder
# ---------------------------------------------------------------------------
class ClientHolder:
    def __init__(self):
        self.cfg = {}
        self.client = None
        self.model_name = DEFAULT_MODEL
        self.permission_mode = DEFAULT_PERMISSION_MODE
        self.auto_test = False

    def init(self):
        """Explicit startup step so importing the app never triggers network setup."""
        ensure_data_dir()
        self.cfg = get_api_config(load_config())
        self._apply_cfg()

    def _apply_cfg(self):
        self.model_name = self.cfg.get("model", DEFAULT_MODEL)
        self.permission_mode = self.cfg.get("permission_mode", DEFAULT_PERMISSION_MODE)
        self.auto_test = bool(self.cfg.get("auto_test", False))
        self.client = OpenAI(
            base_url=self.cfg.get("base_url"),
            api_key=self.cfg.get("api_key"),
            default_headers={"X-Title": "Ghost"},
        )

    def reconfigure(self):
        self.cfg = get_api_config({})
        self._apply_cfg()

    def switch_model(self, target_model: str = None):
        if not target_model:
            console.print("\n[bold cyan]Available Models:[/bold cyan]")
            for key, (m_id, desc) in MODEL_PRESETS.items():
                active_marker = " [bold green](active)[/bold green]" if m_id == self.model_name else ""
                console.print(f"[cyan]{key}.[/cyan] {m_id} — [dim]{desc}[/dim]{active_marker}")
            console.print("[cyan]6.[/cyan] Custom model identifier")

            choice = Prompt.ask(
                "\n[bold white]Choose model (1-6 or enter full model ID)[/bold white]",
                default="1",
            ).strip()

            if choice in MODEL_PRESETS:
                target_model = MODEL_PRESETS[choice][0]
            elif choice == "6":
                target_model = Prompt.ask("[bold white]Enter model ID (e.g. anthropic/claude-3.5-sonnet)[/bold white]").strip()
            else:
                target_model = choice

        self.model_name = target_model
        self.cfg["model"] = self.model_name
        save_config(self.cfg)
        console.print(f"\n[bold green]✔ Model updated to:[/bold green] [bold cyan]{self.model_name}[/bold cyan]\n")

    def set_permission_mode(self, mode: str) -> bool:
        mode = mode.strip().lower()
        if mode not in PERMISSION_MODES:
            return False
        self.permission_mode = mode
        self.cfg["permission_mode"] = mode
        save_config(self.cfg)
        return True

    def set_auto_test(self, enabled: bool):
        self.auto_test = enabled
        self.cfg["auto_test"] = enabled
        save_config(self.cfg)


HOLDER = ClientHolder()


# ---------------------------------------------------------------------------
# Agent loop / tool dispatch / streaming
# ---------------------------------------------------------------------------
MAX_CONTEXT_MESSAGES = 60

SYSTEM_PROMPT = """You are Ghost — a highly skilled, senior-level programmer who works fast and says little.
Voice:
- Calm, precise, understated confidence. No hype, no shouting.
- Short, sharp commentary. You explain *why* a decision was made, not just *what* you did.
- If the user's code has an obvious flaw, name it plainly and fix it.
- You favor clean, minimal, well-tested code over clever code.
- You never leave debris: no __pycache__, no stray temp files, no dead code.
- Before editing a file you haven't seen this session, read it first.
- When a task is ambiguous, ask one sharp clarifying question instead of guessing.

You have real tools: run_command, read_file, list_dir, search_code, write_file,
edit_file, git_status, git_diff, git_log, git_commit, git_push, run_tests,
scan_project — use them instead of describing what you would do.

Risky actions (writes, shell commands, git commit/push) may prompt the user
for confirmation before they execute — if one is declined, accept it and
adjust course rather than repeating the same call.
"""

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "run_command", "description": "Execute a bash/terminal command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file's contents.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files in a directory (recursive, depth-limited).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Search for a pattern across files (uses ripgrep when available).",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create a new file, or fully overwrite an existing one, with given content.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filepath", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Find-and-replace a text block in a file. Read the file first for an exact match.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["filepath", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "git_status", "description": "Show the git working tree status.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "git_diff", "description": "Show unstaged/staged git changes, optionally scoped to a path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "git_log", "description": "Show recent commit history.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "git_commit", "description": "Stage all changes and commit with a message. Always asks the user to confirm first.",
        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    {"type": "function", "function": {
        "name": "git_push", "description": "Push commits to a remote. Always asks the user to confirm first.",
        "parameters": {"type": "object", "properties": {"remote": {"type": "string"}, "branch": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Detect and run the project's Python/JavaScript test suite.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "scan_project", "description": "Quick scan of the project: languages, file count, key markers (cached).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

PLANNING_TOOL_DEFINITIONS = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in READ_ONLY_TOOLS]


def _run_tests_tool() -> str:
    result = run_tests(".")
    if not result["ran"]:
        return result["output"]
    summary = format_results(result)
    if not result["success"]:
        failing = [r for r in result["results"] if not r["success"]]
        detail = "\n\n".join(f"--- {r['lang']} output ---\n{r['output']}" for r in failing)
        return f"{summary}\n\n{detail}"[:6000]
    return summary


def _scan_project_tool() -> str:
    result, cached = get_cached_scan(".", scan_project)
    lines = [f"Root: {result['root']} (cache {'hit' if cached else 'miss'})",
             f"Files: {result['file_count']}"]
    if result["languages"]:
        lines.append("Languages: " + ", ".join(f"{k} ({v})" for k, v in result["languages"].items()))
    markers = [k for k, v in result["markers"].items() if v]
    if markers:
        lines.append("Markers: " + ", ".join(markers))
    lines.append("Top-level: " + ", ".join(result["top_level"][:20]))
    return "\n".join(lines)


TOOL_MAP = {
    "run_command": run_command,
    "read_file": read_file,
    "list_dir": list_dir,
    "search_code": search_code,
    "write_file": write_file,
    "edit_file": edit_file,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_commit": git_commit,
    "git_push": git_push,
    "run_tests": _run_tests_tool,
    "scan_project": _scan_project_tool,
}


def _build_confirm(risk, description):
    def _confirm(diff_text=""):
        if not needs_confirmation(HOLDER.permission_mode, risk):
            return True
        preview = None
        if diff_text:
            preview = lambda: console.print(Syntax(diff_text, "diff", theme="monokai", word_wrap=True))
        return confirm_with_preview(description, risk, preview)
    return _confirm


def dispatch_tool_call(tool_call, planning_restricted=False, edited=None):
    fn_name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except Exception:
        args = {}

    if planning_restricted and fn_name not in READ_ONLY_TOOLS:
        return f"'{fn_name}' is unavailable in planning mode (read-only tools only)."

    risk, description = assess_tool_call(fn_name, args)

    if fn_name in ("write_file", "edit_file"):
        args = dict(args)
        args["confirm"] = _build_confirm(risk, description)
    elif needs_confirmation(HOLDER.permission_mode, risk):
        if not prompt_confirm(description, risk):
            return f"{fn_name} cancelled by user."

    fn = TOOL_MAP.get(fn_name)
    if fn is None:
        return f"Unknown tool: {fn_name}"

    try:
        result = fn(**args)
    except TypeError as e:
        return f"Tool '{fn_name}' called with bad arguments: {e}"
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return f"Tool '{fn_name}' raised an error: {e}"

    bump("commands", 1)
    if fn_name in ("edit_file", "write_file") and isinstance(result, str) and (
        result.startswith("Successfully") or result.startswith("Created") or result.startswith("Overwrote")
    ):
        bump("edits", 1)
        if edited is not None:
            edited.append(args.get("filepath"))

    return result


def stream_completion(messages, tool_definitions):
    text_parts = []
    tool_call_parts = {}
    printed_header = False
    status = console.status("[cyan]thinking[/cyan]", spinner="dots")
    status.start()

    # Pass NVIDIA-specific parameters when connecting to integrate.api.nvidia.com
    extra_params = {}
    base_url = str(HOLDER.cfg.get("base_url", "")).lower()
    if "nvidia.com" in base_url:
        extra_params["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        extra_params["temperature"] = 1
        extra_params["top_p"] = 0.95

    try:
        stream = HOLDER.client.chat.completions.create(
            model=HOLDER.model_name,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
            stream=True,
            **extra_params,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if not printed_header:
                    status.stop()
                    console.print(Rule(style="grey50"))
                    console.print(f"[bold white]👻 ghost[/bold white] [dim]({HOLDER.model_name})[/dim]")
                    printed_header = True
                console.print(delta.content, end="")
                text_parts.append(delta.content)

            if delta.tool_calls:
                if not printed_header:
                    status.stop()
                    printed_header = True
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    slot = tool_call_parts.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    if tc_delta.id:
                        slot["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            slot["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            slot["arguments"] += tc_delta.function.arguments
    except KeyboardInterrupt:
        status.stop()
        console.print()
        raise
    finally:
        status.stop()

    if text_parts:
        console.print()
        console.print(Rule(style="grey50"))

    tool_calls = None
    if tool_call_parts:
        tool_calls = [
            SimpleNamespace(id=slot["id"], function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]))
            for _, slot in sorted(tool_call_parts.items())
        ]
    return SimpleNamespace(content="".join(text_parts) or None, tool_calls=tool_calls)


def run_turn(messages, tool_definitions, planning_restricted=False, edited=None):
    """Streams one assistant turn, running any tool calls, until the model replies with plain text."""
    while True:
        try:
            response_msg = stream_completion(messages, tool_definitions)
        except KeyboardInterrupt:
            console.print("\n[yellow]▲ Generation halted by user (Ctrl+C).[/yellow]")
            return False
        except Exception as e:
            console.print(f"[bold red]API error:[/bold red] {e}")
            return False

        if response_msg is None:
            return False

        if response_msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in response_msg.tool_calls
                ],
            })
            for tool_call in response_msg.tool_calls:
                try:
                    tool_output = dispatch_tool_call(tool_call, planning_restricted, edited)
                except KeyboardInterrupt:
                    console.print("\n[yellow]▲ Tool execution cancelled by user (Ctrl+C).[/yellow]")
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Tool execution interrupted by user (Ctrl+C)."})
                    return False

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(tool_output)})
        else:
            messages.append({"role": "assistant", "content": response_msg.content})
            return True


def trim_messages(messages):
    """Keep context bounded; only ever called at a safe top-level boundary."""
    if len(messages) <= MAX_CONTEXT_MESSAGES + 1:
        return messages
    system = messages[0]
    tail = messages[1:][-MAX_CONTEXT_MESSAGES:]
    for i, m in enumerate(tail):
        if m.get("role") == "user":
            tail = tail[i:]
            break
    return [system] + tail


def run_test_fix_loop(messages, initial_result=None, max_retries=3):
    result = initial_result
    try:
        for attempt in range(1, max_retries + 1):
            if result is None:
                console.print(f"[dim]Re-running tests (attempt {attempt}/{max_retries})...[/dim]")
                result = run_tests(".")
            if not result["ran"]:
                console.print(f"[dim]{result['output']}[/dim]")
                return
            if result["success"]:
                console.print(f"[bold green]✔ Tests passing[/bold green] (attempt {attempt}/{max_retries}).")
                add_xp(20, "green build")
                return
            console.print(f"[yellow]✖ Tests failing — attempt {attempt}/{max_retries}[/yellow]")
            console.print(format_results(result))
            if attempt == max_retries:
                console.print("[red]Max retries reached. Leaving for manual review.[/red]")
                return
            failing = [r for r in result["results"] if not r["success"]]
            failure_text = "\n\n".join(f"[{r['lang']}] {' '.join(r['cmd'])}\n{r['output']}" for r in failing)
            messages.append({
                "role": "user",
                "content": (
                    "The test suite is failing:\n\n" + failure_text[:4000] +
                    "\n\nAnalyze the failure and fix the code using your tools. "
                    "Make the fix directly; keep commentary short."
                ),
            })
            ok = run_turn(messages, TOOL_DEFINITIONS)
            if not ok:
                console.print("[red]Fix attempt aborted.[/red]")
                return
            result = None
    except KeyboardInterrupt:
        console.print("\n[yellow]▲ Test-fix loop cancelled by user (Ctrl+C).[/yellow]")


def maybe_autotest(messages, edited):
    if not HOLDER.auto_test or not edited:
        return
    console.print("\n[dim]auto-test: running suite after edits...[/dim]")
    result = run_tests(".")
    if not result["ran"]:
        console.print(f"[dim]{result['output']}[/dim]")
        return
    console.print(format_results(result))
    if result["success"]:
        add_xp(20, "clean test run")
        return
    try:
        ans = Prompt.ask("[yellow]Tests are failing. Attempt an automatic fix loop (up to 3 tries)? [y/N][/yellow]", default="n")
        if ans.strip().lower() in ("y", "yes"):
            run_test_fix_loop(messages, initial_result=result)
    except KeyboardInterrupt:
        console.print("\n[yellow]Auto-fix cancelled.[/yellow]")


def _print_scan(result, cached):
    table = Table(title=f"Project scan{' (cached)' if cached else ''}", border_style="grey50", show_header=False)
    table.add_row("Root", result["root"])
    table.add_row("Files", str(result["file_count"]))
    table.add_row("Languages", ", ".join(f"{k} ({v})" for k, v in result["languages"].items()) or "(none detected)")
    markers = [k for k, v in result["markers"].items() if v]
    table.add_row("Markers", ", ".join(markers) or "(none)")
    console.print(table)


def handle_slash_command(user_input, messages):
    """Returns 'exit' to quit, True if the input was a handled command, False otherwise."""
    low = user_input.lower().strip()

    if low in ("exit", "quit", "q", "/exit"):
        flush_stats()
        console.print("[dim]Ghost fades out.[/dim]")
        return "exit"

    if low == "/upgrade":
        upgrade_ghost()
        return True

    if low == "/stats":
        render_stats(load_stats(), HOLDER.model_name, HOLDER.permission_mode, HOLDER.auto_test)
        return True

    if low == "/rank":
        s = load_stats()
        console.print(f"[cyan]{get_rank(s['level'])}[/cyan]")
        return True

    if low == "/model" or low.startswith("/model "):
        parts = user_input.split(maxsplit=1)
        HOLDER.switch_model(parts[1].strip() if len(parts) > 1 else None)
        return True

    if low == "/config":
        HOLDER.reconfigure()
        render_header(load_stats(), HOLDER.model_name)
        return True

    if low.startswith("/mode"):
        parts = user_input.split()
        if len(parts) < 2:
            console.print(f"[dim]Permission mode: {HOLDER.permission_mode} (safe / ask / strict). Usage: /mode <mode>[/dim]")
        elif HOLDER.set_permission_mode(parts[1]):
            console.print(f"[green]Permission mode set to {HOLDER.permission_mode}.[/green]")
        else:
            console.print("[red]Invalid mode. Choose: safe, ask, strict.[/red]")
        return True

    if low.startswith("/plan"):
        task = user_input[len("/plan"):].strip()
        if not task:
            console.print("[yellow]Usage: /plan <task description>[/yellow]")
            return True
        PLAN_STATE.start(task)
        messages.append({"role": "system", "content": PLANNING_SYSTEM_SUFFIX})
        messages.append({"role": "user", "content": f"Create a plan for: {task}"})
        ok = run_turn(messages, PLANNING_TOOL_DEFINITIONS, planning_restricted=True)
        if ok:
            plan_text = messages[-1].get("content") or "(no plan text returned)"
            PLAN_STATE.finish_drafting(plan_text)
            console.print(Panel(plan_text, title="Proposed Plan", border_style="cyan"))
            console.print("[dim]Type /execute to run this plan, or /cancelplan to discard it.[/dim]")
        else:
            PLAN_STATE.clear()
        flush_stats()
        return True

    if low == "/execute":
        if not PLAN_STATE.has_plan():
            console.print("[yellow]No plan to execute. Use /plan <task> first.[/yellow]")
            return True
        plan_text = PLAN_STATE.plan_text
        messages.append({"role": "user", "content": f"Execute this approved plan step by step, using your tools:\n\n{plan_text}"})
        edited = []
        ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
        if ok:
            maybe_autotest(messages, edited)
        PLAN_STATE.clear()
        flush_stats()
        return True

    if low == "/cancelplan":
        if PLAN_STATE.has_plan() or PLAN_STATE.active:
            PLAN_STATE.clear()
            console.print("[cyan]Plan discarded.[/cyan]")
        else:
            console.print("[dim]No active plan.[/dim]")
        return True

    if low.startswith("/git"):
        parts = user_input.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub == "status":
            console.print(git_status())
        elif sub == "diff":
            console.print(git_diff(parts[2] if len(parts) > 2 else None))
        elif sub == "log":
            n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 10
            console.print(git_log(n))
        elif sub == "commit":
            msg = parts[2] if len(parts) > 2 else ""
            if not msg:
                console.print("[yellow]Usage: /git commit <message>[/yellow]")
            elif prompt_confirm(f'git commit -m "{msg}"', RISK_CRITICAL):
                console.print(git_commit(msg))
                add_xp(15, "sealed a commit")
            else:
                console.print("[dim]Commit cancelled.[/dim]")
        elif sub == "push":
            rest = parts[2].split() if len(parts) > 2 else []
            remote = rest[0] if rest else "origin"
            branch = rest[1] if len(rest) > 1 else None
            if prompt_confirm(f"git push {remote} {branch or '(current branch)'}", RISK_CRITICAL):
                console.print(git_push(remote, branch))
            else:
                console.print("[dim]Push cancelled.[/dim]")
        else:
            console.print("[yellow]Usage: /git status|diff [path]|log [n]|commit <msg>|push [remote branch][/yellow]")
        flush_stats()
        return True

    if low == "/test":
        result = run_tests(".")
        if not result["ran"]:
            console.print(f"[dim]{result['output']}[/dim]")
        else:
            console.print(format_results(result))
            if result["success"]:
                add_xp(20, "clean test run")
        flush_stats()
        return True

    if low == "/testfix":
        run_test_fix_loop(messages)
        flush_stats()
        return True

    if low.startswith("/autotest"):
        parts = user_input.split()
        if len(parts) < 2 or parts[1].lower() not in ("on", "off"):
            console.print(f"[dim]Auto-test is currently {'on' if HOLDER.auto_test else 'off'}. Usage: /autotest on|off[/dim]")
        else:
            HOLDER.set_auto_test(parts[1].lower() == "on")
            console.print(f"[green]Auto-test turned {parts[1].lower()}.[/green]")
        return True

    if low == "/scan":
        result, cached = get_cached_scan(".", scan_project)
        _print_scan(result, cached)
        return True

    if low.startswith("/undo"):
        parts = user_input.split()
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        reverted, err = history_undo(n)
        if reverted:
            console.print(f"[cyan]Reverted {len(reverted)} edit(s): {', '.join(reverted)}[/cyan]")
        if err:
            color = "yellow" if err == "Nothing to undo." else "red"
            console.print(f"[{color}]{err}[/{color}]")
        return True

    if low == "/history":
        entries = history_list_recent(10)
        if not entries:
            console.print("[dim]No edit history yet.[/dim]")
        else:
            table = Table(title="Recent edits (most recent first)", border_style="grey50")
            table.add_column("#")
            table.add_column("Action")
            table.add_column("File")
            table.add_column("When")
            for i, e in enumerate(entries, 1):
                when = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
                table.add_row(str(i), e["action"], e["filepath"], when)
            console.print(table)
        return True

    if low == "/clear":
        render_header(load_stats(), HOLDER.model_name)
        return True

    if low == "/help":
        print_help()
        return True

    return False


def run_agent():
    HOLDER.init()
    render_header(load_stats(), HOLDER.model_name)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        messages = trim_messages(messages)
        try:
            user_input = Prompt.ask("\n[bold white]you[/bold white]").strip()
        except KeyboardInterrupt:
            console.print("\n[dim](Press Ctrl+D or type /exit to quit)[/dim]")
            continue
        except EOFError:
            flush_stats()
            console.print("\n[dim]Ghost fades out.[/dim]")
            break

        if not user_input:
            continue

        try:
            result = handle_slash_command(user_input, messages)
        except KeyboardInterrupt:
            console.print("\n[yellow]▲ Command halted by user (Ctrl+C).[/yellow]")
            continue

        if result == "exit":
            break
        if result:
            continue

        pre_len = len(messages)
        messages.append({"role": "user", "content": user_input})

        edited = []
        ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
        if not ok:
            del messages[pre_len:]
        else:
            maybe_autotest(messages, edited)

        flush_stats()


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\nGhost fades out.")
