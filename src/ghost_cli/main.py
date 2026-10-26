"""Single-file version of the Ghost CLI v2.3.0. This preserves the original modular
behavior while keeping the entire application in one entrypoint.
"""

__version__ = "2.3.0"

import ast
import difflib
import hashlib
import json
import logging
import os
import platform
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional tiktoken for accurate token counting
try:
    import tiktoken as _tiktoken
    _TK_ENC = _tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        try:
            return len(_TK_ENC.encode(text))
        except Exception:
            return max(1, len(text) // 4)
except ImportError:
    _tiktoken = None
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
SESSIONS_DIR = DATA_DIR / "sessions"
CONFIG_FILE = DATA_DIR / "ghost_config.json"
STATS_FILE = DATA_DIR / "ghost_stats.json"
UNDO_FILE = DATA_DIR / ".ghost_undo.json"
HISTORY_FILE = DATA_DIR / "ghost_history.json"
CACHE_FILE = DATA_DIR / "ghost_cache.json"
CHECKPOINT_FILE = DATA_DIR / "ghost_checkpoints.json"
GHOST_MD_FILE = Path("GHOST.md")

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
GITHUB_REPO_URL = "git+[https://github.com/ThunderBoltX7/ghost](https://github.com/ThunderBoltX7/ghost)"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg):
    try:
        ensure_data_dir()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Terminal UI (Quiet by default, Verbose optional)
# ---------------------------------------------------------------------------
console = Console()

DOT_SYM = "[•]"
CHECK_SYM = "[✓]"
CROSS_SYM = "[✗]"
WARN_SYM = "[!]"


class TerminalUI:

    def __init__(self, console_instance: Console):
        self.console = console_instance
        self.verbose = False

    def set_verbose(self, enabled: bool):
        self.verbose = bool(enabled)

    def status(self, message: str):
        self.console.print(f"[dim cyan]{DOT_SYM}[/dim cyan] {message}")

    def success(self, message: str):
        self.console.print(f"[bold green]{CHECK_SYM}[/bold green] {message}")

    def warn(self, message: str):
        self.console.print(f"[bold yellow]{WARN_SYM}[/bold yellow] {message}")

    def error(self, message: str):
        self.console.print(f"[bold red]{CROSS_SYM}[/bold red] {message}")

    def detail(self, renderable: Any):
        if self.verbose:
            self.console.print(renderable)


UI = TerminalUI(console)


# ---------------------------------------------------------------------------
# Interactive setup & configuration
# ---------------------------------------------------------------------------
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
    console.print("[cyan]2.[/cyan] OmniRoute Gateway")
    console.print("[cyan]3.[/cyan] OpenAI")
    console.print("[cyan]4.[/cyan] Groq")
    console.print("[cyan]5.[/cyan] DeepSeek")
    console.print("[cyan]6.[/cyan] NVIDIA NIM")
    console.print("[cyan]7.[/cyan] Custom (Any OpenAI-compatible endpoint)")

    choice = Prompt.ask("\n[bold white]Enter choice (1-7)[/bold white]", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")

    if choice == "2":
        provider_name = "OmniRoute"
        base_url = Prompt.ask(
            "[bold white]Enter OmniRoute Base URL[/bold white]",
            default="http://localhost:20128/v1",
        ).strip().rstrip("/")
        default_model = Prompt.ask(
            "[bold white]Enter default model name[/bold white]",
            default="auto",
        ).strip()
    elif choice == "7":
        provider_name = "Custom"
        base_url = Prompt.ask("[bold white]Enter Base URL (e.g., http://localhost:11434/v1)[/bold white]").strip().rstrip("/")
        default_model = Prompt.ask("[bold white]Enter default model name[/bold white]", default="gpt-4").strip()
    else:
        providers = {
            "1": ("OpenRouter", "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)", "deepseek/deepseek-v3.2"),
            "3": ("OpenAI", "[https://api.openai.com/v1](https://api.openai.com/v1)", "gpt-4o"),
            "4": ("Groq", "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)", "llama-3.3-70b-versatile"),
            "5": ("DeepSeek", "[https://api.deepseek.com](https://api.deepseek.com)", "deepseek-chat"),
            "6": ("NVIDIA", "[https://integrate.api.nvidia.com/v1](https://integrate.api.nvidia.com/v1)", "deepseek-ai/deepseek-v4-pro-0813"),
        }
        provider_name, base_url, default_model = providers[choice]

    while True:
        new_key = Prompt.ask(
            f"\n[bold white]Enter your {provider_name} API Key[/bold white]",
            password=True,
            default="omniroute-default" if choice == "2" else ...,
        ).strip()

        if len(new_key) >= 1:
            cfg["provider"] = provider_name
            cfg["base_url"] = base_url
            cfg["api_key"] = new_key
            cfg["model"] = default_model
            cfg.setdefault("fast_model", default_model)
            cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
            cfg.setdefault("auto_test", False)
            cfg.setdefault("verbose", False)

            save_config(cfg)

            console.print(f"\n[bold green]✔ {provider_name} configured and saved securely to {CONFIG_FILE}[/bold green]")
            console.print("[dim]Booting system...[/dim]\n")
            return cfg
        else:
            console.print("[red]Invalid key length. Try again.[/red]")


def get_api_config(cfg):
    omni_key = os.environ.get("OMNIROUTE_API_KEY")
    if omni_key and "api_key" not in cfg:
        cfg["api_key"] = omni_key
        cfg["base_url"] = os.environ.get("OMNIROUTE_BASE_URL", "http://localhost:20128/v1").rstrip("/")
        cfg["model"] = os.environ.get("GHOST_MODEL", "auto")
        cfg["provider"] = "OmniRoute"
        cfg.setdefault("fast_model", os.environ.get("GHOST_FAST_MODEL", cfg["model"]))
        cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
        cfg.setdefault("auto_test", False)
        cfg.setdefault("verbose", False)
        return cfg

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
            cfg["base_url"] = "[https://integrate.api.nvidia.com/v1](https://integrate.api.nvidia.com/v1)"
            cfg["model"] = "deepseek-ai/deepseek-v4-pro-0813"
            cfg["provider"] = "NVIDIA NIM"
        else:
            cfg["base_url"] = os.environ.get("OPENAI_BASE_URL", "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)")
            cfg["model"] = os.environ.get("GHOST_MODEL", DEFAULT_MODEL)
            cfg["provider"] = "Environment Variable"

    if not cfg.get("api_key"):
        cfg = interactive_setup(cfg)

    cfg.setdefault("fast_model", os.environ.get("GHOST_FAST_MODEL", cfg.get("model", DEFAULT_MODEL)))
    cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
    cfg.setdefault("auto_test", False)
    cfg.setdefault("verbose", False)
    return cfg


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
            with open(STATS_FILE, "r", encoding="utf-8") as f:
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
        with open(STATS_FILE, "w", encoding="utf-8") as f:
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
    if UI.verbose and reason:
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
# Diffing & Syntax Validation
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


def validate_syntax(filepath: str, content: Optional[str] = None) -> Tuple[bool, str]:
    """Validates code syntax using standard library parsers."""
    path = Path(filepath)
    if content is None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return False, f"Could not read file for validation: {e}"

    ext = path.suffix.lower()

    if ext == ".py":
        try:
            ast.parse(content, filename=filepath)
            compile(content, filepath, "exec")
            return True, ""
        except SyntaxError as e:
            line_txt = e.text.strip() if e.text else ""
            return False, f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  Line {e.lineno}: {line_txt}"
        except Exception as e:
            return False, f"Python compile error: {e}"

    if ext == ".json":
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSONDecodeError at line {e.lineno}, col {e.colno}: {e.msg}"

    return True, ""


# ---------------------------------------------------------------------------
# Lightweight Codebase Intelligence (In-Memory AST & Regex Index)
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


class SymbolKind:
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    INTERFACE = "interface"
    STRUCT = "struct"


class SymbolItem:
    def __init__(
        self,
        name: str,
        kind: str,
        filepath: str,
        line: int = 1,
        col: int = 0,
        signature: str = "",
        docstring: str = "",
        parent: Optional[str] = None,
        bases: Optional[List[str]] = None,
    ):
        self.name = name
        self.kind = kind
        self.filepath = filepath
        self.line = line
        self.col = col
        self.signature = signature
        self.docstring = docstring
        self.parent = parent
        self.bases = bases or []

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "kind": self.kind,
            "filepath": self.filepath,
            "line": self.line,
        }
        if self.signature:
            d["signature"] = self.signature
        if self.parent:
            d["parent"] = self.parent
        if self.bases:
            d["bases"] = self.bases
        if self.docstring:
            d["docstring"] = self.docstring[:120]
        return d


class PythonAstIndexer(ast.NodeVisitor):
    """Deep Python AST visitor extracting classes, functions, methods, imports, and references."""

    def __init__(self, filepath: str, source_lines: List[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.symbols: List[SymbolItem] = []
        self.imports: List[Dict[str, Any]] = []
        self.references: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute) and hasattr(ast, "unparse"):
                bases.append(ast.unparse(b))
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)

        doc = ast.get_docstring(node) or ""
        sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        sym = SymbolItem(
            name=node.name,
            kind=SymbolKind.CLASS,
            filepath=self.filepath,
            line=node.lineno,
            col=node.col_offset,
            signature=sig,
            docstring=doc,
            parent=self.current_class,
            bases=bases,
        )
        self.symbols.append(sym)

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_func(node, is_async=True)

    def _handle_func(self, node: Any, is_async: bool = False):
        doc = ast.get_docstring(node) or ""
        kind = SymbolKind.METHOD if self.current_class else SymbolKind.FUNCTION

        args_list = []
        for a in node.args.args:
            args_list.append(a.arg)
        if node.args.vararg:
            args_list.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args_list.append(f"**{node.args.kwarg.arg}")

        prefix = "async def " if is_async else "def "
        sig = f"{prefix}{node.name}({', '.join(args_list)})"
        if getattr(node, "returns", None) and hasattr(ast, "unparse"):
            try:
                sig += f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        sym = SymbolItem(
            name=node.name,
            kind=kind,
            filepath=self.filepath,
            line=node.lineno,
            col=node.col_offset,
            signature=sig,
            docstring=doc,
            parent=self.current_class,
        )
        self.symbols.append(sym)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.append({"module": alias.name, "name": name, "line": node.lineno})
            self.symbols.append(
                SymbolItem(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    filepath=self.filepath,
                    line=node.lineno,
                    signature=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.append({"module": mod, "name": alias.name, "asname": alias.asname, "line": node.lineno})
            self.symbols.append(
                SymbolItem(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    filepath=self.filepath,
                    line=node.lineno,
                    signature=f"from {mod} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.references.append({"symbol": node.id, "filepath": self.filepath, "line": node.lineno})
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self.references.append({"symbol": node.attr, "filepath": self.filepath, "line": node.lineno})
        self.generic_visit(node)


def _python_search_code(pattern: str, root: str = ".", max_matches: int = 50) -> str:
    """Native Python regex search fallback when neither rg nor grep is installed."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    matches = []
    root_path = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz", ".whl", ".exe", ".bin", ".pyc"}:
                continue
            fp = Path(dirpath) / fn
            try:
                rel = str(fp.relative_to(root_path))
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{rel}:{line_no}:{line.strip()}")
                            if len(matches) >= max_matches:
                                break
            except Exception:
                continue
            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    return "\n".join(matches) if matches else "No matches."


def _python_search_code_structured(pattern: str, root: str = ".") -> List[Dict[str, Any]]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    results = []
    root_path = Path(root)
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in _EXT_LANG or ext in (".json", ".md", ".toml", ".yaml", ".yml", ".txt"):
                fp = Path(dirpath) / fn
                try:
                    rel = str(fp.relative_to(root_path))
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append({
                                    "symbol": pattern,
                                    "filepath": rel,
                                    "line": line_no,
                                    "context": line.strip()[:140],
                                })
                                if len(results) >= 50:
                                    return results
                except Exception:
                    continue
    return results


class CodebaseIndex:
    """Maintains an in-memory project index with selective incremental invalidation."""

    def __init__(self, root: str = "."):
        self.root = Path(root).resolve()
        self.symbols: Dict[str, List[SymbolItem]] = {}
        self.definitions: Dict[str, List[SymbolItem]] = {}
        self.references: Dict[str, List[Dict[str, Any]]] = {}
        self.imports: Dict[str, List[Dict[str, Any]]] = {}
        self.classes: Dict[str, List[SymbolItem]] = {}
        self.file_symbols: Dict[str, List[SymbolItem]] = {}
        self.file_mtimes: Dict[str, float] = {}
        self.initialized = False

    def build_index(self, max_files: int = 2000):
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in _EXT_LANG:
                    fp = os.path.relpath(os.path.join(dirpath, fn), str(self.root))
                    self.index_file(fp)
                    count += 1
                    if count >= max_files:
                        break
            if count >= max_files:
                break
        self.initialized = True

    def ensure_indexed(self):
        if not self.initialized:
            self.build_index()

    def remove_file(self, filepath: str):
        norm_path = os.path.normpath(filepath)
        old_symbols = self.file_symbols.pop(norm_path, [])
        for sym in old_symbols:
            key = sym.name.lower()
            if key in self.symbols:
                self.symbols[key] = [s for s in self.symbols[key] if os.path.normpath(s.filepath) != norm_path]
                if not self.symbols[key]:
                    del self.symbols[key]
            if key in self.definitions:
                self.definitions[key] = [s for s in self.definitions[key] if os.path.normpath(s.filepath) != norm_path]
                if not self.definitions[key]:
                    del self.definitions[key]
            if key in self.classes:
                self.classes[key] = [s for s in self.classes[key] if os.path.normpath(s.filepath) != norm_path]
                if not self.classes[key]:
                    del self.classes[key]

        self.imports.pop(norm_path, None)
        self.file_mtimes.pop(norm_path, None)
        for sym_name in list(self.references.keys()):
            self.references[sym_name] = [
                r for r in self.references[sym_name] if os.path.normpath(r.get("filepath", "")) != norm_path
            ]
            if not self.references[sym_name]:
                del self.references[sym_name]

    def index_file(self, filepath: str):
        norm_path = os.path.normpath(filepath)
        full_path = self.root / norm_path if not os.path.isabs(filepath) else Path(filepath)
        if not full_path.exists() or not full_path.is_file():
            self.remove_file(norm_path)
            return

        try:
            mtime = full_path.stat().st_mtime
            if norm_path in self.file_mtimes and self.file_mtimes[norm_path] == mtime:
                return
        except Exception:
            return

        self.remove_file(norm_path)

        try:
            size = full_path.stat().st_size
            if size > 500_000:
                return
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return

        self.file_mtimes[norm_path] = mtime
        ext = full_path.suffix.lower()

        if ext == ".py":
            self._index_python(norm_path, content)
        else:
            self._index_regex(norm_path, content, ext)

    def _index_python(self, filepath: str, content: str):
        lines = content.splitlines()
        try:
            tree = ast.parse(content, filename=filepath)
            indexer = PythonAstIndexer(filepath, lines)
            indexer.visit(tree)
            symbols = indexer.symbols
            imports = indexer.imports
            refs = indexer.references
        except Exception:
            self._index_regex(filepath, content, ".py")
            return

        self.file_symbols[filepath] = symbols
        self.imports[filepath] = imports

        for sym in symbols:
            key = sym.name.lower()
            self.symbols.setdefault(key, []).append(sym)
            if sym.kind in (SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.METHOD):
                self.definitions.setdefault(key, []).append(sym)
            if sym.kind == SymbolKind.CLASS:
                self.classes.setdefault(key, []).append(sym)

        for ref in refs:
            rkey = str(ref["symbol"]).lower()
            self.references.setdefault(rkey, []).append(ref)

    def _index_regex(self, filepath: str, content: str, ext: str):
        symbols = []
        imports = []
        lines = content.splitlines()

        patterns = [
            (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z0-9_$]+)(?:\s+extends\s+([A-Za-z0-9_$.]+))?", re.M), SymbolKind.CLASS),
            (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(([^)]*)\)", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*function\s*\(([^)]*)\)", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*class\s+([A-Za-z0-9_]+)(?:\(([^)]*)\))?:", re.M), SymbolKind.CLASS),
            (re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z0-9_]+)\s*\(", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z0-9_]+)", re.M), SymbolKind.STRUCT),
            (re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)\s*\(", re.M), SymbolKind.FUNCTION),
            (re.compile(r"^\s*type\s+([A-Za-z0-9_]+)\s+struct", re.M), SymbolKind.STRUCT),
            (re.compile(r"^\s*type\s+([A-Za-z0-9_]+)\s+interface", re.M), SymbolKind.INTERFACE),
        ]

        import_patterns = [
            re.compile(r"^\s*import\s+(?:\{([^}]+)\}|\*\s+as\s+([A-Za-z0-9_$]+)|([A-Za-z0-9_$]+))\s+from\s+['\"]([^'\"]+)['\"]", re.M),
            re.compile(r"^\s*(?:const|let|var)\s+(?:\{([^}]+)\}|([A-Za-z0-9_$]+))\s*=\s*require\(['\"]([^'\"]+)['\"]\)", re.M),
            re.compile(r"^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)|import\s+([A-Za-z0-9_., ]+))", re.M),
        ]

        for line_no, line in enumerate(lines, 1):
            for pat, kind in patterns:
                m = pat.match(line)
                if m:
                    name = m.group(1)
                    sig = line.strip().rstrip("{: ")
                    sym = SymbolItem(name=name, kind=kind, filepath=filepath, line=line_no, signature=sig)
                    symbols.append(sym)
                    break
            for ipat in import_patterns:
                im = ipat.match(line)
                if im:
                    imports.append({"line": line_no, "text": line.strip()})
                    break

        self.file_symbols[filepath] = symbols
        self.imports[filepath] = imports
        for sym in symbols:
            key = sym.name.lower()
            self.symbols.setdefault(key, []).append(sym)
            if sym.kind in (SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.STRUCT, SymbolKind.INTERFACE):
                self.definitions.setdefault(key, []).append(sym)
            if sym.kind == SymbolKind.CLASS:
                self.classes.setdefault(key, []).append(sym)

    def invalidate_file(self, filepath: str):
        """Invalidates only the modified file and updates its indexed symbols."""
        norm_path = os.path.normpath(filepath)
        self.index_file(norm_path)

    def find_symbol(self, query: str, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_indexed()
        query_low = query.strip().lower()
        results = []
        for key, sym_list in self.symbols.items():
            if query_low in key or query_low == key:
                for sym in sym_list:
                    if kind and sym.kind.lower() != kind.lower():
                        continue
                    results.append(sym.to_dict())
        return sorted(results, key=lambda x: (0 if x["name"].lower() == query_low else 1, x["filepath"], x["line"]))[:50]

    def find_definition(self, symbol: str, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_indexed()
        sym_low = symbol.strip().lower()
        matches = self.definitions.get(sym_low, [])
        if not matches:
            return self.find_symbol(symbol)
        results = [m.to_dict() for m in matches]
        if filepath:
            norm_f = os.path.normpath(filepath)
            results.sort(key=lambda x: 0 if os.path.normpath(x["filepath"]) == norm_f else 1)
        return results[:30]

    def find_references(self, symbol: str) -> List[Dict[str, Any]]:
        self.ensure_indexed()
        sym_low = symbol.strip().lower()
        refs = self.references.get(sym_low, [])
        if refs:
            return refs[:50]
        return _python_search_code_structured(symbol, str(self.root))[:50]

    def find_related_files(self, filepath: str) -> List[Dict[str, str]]:
        self.ensure_indexed()
        norm_target = os.path.normpath(filepath)
        target_name = Path(norm_target).stem.lower()
        target_dir = os.path.dirname(norm_target)
        related = {}

        target_imports = self.imports.get(norm_target, [])
        for imp in target_imports:
            mod = imp.get("module") or imp.get("text", "")
            for f in self.file_symbols:
                if f != norm_target and (Path(f).stem.lower() in mod.lower() or mod.lower() in f.lower()):
                    related[f] = f"Imported by {norm_target}"

        for f, imp_list in self.imports.items():
            if f == norm_target:
                continue
            for imp in imp_list:
                mod = imp.get("module") or imp.get("text", "")
                if target_name in mod.lower() or norm_target.replace("\\", "/").lower() in mod.lower():
                    related[f] = f"Imports {norm_target}"

        for f in self.file_symbols:
            if f != norm_target and os.path.dirname(f) == target_dir:
                related.setdefault(f, "Same directory")

        target_syms = {s.name.lower() for s in self.file_symbols.get(norm_target, [])}
        for f, syms in self.file_symbols.items():
            if f != norm_target:
                for s in syms:
                    for b in s.bases:
                        if b.lower() in target_syms:
                            related[f] = f"Inherits {b} from {norm_target}"

        return [{"filepath": k, "reason": v} for k, v in list(related.items())[:25]]


INDEX = CodebaseIndex()


# ---------------------------------------------------------------------------
# Project scan / cache
# ---------------------------------------------------------------------------
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
        "GHOST.md": (root_path / "GHOST.md").exists(),
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
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(data):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
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


def invalidate_cache(root="."):
    cache = _load_cache()
    key = str(Path(root).resolve())
    if key in cache:
        del cache[key]
        _save_cache(cache)


# ---------------------------------------------------------------------------
# Git helpers & Shadow Checkpoints (Tier 2)
# ---------------------------------------------------------------------------
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
    """Create a non-destructive git snapshot of tracked + untracked state before a turn."""
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
    """Restore the most recent git shadow checkpoint."""
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
        INDEX.build_index()
        invalidate_cache(".")
        return True, f"Restored workspace to checkpoint ({entry.get('label', 'turn')}) [{ref[:8]}]."
    except Exception as e:
        return False, f"Failed to restore git checkpoint: {e}"


# ---------------------------------------------------------------------------
# Undo / history
# ---------------------------------------------------------------------------
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
    """Revert the last `n` recorded edits, or fall back to shadow git checkpoint."""
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
            INDEX.invalidate_file(entry["filepath"])
        except Exception as e:
            _save_history(stack)
            return reverted, f"Undo failed on {entry['filepath']}: {e}"
    _save_history(stack)
    invalidate_cache(".")
    return reverted, None


def history_list_recent(limit: int = 10):
    stack = _load_history()
    return list(reversed(stack[-limit:]))


def history_clear():
    _save_history([])


def history_count():
    return len(_load_history())


# ---------------------------------------------------------------------------
# Live Todo / Checklist Tool (Tier 2)
# ---------------------------------------------------------------------------
class TodoManager:
    def __init__(self):
        self.items: List[Dict[str, str]] = []

    def update(self, todos: List[Dict[str, Any]]) -> str:
        cleaned = []
        for idx, item in enumerate(todos, 1):
            if not isinstance(item, dict):
                continue
            tid = str(item.get("id", idx))
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            if status not in ("pending", "in_progress", "completed"):
                status = "pending"
            if content:
                cleaned.append({"id": tid, "content": content, "status": status})
        self.items = cleaned
        self.render()
        return self.format_text()

    def format_text(self) -> str:
        if not self.items:
            return "Todo list is empty."
        lines = ["Current Todo Checklist:"]
        for item in self.items:
            sym = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}.get(item["status"], "[ ]")
            lines.append(f"  {sym} #{item['id']}: {item['content']} ({item['status']})")
        return "\n".join(lines)

    def render(self):
        if not self.items:
            return
        lines = []
        for item in self.items:
            st = item["status"]
            if st == "completed":
                lines.append(f"[bold green]✔[/bold green] [strike dim]{item['content']}[/strike dim]")
            elif st == "in_progress":
                lines.append(f"[bold cyan]▶[/bold cyan] [bold white]{item['content']}[/bold white]")
            else:
                lines.append(f"[dim]○ {item['content']}[/dim]")
        console.print(Panel("\n".join(lines), title="[bold cyan]Live Checklist[/bold cyan]", border_style="grey50"))


TODO_STATE = TodoManager()


def todo_write(todos: List[Dict[str, Any]]) -> str:
    """Create or update the live checklist for the current task."""
    return TODO_STATE.update(todos)


def todo_read() -> str:
    """Read the current live checklist."""
    return TODO_STATE.format_text()


# ---------------------------------------------------------------------------
# Structured Planning & Execution System
# ---------------------------------------------------------------------------
READ_ONLY_TOOLS = {
    "read_file",
    "list_dir",
    "search_code",
    "find_symbol",
    "find_definition",
    "find_references",
    "find_related_files",
    "git_status",
    "git_diff",
    "git_log",
    "scan_project",
    "todo_read",
    "bg_status",
}

PLANNING_SYSTEM_SUFFIX = """

PLANNING MODE is active. You may only use read-only inspection tools:
read_file, list_dir, search_code, find_symbol, find_definition, find_references, find_related_files, git_status, git_diff, git_log, scan_project.
Do NOT call write_file, edit_file, run_command, git_commit, or git_push right now.

Investigate the codebase as needed, then respond with a structured execution plan enclosed in a ```json ``` block matching:
{
  "objective": "Concise summary of what will be achieved",
  "affected_files": ["path/to/file1.py", "path/to/file2.py"],
  "risks": "Potential edge cases or risks",
  "tasks": [
    {
      "id": 1,
      "description": "Concise action description",
      "affected_files": ["path/to/file1.py"],
      "dependencies": [],
      "expected_changes": "What will be created or modified",
      "verification_methods": ["syntax_check", "run_tests"],
      "risks": "Risks for this step"
    }
  ]
}

End your response once the JSON plan is formulated. Do not perform the work yet.
"""


class TaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class TaskItem:
    def __init__(
        self,
        task_id: Any,
        description: str,
        affected_files: Optional[List[str]] = None,
        dependencies: Optional[List[Any]] = None,
        expected_changes: str = "",
        verification_methods: Optional[List[str]] = None,
        risks: str = "",
    ):
        self.id = task_id
        self.description = description
        self.affected_files = affected_files or []
        self.dependencies = dependencies or []
        self.expected_changes = expected_changes
        self.verification_methods = verification_methods or ["syntax_check"]
        self.risks = risks
        self.status = TaskStatus.PENDING
        self.result = ""
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "affected_files": self.affected_files,
            "dependencies": self.dependencies,
            "expected_changes": self.expected_changes,
            "verification_methods": self.verification_methods,
            "risks": self.risks,
            "status": self.status,
            "attempts": self.attempts,
        }


class StructuredPlan:
    def __init__(
        self,
        objective: str = "",
        affected_files: Optional[List[str]] = None,
        tasks: Optional[List[TaskItem]] = None,
        risks: str = "",
    ):
        self.objective = objective
        self.affected_files = affected_files or []
        self.tasks = tasks or []
        self.risks = risks
        self.raw_text = ""
        self.critique = ""

    def get_task(self, task_id: Any) -> Optional[TaskItem]:
        for t in self.tasks:
            if str(t.id) == str(task_id):
                return t
        return None

    def is_complete(self) -> bool:
        return len(self.tasks) > 0 and all(t.status == TaskStatus.COMPLETED for t in self.tasks)


class PlanState:
    def __init__(self):
        self.active = False
        self.task_description = None
        self.structured_plan: Optional[StructuredPlan] = None

    def start(self, task: str):
        self.active = True
        self.task_description = task
        self.structured_plan = None

    def finish_drafting(self, plan: StructuredPlan):
        self.active = False
        self.structured_plan = plan

    def clear(self):
        self.active = False
        self.task_description = None
        self.structured_plan = None

    def has_plan(self) -> bool:
        return self.structured_plan is not None and len(self.structured_plan.tasks) > 0


PLAN_STATE = PlanState()


def parse_plan_response(text: str, default_task: str = "") -> StructuredPlan:
    """Parses model response into a StructuredPlan, supporting JSON blocks and structured text."""
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    data = None
    if json_match:
        try:
            data = json.loads(json_match.group(1))
        except Exception:
            data = None

    if not data:
        raw_match = re.search(r"(\{[\s\S]*\"tasks\"[\s\S]*\})", text)
        if raw_match:
            try:
                data = json.loads(raw_match.group(1))
            except Exception:
                data = None

    if isinstance(data, dict) and "tasks" in data:
        objective = str(data.get("objective", default_task or "Execution plan"))
        affected_files = list(data.get("affected_files", []))
        risks = str(data.get("risks", ""))
        tasks = []
        for idx, t in enumerate(data.get("tasks", []), 1):
            if isinstance(t, dict):
                tid = t.get("id", idx)
                desc = t.get("description", f"Step {idx}")
                afiles = t.get("affected_files", [])
                deps = t.get("dependencies", [])
                changes = t.get("expected_changes", "")
                verif = t.get("verification_methods", ["syntax_check"])
                risk = t.get("risks", "")
                tasks.append(
                    TaskItem(
                        task_id=tid,
                        description=desc,
                        affected_files=afiles,
                        dependencies=deps,
                        expected_changes=changes,
                        verification_methods=verif,
                        risks=risk,
                    )
                )
            elif isinstance(t, str):
                tasks.append(TaskItem(task_id=idx, description=t))

        plan = StructuredPlan(objective=objective, affected_files=affected_files, tasks=tasks, risks=risks)
        plan.raw_text = text
        return plan

    tasks = []
    objective = default_task or "Execution plan"
    affected_files = []
    risks = ""

    lines = text.splitlines()
    step_pattern = re.compile(r"^\s*(?:###?\s*)?(?:Step\s*)?(\d+)[\.\:\)]\s*(.+)", re.IGNORECASE)
    file_pattern = re.compile(r"[`'\"]([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)[`'\"]")

    for line in lines:
        line_clean = line.strip()
        if line_clean.lower().startswith(("objective:", "**objective:**", "# objective")):
            objective = re.sub(r"^(?:#+\s*)?(?:\*\*)?objective:(?:\*\*)?\s*", "", line_clean, flags=re.IGNORECASE).strip()
            continue
        if line_clean.lower().startswith(("risks:", "**risks:**", "# risks")):
            risks = re.sub(r"^(?:#+\s*)?(?:\*\*)?risks:(?:\*\*)?\s*", "", line_clean, flags=re.IGNORECASE).strip()
            continue
        m = step_pattern.match(line_clean)
        if m:
            tid = int(m.group(1))
            desc = m.group(2).strip()
            found_files = file_pattern.findall(desc)
            affected_files.extend(found_files)
            tasks.append(
                TaskItem(
                    task_id=tid,
                    description=desc,
                    affected_files=found_files,
                    dependencies=[tid - 1] if tid > 1 else [],
                    expected_changes="",
                    verification_methods=["syntax_check"],
                    risks="",
                )
            )

    if not tasks:
        tasks.append(
            TaskItem(
                task_id=1,
                description=default_task or "Perform planned updates",
                affected_files=file_pattern.findall(text),
                dependencies=[],
                expected_changes="",
                verification_methods=["syntax_check"],
                risks="",
            )
        )

    plan = StructuredPlan(
        objective=objective,
        affected_files=sorted(list(set(affected_files))),
        tasks=tasks,
        risks=risks,
    )
    plan.raw_text = text
    return plan


def render_structured_plan(plan: StructuredPlan):
    """Renders structured plan in terminal."""
    body_lines = [
        f"[bold white]Objective:[/bold white] {plan.objective}",
    ]
    if plan.affected_files:
        body_lines.append(f"[bold white]Affected Files:[/bold white] {', '.join(plan.affected_files)}")
    if plan.risks:
        body_lines.append(f"[bold yellow]Risks:[/bold yellow] {plan.risks}")
    if plan.critique:
        body_lines.append(f"[bold cyan]Critique Notes:[/bold cyan] {plan.critique}")

    console.print(Panel("\n".join(body_lines), title="Structured Plan", border_style="cyan"))

    table = Table(title="Execution Tasks", border_style="grey50", header_style="bold cyan")
    table.add_column("#", style="bold white", width=4)
    table.add_column("Task Description", style="white")
    table.add_column("Dependencies", style="dim", width=14)
    table.add_column("Verification", style="dim", width=16)
    table.add_column("Status", width=12)

    status_styles = {
        TaskStatus.PENDING: "[dim]PENDING[/dim]",
        TaskStatus.RUNNING: "[bold cyan]RUNNING[/bold cyan]",
        TaskStatus.COMPLETED: "[bold green]COMPLETED[/bold green]",
        TaskStatus.FAILED: "[bold red]FAILED[/bold red]",
        TaskStatus.BLOCKED: "[yellow]BLOCKED[/yellow]",
        TaskStatus.SKIPPED: "[dim]SKIPPED[/dim]",
    }

    for t in plan.tasks:
        deps_str = ", ".join(str(d) for d in t.dependencies) if t.dependencies else "None"
        verif_str = ", ".join(t.verification_methods) if t.verification_methods else "syntax_check"
        status_str = status_styles.get(t.status, t.status)
        table.add_row(str(t.id), t.description, deps_str, verif_str, status_str)

    console.print(table)
    console.print("[dim]Type /execute to carry out this plan step-by-step, or /cancelplan to discard.[/dim]")


def render_plan_summary(plan: StructuredPlan):
    table = Table(title=f"Plan Summary: {plan.objective}", border_style="grey50")
    table.add_column("#", width=4)
    table.add_column("Task", style="white")
    table.add_column("Status", width=12)
    table.add_column("Attempts", width=10)

    status_styles = {
        TaskStatus.PENDING: "[dim]PENDING[/dim]",
        TaskStatus.RUNNING: "[bold cyan]RUNNING[/bold cyan]",
        TaskStatus.COMPLETED: "[bold green]COMPLETED[/bold green]",
        TaskStatus.FAILED: "[bold red]FAILED[/bold red]",
        TaskStatus.BLOCKED: "[yellow]BLOCKED[/yellow]",
        TaskStatus.SKIPPED: "[dim]SKIPPED[/dim]",
    }
    for t in plan.tasks:
        table.add_row(str(t.id), t.description, status_styles.get(t.status, t.status), str(t.attempts or 1))
    console.print(table)


# ---------------------------------------------------------------------------
# Safety system, Sandbox & Session Allowlist (Tier 1 #10)
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
    r"rm\s+--recursive",
    r"rm\s+(?:-[a-zA-Z]+\s+)*-r\s+(?:-[a-zA-Z]+\s+)*-f",
    r"rm\s+(?:-[a-zA-Z]+\s+)*-f\s+(?:-[a-zA-Z]+\s+)*-r",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"mkfs(\.\w+)?\s",
    r"dd\s+.*of=/dev/",
    r">\s*/dev/sd[a-z]",
    r"chmod\s+-R\s+000",
    r"chmod\s+-R\s+777\s+/",
    r"git\s+push\s+.*--force",
    r"git\s+push\s+.*-f\b",
    r"git\s+branch\s+-D\b",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[a-zA-Z]*f",
    r"drop\s+(table|database)",
    r"\bshutdown\b",
    r"\breboot\b",
    r"format\s+[a-zA-Z]:",
    r"del\s+/[sf]\s+/[qs]",
    r"Remove-Item\s+.*-Recurse.*-Force",
    r"rd\s+/[sS]",
    r"find\s+.*-delete",
    r"find\s+.*-exec\s+rm",
    r"python[23]?\s+-c\s+.*(shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|os\.system|subprocess|pathlib\.Path.*unlink)",
    r"perl\s+-e\s+.*unlink",
    r"ruby\s+-e\s+.*FileUtils\.rm",
    r"node\s+-e\s+.*(rmSync|unlinkSync|rmdirSync)",
    r"curl\s+.*\|\s*(sudo\s+)?(sh|bash|zsh|python|perl|ruby)",
    r"wget\s+.*-O\s*-\s*\|",
    r"wget\s+.*\|\s*(sudo\s+)?(sh|bash|zsh|python|perl|ruby)",
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
    r"\bkill\b",
    r"\bpkill\b",
    r"docker\s+(rm|rmi|system\s+prune)",
    r">\s*[^&|]",
    r"sed\s+.*-i",
    r"tee\s+",
    r"python[23]?\s+-c\b",
]

# Session-level "always allow" prefixes for RISKY (not CRITICAL) commands
_SESSION_ALLOWED_PREFIXES: Set[str] = set()
_SESSION_APPROVED_EXTERNAL_DIRS: Set[str] = set()

STATIC_TOOL_RISK = {
    "read_file": RISK_LOW,
    "list_dir": RISK_LOW,
    "search_code": RISK_LOW,
    "find_symbol": RISK_LOW,
    "find_definition": RISK_LOW,
    "find_references": RISK_LOW,
    "find_related_files": RISK_LOW,
    "git_status": RISK_LOW,
    "git_diff": RISK_LOW,
    "git_log": RISK_LOW,
    "run_tests": RISK_LOW,
    "scan_project": RISK_LOW,
    "todo_write": RISK_LOW,
    "todo_read": RISK_LOW,
    "spawn_explorer": RISK_LOW,
    "bg_status": RISK_LOW,
    "bg_kill": RISK_RISKY,
    "write_file": RISK_RISKY,
    "edit_file": RISK_RISKY,
    "git_commit": RISK_CRITICAL,
    "git_push": RISK_CRITICAL,
}


def is_within_project_root(filepath: str) -> Tuple[bool, Path]:
    """Check if filepath resolves inside the current project root (CWD)."""
    root = Path.cwd().resolve()
    try:
        resolved = Path(filepath).resolve()
        resolved.relative_to(root)
        return True, resolved
    except ValueError:
        return False, Path(filepath).resolve()


def check_path_sandbox(filepath: str) -> bool:
    """Ensure writes outside the project root are explicitly approved by the user."""
    inside, resolved = is_within_project_root(filepath)
    if inside:
        return True
    parent_str = str(resolved.parent)
    if parent_str in _SESSION_APPROVED_EXTERNAL_DIRS:
        return True
    console.print(
        f"\n[bold red]⚠ Sandbox Warning:[/bold red] File [bold]{resolved}[/bold] is outside project root ([dim]{Path.cwd().resolve()}[/dim])."
    )
    try:
        ans = Prompt.ask(
            "Allow writing outside project root? ([y]es once / [a]lways allow this directory / [n]o)",
            choices=["y", "a", "n"],
            default="n",
        ).strip().lower()
        if ans == "a":
            _SESSION_APPROVED_EXTERNAL_DIRS.add(parent_str)
            return True
        return ans == "y"
    except KeyboardInterrupt:
        return False


def _extract_command_prefix(command: str) -> str:
    tokens = command.strip().split()
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[0] in ("git", "npm", "pip", "pytest", "python", "python3", "cargo", "go", "docker"):
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


def is_command_allowed_by_session(command: str) -> bool:
    cmd_strip = command.strip()
    for prefix in _SESSION_ALLOWED_PREFIXES:
        if cmd_strip == prefix or cmd_strip.startswith(prefix + " "):
            return True
    return False


def classify_command(command: str) -> str:
    """Classify a raw shell command string into low / risky / critical."""
    for pat in CRITICAL_COMMAND_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return RISK_CRITICAL
    for pat in RISKY_COMMAND_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            if is_command_allowed_by_session(command):
                return RISK_LOW
            return RISK_RISKY
    return RISK_LOW


def assess_tool_call(fn_name: str, args: dict) -> tuple:
    """Returns (risk, short_description) for a tool call."""
    if fn_name == "run_command":
        cmd = str(args.get("command", ""))
        return classify_command(cmd), f"run_command: {cmd}"
    if fn_name == "write_file":
        fp = str(args.get("filepath", "?"))
        inside, _ = is_within_project_root(fp)
        risk = RISK_RISKY if inside else RISK_CRITICAL
        return risk, f"write_file → {fp}"
    if fn_name == "edit_file":
        fp = str(args.get("filepath", "?"))
        inside, _ = is_within_project_root(fp)
        risk = RISK_RISKY if inside else RISK_CRITICAL
        return risk, f"edit_file → {fp}"
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


def _ask(risk: str, command_for_prefix: Optional[str] = None) -> bool:
    try:
        if risk == RISK_CRITICAL:
            ans = Prompt.ask("[bold red]Type 'yes' to proceed, anything else cancels[/bold red]", default="no")
            return ans.strip().lower() == "yes"
        if command_for_prefix:
            prefix = _extract_command_prefix(command_for_prefix)
            ans = Prompt.ask(
                f"Proceed? [y/N/a (always allow '{prefix}')]",
                default="n",
            ).strip().lower()
            if ans == "a" and prefix:
                _SESSION_ALLOWED_PREFIXES.add(prefix)
                console.print(f"[dim green]✔ Added '{prefix}' to session allowlist.[/dim green]")
                return True
            return ans in ("y", "yes")
        ans = Prompt.ask("Proceed? [y/N]", default="n")
        return ans.strip().lower() in ("y", "yes")
    except KeyboardInterrupt:
        console.print("\n[yellow]Action cancelled by user.[/yellow]")
        return False


def prompt_confirm(description: str, risk: str, command_for_prefix: Optional[str] = None) -> bool:
    """Ask the user to approve an action. Returns True if approved."""
    console.print(f"\n[bold]⚠ confirmation required[/bold] ({_risk_tag(risk)}) — {description}")
    return _ask(risk, command_for_prefix=command_for_prefix)


def confirm_with_preview(description: str, risk: str, show_preview) -> bool:
    """Prints preview first and then invokes confirmation prompt."""
    console.print(f"\n[bold]⚠ confirmation required[/bold] ({_risk_tag(risk)}) — {description}")
    if show_preview is not None:
        show_preview()
    return _ask(risk)


# ---------------------------------------------------------------------------
# Background Process Manager & Live Streaming `run_command` (Tier 1 #9)
# ---------------------------------------------------------------------------
RIPGREP_AVAILABLE = shutil.which("rg") is not None
MAX_READ_BYTES = 80_000   # ~20k tokens
_TOOL_OUT_CHARS = 12_000  # per-tool output cap


def truncate_output(text: str, max_chars: int = _TOOL_OUT_CHARS) -> str:
    """Keep head and tail with a truncation marker when output exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    removed_lines = text[half:-half].count("\n")
    removed_chars = len(text) - max_chars
    marker = f"\n... [truncated {removed_lines} lines / {removed_chars:,} chars] ...\n"
    return head + marker + tail


class BackgroundProcessManager:
    """Manages long-running background processes (e.g., dev servers, watchers)."""

    def __init__(self):
        self.procs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._counter = 0

    def start(self, command: str) -> str:
        with self.lock:
            self._counter += 1
            job_id = f"bg_{self._counter}"

        buf: List[str] = []
        buf_lock = threading.Lock()

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        def _reader():
            try:
                for line in iter(proc.stdout.readline, ""):
                    with buf_lock:
                        buf.append(line)
                        if len(buf) > 2000:
                            del buf[:500]
            except Exception:
                pass

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        with self.lock:
            self.procs[job_id] = {
                "id": job_id,
                "command": command,
                "proc": proc,
                "buffer": buf,
                "lock": buf_lock,
                "started_at": time.time(),
            }

        time.sleep(0.4)
        rc = proc.poll()
        with buf_lock:
            initial_logs = "".join(buf[-30:]).strip()
        status_str = "running" if rc is None else f"exited ({rc})"
        UI.success(f"Background process {job_id} ({status_str}) — PID {proc.pid}")
        return (
            f"Started background process '{job_id}' (PID {proc.pid}, status: {status_str}).\n"
            f"Initial output:\n{initial_logs or '(no output yet)'}"
        )

    def status(self, job_id: Optional[str] = None, tail_lines: int = 50) -> str:
        with self.lock:
            if not self.procs:
                return "No background processes."
            if job_id:
                info = self.procs.get(job_id)
                if not info:
                    return f"Background job '{job_id}' not found. Active IDs: {', '.join(self.procs.keys())}"
                targets = [info]
            else:
                targets = list(self.procs.values())

        out_blocks = []
        for info in targets:
            proc = info["proc"]
            rc = proc.poll()
            state = "running" if rc is None else f"exited ({rc})"
            with info["lock"]:
                logs = "".join(info["buffer"][-max(1, int(tail_lines)):]).strip()
            out_blocks.append(
                f"[{info['id']}] PID {proc.pid} | {state} | cmd: {info['command']}\n"
                f"--- last {tail_lines} lines ---\n{logs or '(empty)'}"
            )
        return truncate_output("\n\n".join(out_blocks))

    def kill(self, job_id: str) -> str:
        with self.lock:
            info = self.procs.get(job_id)
        if not info:
            return f"Background job '{job_id}' not found."
        proc = info["proc"]
        if proc.poll() is not None:
            return f"Job '{job_id}' already exited with code {proc.returncode}."
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            UI.success(f"Terminated background job {job_id} (PID {proc.pid})")
            return f"Terminated background job '{job_id}' (PID {proc.pid})."
        except Exception as e:
            return f"Error killing job '{job_id}': {e}"


BG_MANAGER = BackgroundProcessManager()


def bg_status(job_id: Optional[str] = None, tail_lines: int = 50) -> str:
    """Check status and recent logs of background processes."""
    return BG_MANAGER.status(job_id=job_id, tail_lines=tail_lines)


def bg_kill(job_id: str) -> str:
    """Terminate a background process by its job_id."""
    return BG_MANAGER.kill(job_id)


def run_command(command: str, timeout: int = 60, background: bool = False) -> str:
    """Execute a shell command with live output streaming, stderr retention, and optional background mode."""
    if background:
        UI.status(f"Starting background process: {command}")
        return BG_MANAGER.start(command)

    timeout = max(1, min(int(timeout), 600))
    UI.status(f"Running: {command}")
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def _read_stream(stream, collector, is_err=False):
            try:
                for line in iter(stream.readline, ""):
                    collector.append(line)
                    if UI.verbose:
                        prefix = "[dim red]│ [/dim red]" if is_err else "[dim]│ [/dim]"
                        console.print(prefix + line.rstrip("\r\n"))
            except Exception:
                pass

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, False), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, True), daemon=True)
        t_out.start()
        t_err.start()

        start_time = time.time()
        while proc.poll() is None:
            if time.time() - start_time > timeout:
                proc.kill()
                UI.error(f"Command timed out after {timeout}s.")
                partial_out = "".join(stdout_lines).strip()
                partial_err = "".join(stderr_lines).strip()
                combined = "\n".join(filter(None, [partial_out, f"[stderr]\n{partial_err}" if partial_err else ""]))
                return truncate_output(f"Execution error: command timed out after {timeout}s.\n{combined}".strip())
            time.sleep(0.05)

        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

        out_str = "".join(stdout_lines).strip()
        err_str = "".join(stderr_lines).strip()

        parts = []
        if out_str:
            parts.append(out_str)
        if err_str:
            label = "stderr" if proc.returncode == 0 else "stderr (error)"
            parts.append(f"[{label}]\n{err_str}")
        output = "\n".join(parts).strip()

        if proc.returncode == 0:
            add_xp(15, "clean execution")
            UI.success("Command completed")
        else:
            add_xp(5, "hit a bug")
            UI.warn(f"Command exited with code {proc.returncode}")

        if not output:
            return "Command executed successfully." if proc.returncode == 0 else f"Command failed with exit code {proc.returncode}"
        result = output if proc.returncode == 0 else f"Exit code {proc.returncode}:\n{output}"
        return truncate_output(result)

    except KeyboardInterrupt:
        try:
            proc.kill()
        except Exception:
            pass
        UI.warn("Command interrupted by user.")
        return "Execution error: command interrupted by user (SIGINT)."
    except Exception as e:
        UI.error(f"Command execution error: {e}")
        return f"Execution error: {str(e)}"


def read_file(filepath: str, offset: int = 1, limit: int = 2000) -> str:
    """Read a file, optionally by line range. Returns numbered lines."""
    try:
        UI.status(f"Reading {filepath}...")
        path = Path(filepath)
        if not path.exists():
            UI.error(f"File '{filepath}' not found.")
            return f"File read error: '{filepath}' not found."

        raw_bytes = path.read_bytes()
        if len(raw_bytes) > MAX_READ_BYTES:
            UI.warn(f"{filepath} is {len(raw_bytes):,} bytes (large file, line-based read applied)")

        with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
            all_lines = f.read().splitlines()

        total_lines = len(all_lines)
        offset = max(1, int(offset))
        limit = max(1, int(limit))
        end = min(offset + limit - 1, total_lines)
        selected = all_lines[offset - 1:end]

        width = len(str(total_lines))
        numbered = "\n".join(f"{str(offset + i).rjust(width)}\t{line}" for i, line in enumerate(selected))

        hint = ""
        if end < total_lines:
            hint = f"\n[file has {total_lines} lines; showing {offset}–{end} ({total_lines - end} lines remaining). Use offset={end + 1} to read more]"
        elif offset > 1:
            hint = f"\n[file has {total_lines} lines; showing {offset}–{end}]"

        add_xp(5, "scanned file")
        UI.success(f"Read {filepath} (lines {offset}–{end} of {total_lines})")

        if UI.verbose:
            try:
                lexer = Syntax.guess_lexer(filepath, code=numbered)
                console.print(Syntax(numbered, lexer, theme="monokai", line_numbers=False, word_wrap=True))
            except Exception:
                console.print(numbered[:3000])

        return truncate_output(numbered + hint)
    except UnicodeDecodeError:
        UI.error(f"'{filepath}' does not look like a text file.")
        return f"File read error: '{filepath}' does not look like a text file."
    except Exception as e:
        UI.error(f"File read error: {e}")
        return f"File read error: {str(e)}"


def list_dir(path: str = ".") -> str:
    try:
        UI.status(f"Listing {path}...")
        entries = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
            depth = root.replace(path, "").count(os.sep)
            if depth > 2:
                continue
            for f in files:
                entries.append(os.path.relpath(os.path.join(root, f), path))
        entries_sorted = sorted(entries)[:200]
        listing = "\n".join(entries_sorted)
        add_xp(3, "surveyed the terrain")
        UI.success(f"Listed {len(entries_sorted)} items in {path}")
        if UI.verbose and listing:
            UI.detail(listing)
        return truncate_output(listing or "(empty)")
    except Exception as e:
        UI.error(f"List error: {e}")
        return f"List error: {str(e)}"


def search_code(pattern: str, path: str = ".") -> str:
    try:
        UI.status(f"Searching code for '{pattern}'...")
        if RIPGREP_AVAILABLE:
            cmd = ["rg", "-n", "--no-heading", "--color=never", "-g", "!.git", "-g", "!__pycache__", "-g", "!node_modules", pattern, path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = res.stdout.strip()
        else:
            out = _python_search_code(pattern, path)

        add_xp(5, "swept the codebase")
        match_count = len(out.splitlines()) if out and out != "No matches." else 0
        UI.success(f"Found {match_count} match(es) for '{pattern}'")
        if UI.verbose and out:
            UI.detail(out[:4000])
        return truncate_output(out if out else "No matches.")
    except subprocess.TimeoutExpired:
        UI.error("Search timed out.")
        return "Search error: search timed out."
    except KeyboardInterrupt:
        UI.warn("Search cancelled by user.")
        return "Search error: search cancelled by user."
    except Exception as e:
        UI.error(f"Search error: {e}")
        return f"Search error: {str(e)}"


def _find_symbol_tool(query: str, kind: Optional[str] = None) -> str:
    UI.status(f"Searching symbol '{query}'...")
    results = INDEX.find_symbol(query, kind)
    if not results:
        UI.status(f"No symbols found matching '{query}'.")
        return f"No symbols found matching '{query}'."
    UI.success(f"Found {len(results)} symbol(s) matching '{query}'")
    lines = [f"{r['kind']} {r['name']} — {r['filepath']}:{r['line']} ({r.get('signature', '')})" for r in results]
    if UI.verbose:
        UI.detail("\n".join(lines))
    return truncate_output("\n".join(lines))


def _find_definition_tool(symbol: str, filepath: Optional[str] = None) -> str:
    UI.status(f"Finding definition of '{symbol}'...")
    results = INDEX.find_definition(symbol, filepath)
    if not results:
        UI.status(f"No definition found for '{symbol}'.")
        return f"No definition found for '{symbol}'."
    UI.success(f"Located definition for '{symbol}'")
    lines = [
        f"Definition of {r['name']} ({r.get('kind', 'symbol')}): {r['filepath']}:{r['line']}\n  {r.get('signature', '')}"
        for r in results
    ]
    if UI.verbose:
        UI.detail("\n".join(lines))
    return truncate_output("\n".join(lines))


def _find_references_tool(symbol: str) -> str:
    UI.status(f"Finding references to '{symbol}'...")
    results = INDEX.find_references(symbol)
    if not results:
        UI.status(f"No references found for '{symbol}'.")
        return f"No references found for '{symbol}'."
    UI.success(f"Found {len(results)} reference(s) to '{symbol}'")
    lines = [f"{r.get('filepath', '?')}:{r.get('line', '?')} — {r.get('context', symbol)}" for r in results]
    if UI.verbose:
        UI.detail("\n".join(lines))
    return truncate_output("\n".join(lines))


def _find_related_files_tool(filepath: str) -> str:
    UI.status(f"Finding related files for '{filepath}'...")
    results = INDEX.find_related_files(filepath)
    if not results:
        UI.status(f"No related files found for '{filepath}'.")
        return f"No related files found for '{filepath}'."
    UI.success(f"Found {len(results)} related file(s)")
    lines = [f"{r['filepath']} — {r['reason']}" for r in results]
    if UI.verbose:
        UI.detail("\n".join(lines))
    return truncate_output("\n".join(lines))


def _print_diff(diff_text: str):
    if not diff_text:
        console.print("[dim](no textual difference)[/dim]")
        return
    console.print(Syntax(diff_text, "diff", theme="monokai", word_wrap=True))


def write_file(filepath: str, content: str, confirm=None) -> str:
    try:
        if not check_path_sandbox(filepath):
            UI.warn(f"write_file blocked by project root sandbox: {filepath}")
            return f"Error: write_file to '{filepath}' outside project root was denied by user."

        UI.status(f"Writing {filepath}...")
        previous = ""
        existed = os.path.exists(filepath)
        orig_ending = "\n"
        if existed:
            try:
                raw = Path(filepath).read_bytes()
                orig_ending = _detect_line_ending(raw)
                with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
                    previous = f.read()
            except Exception:
                previous = ""

        diff_text = unified_diff_text(previous, content, filepath)
        if confirm is not None and not confirm(diff_text):
            UI.warn(f"write_file cancelled by user for {filepath}.")
            return "write_file cancelled by user."

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        write_content = content
        if existed and orig_ending == "\r\n":
            write_content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(write_content)

        history_push("write_file", filepath, previous if existed else "")
        INDEX.invalidate_file(filepath)
        invalidate_cache(".")

        syntax_ok, syntax_err = validate_syntax(filepath, content)

        if UI.verbose:
            _print_diff(diff_text)

        action = "Created" if not existed else "Overwrote"
        UI.success(f"{action} {filepath} ({len(content)} bytes)")
        add_xp(20, "forged a new file" if not existed else "overwrote a file")

        result_msg = f"{action} {filepath} ({len(content)} bytes)."
        if not syntax_ok:
            UI.warn(f"Syntax warning in {filepath}: {syntax_err}")
            result_msg += f"\nWarning: Syntax validation failed:\n{syntax_err}"
        return result_msg
    except Exception as e:
        UI.error(f"write_file failed on {filepath}: {e}")
        return str(e)


def _detect_line_ending(raw: bytes) -> str:
    """Return '\r\n' if CRLF dominates, else '\n'."""
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_ws(text: str) -> str:
    """Strip trailing whitespace per line and normalize CRLF→LF for fuzzy matching."""
    return "\n".join(line.rstrip().expandtabs(4) for line in text.replace("\r\n", "\n").splitlines())


def _find_closest_region(content: str, old_text: str, n: int = 5) -> str:
    """Return the n closest matching lines from content for the model to use as a reference."""
    content_lines = content.splitlines()
    old_lines = old_text.splitlines()
    if not content_lines:
        return "File is empty."
    best_ratio = 0.0
    best_start = 0
    window = max(1, min(len(old_lines), len(content_lines)))
    for i in range(max(1, len(content_lines) - window + 1)):
        block = "\n".join(content_lines[i:i + window])
        ratio = difflib.SequenceMatcher(None, old_text, block).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
    snippet_lines = content_lines[max(0, best_start - 2):best_start + window + 2]
    start_num = max(1, best_start - 1)
    numbered = "\n".join(f"{start_num + i}\t{l}" for i, l in enumerate(snippet_lines))
    return f"Closest region (lines {start_num}–{start_num + len(snippet_lines) - 1}, similarity {best_ratio:.0%}):\n{numbered}"


def _apply_single_edit(content: str, old_text: str, new_text: str, replace_all: bool = False) -> Tuple[str, str]:
    """
    Apply one old→new replacement. Returns (new_content, error_message).
    error_message is empty on success.
    """
    if not old_text:
        return content, "old_text cannot be empty."

    # Normalize CRLF inside old_text/new_text to match content's internal representation
    work_content = content.replace("\r\n", "\n")
    work_old = old_text.replace("\r\n", "\n")
    work_new = new_text.replace("\r\n", "\n")

    count = work_content.count(work_old)

    if count == 1 or (count > 1 and replace_all):
        if replace_all:
            return work_content.replace(work_old, work_new), ""
        return work_content.replace(work_old, work_new, 1), ""

    if count > 1:
        return content, (
            f"old_text matched {count} locations. Add more surrounding context lines to make it unique, "
            f"or pass replace_all=true to replace all occurrences."
        )

    # Line-by-line whitespace-normalized match
    content_lines = work_content.splitlines()
    old_lines = work_old.splitlines()
    norm_content_lines = [l.rstrip().expandtabs(4) for l in content_lines]
    norm_old_lines = [l.rstrip().expandtabs(4) for l in old_lines]

    matches = []
    w = len(norm_old_lines)
    if w > 0 and w <= len(norm_content_lines):
        for i in range(len(norm_content_lines) - w + 1):
            if norm_content_lines[i:i + w] == norm_old_lines:
                matches.append(i)

    if len(matches) == 1 or (len(matches) > 1 and replace_all):
        result_lines = list(content_lines)
        new_lines = work_new.splitlines()
        for idx in reversed(matches):
            result_lines[idx:idx + w] = new_lines
        rebuilt = "\n".join(result_lines)
        if work_content.endswith("\n"):
            rebuilt += "\n"
        return rebuilt, ""

    if len(matches) > 1:
        return content, (
            f"Whitespace-normalized old_text matched {len(matches)} locations. "
            f"Include more context lines or set replace_all=true."
        )

    closest = _find_closest_region(work_content, work_old)
    return content, f"old_text not found (exact or whitespace-normalized). {closest}"


def edit_file(
    filepath: str,
    old_text: str = "",
    new_text: str = "",
    confirm=None,
    replace_all: bool = False,
    edits: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Find-and-replace in a file. Supports:
    - Single edit via old_text/new_text
    - Atomic multi-edit via edits=[{old_text, new_text}, ...]
    - replace_all=True to replace all occurrences
    Preserves original line endings (CRLF vs LF) and encoding.
    """
    try:
        if not check_path_sandbox(filepath):
            UI.warn(f"edit_file blocked by project root sandbox: {filepath}")
            return f"Error: edit_file on '{filepath}' outside project root was denied by user."

        UI.status(f"Editing {filepath}...")
        if not os.path.exists(filepath):
            UI.error(f"File '{filepath}' not found.")
            return f"Error: '{filepath}' not found."

        raw = Path(filepath).read_bytes()
        orig_ending = _detect_line_ending(raw)
        with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
            content = f.read()

        if edits:
            edit_list = [(e.get("old_text", ""), e.get("new_text", ""), bool(e.get("replace_all", replace_all))) for e in edits]
        else:
            edit_list = [(old_text, new_text, replace_all)]

        working = content
        for i, (ot, nt, ra) in enumerate(edit_list):
            new_working, err = _apply_single_edit(working, ot, nt, ra)
            if err:
                prefix = f"Edit {i + 1}/{len(edit_list)}: " if len(edit_list) > 1 else ""
                UI.error(f"edit_file: {prefix}{err}")
                return f"Error: {prefix}{err}"
            working = new_working

        new_content = working
        diff_text = unified_diff_text(content.replace("\r\n", "\n"), new_content.replace("\r\n", "\n"), filepath)

        if confirm is not None and not confirm(diff_text):
            UI.warn(f"edit_file cancelled by user for {filepath}.")
            return "edit_file cancelled by user."

        history_push("edit_file", filepath, content)

        write_content = new_content.replace("\r\n", "\n")
        if orig_ending == "\r\n":
            write_content = write_content.replace("\n", "\r\n")
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(write_content)

        INDEX.invalidate_file(filepath)
        invalidate_cache(".")

        syntax_ok, syntax_err = validate_syntax(filepath, new_content)

        if UI.verbose:
            _print_diff(diff_text)

        n_edits = len(edit_list)
        UI.success(f"Updated {filepath} ({n_edits} edit(s))")
        add_xp(25, "surgical edit")

        result_msg = f"Successfully updated {filepath} ({n_edits} edit(s))."
        if not syntax_ok:
            UI.warn(f"Syntax warning in {filepath}: {syntax_err}")
            result_msg += f"\nWarning: Syntax validation failed:\n{syntax_err}"
        return result_msg
    except Exception as e:
        UI.error(f"edit_file failed on {filepath}: {e}")
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
    fast_info = f" (fast: {HOLDER.fast_model_name})" if HOLDER.fast_model_name != model_name else ""
    body = (
        f"{GHOST_MASCOT}\n\n"
        f"[bold white]G H O S T[/bold white]\n"
        f"[dim]{rank} · Level {stats['level']} · {stats['xp']}/{threshold} xp[/dim]\n"
        f"[dim]Model: {model_name}{fast_info} · type a task, or /help for commands[/dim]\n"
        f"[dim italic](Press Ctrl+C to halt AI responses cleanly)[/dim italic]"
    )
    console.print(Panel(Align.center(body), border_style="grey50", padding=(1, 4)))


def render_stats(stats, model_name, permission_mode, auto_test, verbose):
    table = Table(title="Ghost — Status", border_style="grey50", show_header=False)
    table.add_row("Rank", get_rank(stats["level"]))
    table.add_row("Level", str(stats["level"]))
    table.add_row("XP", f"{stats['xp']} / {stats['level']*100}")
    table.add_row("Edits made", str(stats.get("edits", 0)))
    table.add_row("Commands run", str(stats.get("commands", 0)))
    table.add_row("Active Model", str(model_name))
    table.add_row("Fast / Explorer Model", str(HOLDER.fast_model_name))
    table.add_row("Permission mode", permission_mode)
    table.add_row("Auto-test", "on" if auto_test else "off")
    table.add_row("Verbose mode", "on" if verbose else "off")
    console.print(table)


def print_help():
    console.print(
        Panel(
            "\n".join(
                [
                    "[cyan]/stats[/cyan]              — show level, xp, rank, and active models",
                    "[cyan]/rank[/cyan]               — show current rank only",
                    "[cyan]/model [id][/cyan]         — switch primary model",
                    "[cyan]/fastmodel [id][/cyan]     — set fast/cheap model for exploration & summaries",
                    "[cyan]/config[/cyan]             — reconfigure provider / API keys",
                    "[cyan]/upgrade[/cyan]            — auto-upgrade Ghost CLI from GitHub",
                    "[cyan]/mode[/cyan]               — show or set permission mode: [cyan]/mode safe|ask|strict[/cyan]",
                    "[cyan]/verbose [on|off][/cyan]    — toggle detailed output vs quiet terminal UI",
                    "",
                    "[cyan]/init[/cyan]               — generate or refresh GHOST.md project memory file",
                    "[cyan]/plan <task>[/cyan]        — analyze, draft, and critique a structured execution plan",
                    "[cyan]/execute[/cyan]            — carry out approved plan with verification, lint & re-planning",
                    "[cyan]/cancelplan[/cyan]         — discard the current plan",
                    "[cyan]/todo[/cyan]               — display current live checklist",
                    "",
                    "[cyan]/compact[/cyan]            — summarize older turns into a working-state note",
                    "[cyan]/resume [id][/cyan]        — list or resume a saved conversation session",
                    "[cyan]/rewind [n][/cyan]         — rewind conversation by n user turns (default 1)",
                    "",
                    "[cyan]/git status|diff|log|commit|push[/cyan] — git helper commands",
                    "[cyan]/test[/cyan]               — detect and run the project's test suite",
                    "[cyan]/testfix[/cyan]            — run tests, then loop analyze→fix→retest (max 3 tries)",
                    "[cyan]/autotest on|off[/cyan]    — toggle running tests automatically after edits",
                    "",
                    "[cyan]/scan[/cyan]               — quick project scan (languages, markers, file count)",
                    "[cyan]/undo [n][/cyan]           — revert last file edits or restore shadow git checkpoint",
                    "[cyan]/history[/cyan]            — list recent undoable edits",
                    "",
                    "[cyan]/clear[/cyan]              — clear the screen",
                    "[cyan]/help[/cyan]               — this menu",
                    "[cyan]/exit[/cyan]               — quit",
                ]
            ),
            title="Commands",
            border_style="grey50",
        )
    )


# ---------------------------------------------------------------------------
# Testing, Scoped Tests, Lint & Typecheck (Tier 2 Verification)
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


def _find_relevant_pytest_targets(edited_files: List[str], root: str = ".") -> List[str]:
    """Locate test files corresponding to modified Python files."""
    targets = []
    root_path = Path(root)
    for fp in edited_files:
        if not fp or not fp.endswith(".py"):
            continue
        p = Path(fp)
        if p.name.startswith("test_") or p.name.endswith("_test.py"):
            if p.exists():
                targets.append(str(p))
            continue
        stem = p.stem
        candidates = list(root_path.rglob(f"test_{stem}.py")) + list(root_path.rglob(f"{stem}_test.py"))
        for c in candidates:
            if not any(part in IGNORE_DIRS for part in c.parts):
                targets.append(str(c))
    return sorted(set(targets))


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


def run_tests(root=".", timeout=120, target_files: Optional[List[str]] = None):
    cmds = detect_test_commands(root)
    if not cmds:
        return {"ran": False, "success": None, "results": [], "output": "No Python or JavaScript test framework detected."}

    # Scope pytest to relevant test files if edited files are provided
    if target_files and shutil.which("pytest"):
        relevant = _find_relevant_pytest_targets(target_files, root)
        if relevant:
            cmds = [("python (scoped)", ["pytest", "-q"] + relevant)] + [c for c in cmds if c[0] != "python"]

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


def run_lint_and_typecheck(edited_files: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Run available linters/typecheckers (ruff, mypy, tsc) on edited files or project."""
    edited_files = [f for f in (edited_files or []) if f and os.path.exists(f)]
    py_files = [f for f in edited_files if f.endswith(".py")]
    ts_files = [f for f in edited_files if f.endswith((".ts", ".tsx", ".js", ".jsx"))]

    checks = []
    if py_files:
        if shutil.which("ruff"):
            checks.append(("ruff", ["ruff", "check", "--quiet"] + py_files))
        if shutil.which("mypy"):
            checks.append(("mypy", ["mypy", "--ignore-missing-imports", "--no-error-summary"] + py_files))
    if ts_files or (Path("tsconfig.json").exists() and edited_files):
        if Path("tsconfig.json").exists() and shutil.which(" tsc"):
            checks.append(("tsc", ["tsc", "--noEmit"]))

    if not checks:
        return True, "No linter/typechecker issues (or tools not installed)."

    failures = []
    for name, cmd in checks:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if res.returncode != 0:
                out = (res.stdout + "\n" + res.stderr).strip()
                failures.append(f"[{name}] failed:\n{out[:2000]}")
        except Exception:
            continue

    if failures:
        return False, "\n\n".join(failures)
    return True, f"Passed checks: {', '.join(n for n, _ in checks)}"


def format_results(result: dict) -> str:
    if not result["ran"]:
        return result["output"]
    lines = []
    for r in result["results"]:
        status = "PASS" if r["success"] else "FAIL"
        lines.append(f"[{r['lang']}] {status} — {' '.join(r['cmd'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI-compatible client holder & Model Routing (Tier 2)
# ---------------------------------------------------------------------------
class ClientHolder:
    def __init__(self):
        self.cfg = {}
        self.client = None
        self.model_name = DEFAULT_MODEL
        self.fast_model_name = DEFAULT_MODEL
        self.permission_mode = DEFAULT_PERMISSION_MODE
        self.auto_test = False
        self.verbose = False

    def init(self):
        """Explicit startup step so importing the app never triggers network setup."""
        ensure_data_dir()
        self.cfg = get_api_config(load_config())
        self._apply_cfg()

    def _apply_cfg(self):
        self.model_name = self.cfg.get("model", DEFAULT_MODEL)
        self.fast_model_name = self.cfg.get("fast_model", self.model_name)
        self.permission_mode = self.cfg.get("permission_mode", DEFAULT_PERMISSION_MODE)
        self.auto_test = bool(self.cfg.get("auto_test", False))
        self.verbose = bool(self.cfg.get("verbose", False))
        UI.set_verbose(self.verbose)
        self.client = OpenAI(
            base_url=self.cfg.get("base_url"),
            api_key=self.cfg.get("api_key"),
            default_headers={"X-Title": "Ghost"},
        )

    def reconfigure(self):
        self.cfg = get_api_config({})
        self._apply_cfg()

    def switch_model(self, target_model: str = None, fast: bool = False):
        label = "Fast / Explorer Model" if fast else "Primary Model"
        current = self.fast_model_name if fast else self.model_name
        if not target_model:
            console.print(f"\n[bold cyan]Available Models ({label}):[/bold cyan]")
            for key, (m_id, desc) in MODEL_PRESETS.items():
                active_marker = " [bold green](active)[/bold green]" if m_id == current else ""
                console.print(f"[cyan]{key}.[/cyan] {m_id} — [dim]{desc}[/dim]{active_marker}")
            console.print("[cyan]6.[/cyan] Custom model identifier")

            choice = Prompt.ask(
                "\n[bold white]Choose model (1-6 or enter full model ID)[/bold white]",
                default="1",
            ).strip()

            if choice in MODEL_PRESETS:
                target_model = MODEL_PRESETS[choice][0]
            elif choice == "6":
                target_model = Prompt.ask("[bold white]Enter model ID[/bold white]").strip()
            else:
                target_model = choice

        if fast:
            self.fast_model_name = target_model
            self.cfg["fast_model"] = target_model
        else:
            self.model_name = target_model
            self.cfg["model"] = target_model
        save_config(self.cfg)
        console.print(f"\n[bold green]✔ {label} updated to:[/bold green] [bold cyan]{target_model}[/bold cyan]\n")

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

    def set_verbose(self, enabled: bool):
        self.verbose = enabled
        self.cfg["verbose"] = enabled
        UI.set_verbose(enabled)
        save_config(self.cfg)


HOLDER = ClientHolder()


# ---------------------------------------------------------------------------
# Subagent Explorer (Tier 2)
# ---------------------------------------------------------------------------
def spawn_explorer(question: str, max_steps: int = 8) -> str:
    """Spawn a fresh-context read-only explorer subagent using fast_model to investigate the repo."""
    UI.status(f"Subagent exploring: {question[:70]}...")
    sub_system = (
        "You are Ghost Explorer — a fast, read-only codebase research subagent.\n"
        "Use read_file, search_code, list_dir, find_symbol, find_definition, find_references, and find_related_files "
        "to answer the user's question with exact file paths, line numbers, and concise architectural findings.\n"
        "Do NOT propose edits; return a dense, factual summary."
    )
    sub_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sub_system},
        {"role": "user", "content": question},
    ]
    sub_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in READ_ONLY_TOOLS]

    for _ in range(max(1, min(int(max_steps), 15))):
        try:
            resp = stream_completion(
                sub_messages,
                sub_tools,
                _model_override=HOLDER.fast_model_name,
                quiet=not UI.verbose,
            )
        except Exception as e:
            return f"Explorer subagent error: {e}"

        if resp is None:
            break

        if resp.tool_calls:
            sub_messages.append({
                "role": "assistant",
                "content": resp.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in resp.tool_calls
                ],
            })
            for tc in resp.tool_calls:
                out = dispatch_tool_call(tc, planning_restricted=True)
                sub_messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_output(str(out))})
        else:
            summary = resp.content or "(no findings)"
            UI.success("Explorer subagent finished.")
            return summary

    UI.warn("Explorer subagent hit step limit; returning partial findings.")
    for m in reversed(sub_messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return "Explorer subagent finished without a final summary."


# ---------------------------------------------------------------------------
# Agent loop / tool dispatch / streaming / Text Tool-Call Fallback
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

Agent Workflow: Understand → Plan (todo_write) → Inspect (or spawn_explorer) → Implement → Verify → Repair → Done.
- For multi-step tasks in normal turns, maintain a live checklist using `todo_write`.
- For broad codebase questions, use `spawn_explorer` to keep your main context clean.
- Make minimal targeted changes; preserve unrelated code.
- Always verify syntax, lint, and test results after modifying code.

You have real tools:
- Inspection & Search: read_file, list_dir, search_code, find_symbol, find_definition, find_references, find_related_files, scan_project, spawn_explorer, git_status, git_diff, git_log
- Task Tracking: todo_write, todo_read
- Modification: write_file, edit_file
- Execution & Background Processes: run_command, bg_status, bg_kill, run_tests
- Version Control: git_commit, git_push

Use tools instead of describing what you would do.
Risky actions (writes, shell commands, git commit/push) may prompt the user for confirmation before they execute — if one is declined, accept it and adjust course rather than repeating the same call.
"""

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Execute a shell command appropriate for the current OS/shell. Streams output live and returns both stdout and stderr. Set background=true for dev servers or watchers.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 60, max 600)."},
            "background": {"type": "boolean", "description": "Run in background and return a job_id (default false)."}
        }, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "bg_status",
        "description": "Poll status and recent output lines of background processes started via run_command(background=true).",
        "parameters": {"type": "object", "properties": {
            "job_id": {"type": "string", "description": "Optional job_id (e.g. bg_1). Omit to list all."},
            "tail_lines": {"type": "integer", "description": "Number of log lines to return (default 50)."}
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "bg_kill",
        "description": "Terminate a background process by its job_id.",
        "parameters": {"type": "object", "properties": {
            "job_id": {"type": "string", "description": "The background job_id to kill (e.g. bg_1)."}
        }, "required": ["job_id"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file's contents with line numbers. Use offset/limit to read large files in chunks.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string", "description": "Path to the file."},
            "offset": {"type": "integer", "description": "1-based start line (default 1)."},
            "limit": {"type": "integer", "description": "Number of lines to read (default 2000)."}
        }, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files in a directory (recursive, depth-limited).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Search for a pattern across files (uses ripgrep or Python search).",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "find_symbol", "description": "Find symbols (classes, functions, methods) in the codebase index.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "kind": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "find_definition", "description": "Locate definition of a symbol (class, function, method).",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "filepath": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "find_references", "description": "Find usages and references of a symbol across the project.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "find_related_files", "description": "Find related files based on imports, shared symbols, and hierarchy.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "spawn_explorer",
        "description": "Spawn a fresh-context read-only subagent to search and summarize a part of the codebase without cluttering the main conversation.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "Detailed research question or search goal for the explorer."},
            "max_steps": {"type": "integer", "description": "Maximum tool steps (default 8)."}
        }, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Update the live task checklist for multi-step work.",
        "parameters": {"type": "object", "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                    },
                    "required": ["content", "status"]
                }
            }
        }, "required": ["todos"]}}},
    {"type": "function", "function": {
        "name": "todo_read",
        "description": "Read the current live task checklist.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create a new file, or fully overwrite an existing one, with given content.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filepath", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": (
            "Find-and-replace in a file. old_text must match exactly once (add surrounding context if ambiguous). "
            "Use edits=[{old_text, new_text},...] for atomic multi-edits. "
            "Use replace_all=true to replace all occurrences."
        ),
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"},
            "old_text": {"type": "string", "description": "Exact block to find (single edit)."},
            "new_text": {"type": "string", "description": "Replacement text (single edit)."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
            "edits": {
                "type": "array",
                "description": "Multiple atomic edits: [{old_text, new_text}, ...]",
                "items": {"type": "object", "properties": {
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                }, "required": ["old_text", "new_text"]}
            }
        }, "required": ["filepath"]}}},
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
    UI.status("Running tests...")
    result = run_tests(".")
    if not result["ran"]:
        UI.status(result["output"])
        return result["output"]
    summary = format_results(result)
    if result["success"]:
        UI.success("Tests passed")
    else:
        UI.error("Tests failed")
    if UI.verbose:
        UI.detail(summary)
    if not result["success"]:
        failing = [r for r in result["results"] if not r["success"]]
        detail = "\n\n".join(f"--- {r['lang']} output ---\n{r['output']}" for r in failing)
        return truncate_output(f"{summary}\n\n{detail}")
    return summary


def _scan_project_tool() -> str:
    UI.status("Scanning project...")
    result, cached = get_cached_scan(".", scan_project)
    lang_summary = ", ".join(f"{k} ({v})" for k, v in result["languages"].items()) if result["languages"] else "none detected"
    UI.success(f"Scanned {result['file_count']} files ({lang_summary}){' [cached]' if cached else ''}")
    if UI.verbose:
        _print_scan(result, cached)
    lines = [f"Root: {result['root']} (cache {'hit' if cached else 'miss'})",
             f"Files: {result['file_count']}"]
    if result["languages"]:
        lines.append("Languages: " + lang_summary)
    markers = [k for k, v in result["markers"].items() if v]
    if markers:
        lines.append("Markers: " + ", ".join(markers))
    lines.append("Top-level: " + ", ".join(result["top_level"][:20]))
    return "\n".join(lines)


TOOL_MAP = {
    "run_command": run_command,
    "bg_status": bg_status,
    "bg_kill": bg_kill,
    "read_file": read_file,
    "list_dir": list_dir,
    "search_code": search_code,
    "find_symbol": _find_symbol_tool,
    "find_definition": _find_definition_tool,
    "find_references": _find_references_tool,
    "find_related_files": _find_related_files_tool,
    "spawn_explorer": spawn_explorer,
    "todo_write": todo_write,
    "todo_read": todo_read,
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


def _get_required_params(fn_name: str) -> List[str]:
    """Return required parameter names from TOOL_DEFINITIONS for fn_name."""
    for td in TOOL_DEFINITIONS:
        if td["function"]["name"] == fn_name:
            return td["function"].get("parameters", {}).get("required", [])
    return []


def dispatch_tool_call(tool_call, planning_restricted=False, edited=None):
    fn_name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"
    try:
        args = json.loads(raw_args)
    except Exception as je:
        return (
            f"invalid JSON arguments for '{fn_name}': {je}. "
            f"Resend the call with valid JSON. Received: {raw_args[:200]}"
        )

    if not isinstance(args, dict):
        return f"invalid JSON arguments: arguments for '{fn_name}' must be a JSON object, got {type(args).__name__}."

    if planning_restricted and fn_name not in READ_ONLY_TOOLS:
        return f"'{fn_name}' is unavailable in planning mode (read-only tools only)."

    required = _get_required_params(fn_name)
    missing = [p for p in required if p not in args]
    if missing:
        return (
            f"Missing required argument(s) for '{fn_name}': {', '.join(missing)}. "
            f"Required: {required}. Provided keys: {list(args.keys())}."
        )

    risk, description = assess_tool_call(fn_name, args)

    if fn_name in ("write_file", "edit_file"):
        args = dict(args)
        args["confirm"] = _build_confirm(risk, description)
    elif needs_confirmation(HOLDER.permission_mode, risk):
        cmd_for_prefix = str(args.get("command", "")) if fn_name == "run_command" else None
        if not prompt_confirm(description, risk, command_for_prefix=cmd_for_prefix):
            return f"{fn_name} cancelled by user."

    fn = TOOL_MAP.get(fn_name)
    if fn is None:
        return f"Unknown tool: {fn_name}"

    try:
        result = fn(**args)
    except TypeError as e:
        return (
            f"Tool '{fn_name}' received unexpected argument(s): {e}. "
            f"Provided: {list(args.keys())}."
        )
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

    return truncate_output(str(result))


# Text-based tool-call fallback parser (Tier 2)
def _parse_text_tool_calls(text: str) -> Tuple[Optional[str], Optional[List[SimpleNamespace]]]:
    """
    Fallback parser for models that emit tool calls as XML (<tool_call>...</tool_call>,
    <function=name>...</function>) or fenced ```tool_call / ```json blocks.
    Returns (cleaned_text, list_of_tool_calls_or_None).
    """
    if not text:
        return text, None

    calls: List[SimpleNamespace] = []
    cleaned = text

    # Pattern 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    xml_matches = list(re.finditer(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", cleaned))
    for m in xml_matches:
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name") or payload.get("function")
            args = payload.get("arguments", payload.get("parameters", {}))
            if name in TOOL_MAP:
                calls.append(
                    SimpleNamespace(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        function=SimpleNamespace(
                            name=name,
                            arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                        ),
                    )
                )
                cleaned = cleaned.replace(m.group(0), "")
        except Exception:
            continue

    # Pattern 2: <function=tool_name>{"arg": "val"}</function>
    fn_matches = list(re.finditer(r"<function=([a-zA-Z0-9_]+)>\s*(\{[\s\S]*?\})\s*</function>", cleaned))
    for m in fn_matches:
        name = m.group(1)
        if name in TOOL_MAP:
            calls.append(
                SimpleNamespace(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=SimpleNamespace(name=name, arguments=m.group(2).strip()),
                )
            )
            cleaned = cleaned.replace(m.group(0), "")

    # Pattern 3: ```tool_call or ```json containing {"name": "<valid_tool>", "arguments": {...}}
    if not calls:
        block_matches = list(re.finditer(r"```(?:tool_call|json)\s*(\{[\s\S]*?\})\s*```", cleaned))
        for m in block_matches:
            try:
                payload = json.loads(m.group(1))
                if isinstance(payload, dict):
                    name = payload.get("name") or payload.get("tool")
                    args = payload.get("arguments", payload.get("parameters"))
                    if name in TOOL_MAP and isinstance(args, dict):
                        calls.append(
                            SimpleNamespace(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                function=SimpleNamespace(name=name, arguments=json.dumps(args)),
                            )
                        )
                        cleaned = cleaned.replace(m.group(0), "")
            except Exception:
                continue

    if calls:
        return (cleaned.strip() or None), calls
    return text, None


def _is_retryable_error(e: Exception) -> bool:
    """True for 429/5xx, timeouts, and connection errors; False for 400/401/403."""
    msg = str(e).lower()
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return any(k in msg for k in ("timeout", "connection", "rate limit", "429", "500", "502", "503", "504", "529"))


def stream_completion(messages, tool_definitions, _model_override: str = None, quiet: bool = False):
    max_retries = 3
    base_delay = 2.0

    extra_params = {}
    base_url = str(HOLDER.cfg.get("base_url", "")).lower()
    if "nvidia.com" in base_url:
        extra_params["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        extra_params["temperature"] = 1
        extra_params["top_p"] = 0.95

    model = _model_override or HOLDER.model_name
    fallback_model = HOLDER.cfg.get("fallback_model")

    for attempt in range(1, max_retries + 2):
        is_fallback = attempt > max_retries and fallback_model
        current_model = fallback_model if is_fallback else model

        text_parts = []
        tool_call_parts = {}
        printed_header = False
        status = console.status("[cyan]thinking[/cyan]", spinner="dots") if not quiet else None
        if status:
            status.start()

        try:
            stream = HOLDER.client.chat.completions.create(
                model=current_model,
                messages=messages,
                tools=tool_definitions if tool_definitions else None,
                tool_choice="auto" if tool_definitions else None,
                stream=True,
                **extra_params,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    if not quiet:
                        if not printed_header:
                            if status:
                                status.stop()
                            console.print(Rule(style="grey50"))
                            label = f"[bold white]👻 ghost[/bold white] [dim]({current_model})[/dim]"
                            if is_fallback:
                                label += " [yellow](fallback)[/yellow]"
                            console.print(label)
                            printed_header = True
                        console.print(delta.content, end="")
                    text_parts.append(delta.content)

                if delta.tool_calls:
                    if not printed_header and status:
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

            break

        except KeyboardInterrupt:
            if status:
                status.stop()
            if not quiet:
                console.print()
            raise
        except Exception as e:
            if status:
                status.stop()
            if not _is_retryable_error(e):
                raise

            if attempt <= max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                console.print(f"[yellow]API error ({e}). Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})...[/yellow]")
                time.sleep(delay)
                text_parts = []
                tool_call_parts = {}
                continue
            elif fallback_model and not is_fallback:
                console.print(f"[yellow]All retries failed. Trying fallback model: {fallback_model}...[/yellow]")
                continue
            else:
                raise
        finally:
            if status:
                status.stop()

    if text_parts and not quiet:
        console.print()
        console.print(Rule(style="grey50"))

    full_text = "".join(text_parts) or None
    tool_calls = None
    if tool_call_parts:
        tool_calls = [
            SimpleNamespace(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
            )
            for _, slot in sorted(tool_call_parts.items())
        ]
    elif full_text and tool_definitions:
        # Fallback for open models emitting XML/JSON tool calls in plain text
        full_text, fallback_calls = _parse_text_tool_calls(full_text)
        if fallback_calls:
            tool_calls = fallback_calls

    return SimpleNamespace(content=full_text, tool_calls=tool_calls)


def _tool_call_hash(tc) -> str:
    """Stable hash of (tool name, normalized args) for loop detection."""
    try:
        args = json.loads(tc.function.arguments or "{}")
    except Exception:
        args = {"_raw": tc.function.arguments or ""}
    key = json.dumps({"n": tc.function.name, "a": args}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def run_turn(messages, tool_definitions, planning_restricted=False, edited=None):
    """Streams one assistant turn, running any tool calls, until the model replies with plain text."""
    max_steps = HOLDER.cfg.get("max_steps", 40)
    steps = 0
    call_counts: Dict[str, int] = {}

    while True:
        if steps >= max_steps:
            console.print(f"\n[yellow]▲ Step budget exhausted ({max_steps} model calls). Stopping turn.[/yellow]")
            if edited:
                _append_interrupt_note(messages, edited)
            return False

        steps += 1
        # Route to fast_model if in read-only planning/exploration mode
        model_to_use = HOLDER.fast_model_name if planning_restricted else HOLDER.model_name

        try:
            response_msg = stream_completion(messages, tool_definitions, _model_override=model_to_use)
        except KeyboardInterrupt:
            console.print("\n[yellow]▲ Generation halted by user (Ctrl+C).[/yellow]")
            if edited:
                _append_interrupt_note(messages, edited)
            return False
        except Exception as e:
            console.print(f"[bold red]API error:[/bold red] {e}")
            if edited:
                _append_interrupt_note(messages, edited)
            return False

        if response_msg is None:
            if edited:
                _append_interrupt_note(messages, edited)
            return False

        if response_msg.tool_calls:
            all_read_only = all(tc.function.name in READ_ONLY_TOOLS for tc in response_msg.tool_calls)

            blocked_ids: Dict[str, str] = {}
            abort_turn = False
            for tc in response_msg.tool_calls:
                h = _tool_call_hash(tc)
                call_counts[h] = call_counts.get(h, 0) + 1
                if call_counts[h] >= 5:
                    console.print(f"\n[bold red]▲ Loop detected: '{tc.function.name}' called 5+ times identically. Aborting turn.[/bold red]")
                    abort_turn = True
                    break
                elif call_counts[h] >= 3:
                    blocked_ids[tc.id] = (
                        f"you're repeating yourself, try a different approach (called '{tc.function.name}' "
                        f"with identical arguments {call_counts[h]} times)."
                    )

            messages.append({
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in response_msg.tool_calls
                ],
            })

            if abort_turn:
                for tc in response_msg.tool_calls:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Turn aborted: repeated identical calls detected."})
                if edited:
                    _append_interrupt_note(messages, edited)
                return False

            if all_read_only and len(response_msg.tool_calls) > 1:
                results: Dict[str, str] = {}

                def _exec_read(tc):
                    if tc.id in blocked_ids:
                        return tc.id, blocked_ids[tc.id]
                    try:
                        out = dispatch_tool_call(tc, planning_restricted, edited)
                        return tc.id, str(out)
                    except KeyboardInterrupt:
                        return tc.id, "Tool execution interrupted by user (Ctrl+C)."
                    except Exception as exc:
                        return tc.id, f"Tool error: {exc}"

                with ThreadPoolExecutor(max_workers=min(8, len(response_msg.tool_calls))) as pool:
                    futures = {pool.submit(_exec_read, tc): tc for tc in response_msg.tool_calls}
                    for fut in as_completed(futures):
                        tid, out = fut.result()
                        results[tid] = out

                for tc in response_msg.tool_calls:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_output(results.get(tc.id, "(no output)"))})
            else:
                interrupted = False
                for tool_call in response_msg.tool_calls:
                    if tool_call.id in blocked_ids:
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": blocked_ids[tool_call.id]})
                        continue
                    try:
                        tool_output = dispatch_tool_call(tool_call, planning_restricted, edited)
                    except KeyboardInterrupt:
                        console.print("\n[yellow]▲ Tool execution cancelled by user (Ctrl+C).[/yellow]")
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Tool execution interrupted by user (Ctrl+C)."})
                        interrupted = True
                        break
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": truncate_output(str(tool_output))})

                if interrupted:
                    if edited:
                        _append_interrupt_note(messages, edited)
                    return False
        else:
            messages.append({"role": "assistant", "content": response_msg.content})
            return True


def _append_interrupt_note(messages: List[dict], edited: List[str]):
    """After an interrupted or failed turn, record which files were already modified so the model doesn't forget."""
    files = ", ".join(sorted(set(f for f in edited if f))) if edited else "none"
    messages.append({
        "role": "user",
        "content": f"[system] interrupted by user; these files were modified: {files}",
    })


def _msg_tokens(msg: dict) -> int:
    """Estimate tokens for a single message dict."""
    parts = []
    content = msg.get("content") or ""
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", ""))
    for tc in msg.get("tool_calls", []):
        parts.append(tc.get("function", {}).get("arguments", ""))
    return _count_tokens(" ".join(parts)) + 4


def _total_tokens(messages: List[dict]) -> int:
    return sum(_msg_tokens(m) for m in messages)


def _compact_messages(messages: List[dict]) -> List[dict]:
    """Summarize old messages into a working-state block using fast_model."""
    if len(messages) < 4:
        return messages
    system = messages[0]
    keep_tail = messages[-6:]
    to_summarize = messages[1:-6]
    if not to_summarize:
        return messages

    transcript_parts = []
    for m in to_summarize:
        role = m.get("role", "")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if role == "assistant" and m.get("tool_calls"):
            calls = ", ".join(tc["function"]["name"] for tc in m.get("tool_calls", []))
            transcript_parts.append(f"[assistant called tools: {calls}]")
        elif content:
            transcript_parts.append(f"[{role}]: {content[:800]}")

    transcript = "\n".join(transcript_parts)
    summary_prompt = (
        "Summarize the following agent conversation into a structured working-state note "
        "(200–400 words) with exact headings:\n"
        "- Goal\n- Files Touched\n- Key Decisions\n- Open Problems\n"
        "Be terse and factual.\n\n"
        f"<transcript>\n{transcript[:12000]}\n</transcript>"
    )

    UI.status("Compacting context into working-state summary...")
    try:
        resp = HOLDER.client.chat.completions.create(
            model=HOLDER.fast_model_name,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=700,
            stream=False,
        )
        summary = resp.choices[0].message.content or "(summary unavailable)"
    except Exception as e:
        UI.warn(f"Context compaction failed ({e}), falling back to tail trim.")
        return _fallback_trim(messages)

    summary_msg = {
        "role": "user",
        "content": f"[Working state summary — earlier context compacted]\n{summary}",
    }
    UI.success("Context compacted.")
    return [system, summary_msg] + keep_tail


def _fallback_trim(messages: List[dict]) -> List[dict]:
    if len(messages) <= MAX_CONTEXT_MESSAGES + 1:
        return messages
    system = messages[0]
    tail = messages[1:][-MAX_CONTEXT_MESSAGES:]
    for i, m in enumerate(tail):
        if m.get("role") == "user":
            tail = tail[i:]
            break
    return [system] + tail


def trim_messages(messages: List[dict]) -> List[dict]:
    """Token-aware context management. Compacts when exceeding 70% of context_limit."""
    context_limit = HOLDER.cfg.get("context_limit", 64_000)
    threshold = int(context_limit * 0.70)
    total = _total_tokens(messages)
    if total <= threshold:
        return messages
    UI.warn(f"Context at ~{total:,} tokens (70% threshold of {context_limit:,}). Compacting...")
    return _compact_messages(messages)


def compact_context_now(messages: List[dict]) -> List[dict]:
    """Manual /compact trigger."""
    before = _total_tokens(messages)
    compacted = _compact_messages(messages)
    after = _total_tokens(compacted)
    console.print(f"[dim]Context compacted: ~{before:,} → ~{after:,} tokens.[/dim]")
    return compacted


# ---------------------------------------------------------------------------
# Plan → Critique → Execute & Stronger Verification (Tier 2)
# ---------------------------------------------------------------------------
def critique_plan(plan: StructuredPlan, messages: List[dict]) -> StructuredPlan:
    """Run a second pass where the fast/critique model reviews the plan against codebase constraints."""
    UI.status("Critiquing plan against codebase...")
    critique_prompt = (
        "Review the drafted execution plan below for missing edge cases, wrong file paths, "
        "missing dependencies, or verification gaps. Return a refined JSON plan block in the exact same schema, "
        "plus a one-sentence 'critique' field summarizing any improvements made.\n\n"
        f"Draft Plan:\n{plan.raw_text}"
    )
    crit_messages = list(messages) + [{"role": "user", "content": critique_prompt}]
    try:
        resp = stream_completion(
            crit_messages,
            PLANNING_TOOL_DEFINITIONS,
            _model_override=HOLDER.fast_model_name,
            quiet=not UI.verbose,
        )
        if resp and resp.content:
            refined = parse_plan_response(resp.content, default_task=plan.objective)
            if refined.tasks:
                json_match = re.search(r"\"critique\"\s*:\s*\"([^\"]+)\"", resp.content)
                refined.critique = json_match.group(1) if json_match else "Verified & refined by critique pass."
                UI.success("Plan critique complete.")
                return refined
    except Exception as e:
        UI.warn(f"Critique pass skipped ({e}).")
    return plan


def review_git_diff_pass(messages: List[dict], edited: List[str]) -> bool:
    """Final reviewer pass over git diff before declaring work done."""
    if not edited or not _is_git_repo():
        return True
    diff = git_diff()
    if not diff or diff == "No changes.":
        return True
    UI.status("Running final reviewer pass over git diff...")
    review_prompt = (
        "Review the following `git diff` of your recent changes. Check for unintended edits, "
        "leftover debug statements, broken imports, or obvious bugs.\n"
        "If everything looks clean, reply starting with 'LGTM:'. "
        "If you spot a bug, fix it immediately using edit_file.\n\n"
        f"```diff\n{diff[:5000]}\n```"
    )
    messages.append({"role": "user", "content": review_prompt})
    return run_turn(messages, TOOL_DEFINITIONS, edited=edited)


def verify_and_repair_task(task: TaskItem, edited: List[str], messages: List[dict], max_attempts: int = 3) -> bool:
    """Performs verification (syntax checks, lint/typecheck, scoped tests) and up to 3 repair attempts."""
    for attempt in range(1, max_attempts + 1):
        task.attempts = attempt
        syntax_errors = []
        for fpath in set(edited):
            if fpath and os.path.exists(fpath):
                ok, err = validate_syntax(fpath)
                if not ok:
                    syntax_errors.append(err)

        if syntax_errors:
            error_report = "Syntax verification failed:\n" + "\n".join(syntax_errors)
            UI.warn(f"Syntax error detected (attempt {attempt}/{max_attempts}). Initiating repair...")
            if attempt == max_attempts:
                console.print(f"[red]{error_report}[/red]")
                return False

            repair_prompt = f"{error_report}\n\nAnalyze and fix the syntax error immediately using edit_file/write_file."
            messages.append({"role": "user", "content": repair_prompt})
            if not run_turn(messages, TOOL_DEFINITIONS, edited=edited):
                return False
            continue

        # Lint & typecheck (ruff, mypy, tsc)
        lint_ok, lint_report = run_lint_and_typecheck(edited)
        if not lint_ok:
            UI.warn(f"Lint/typecheck issue detected (attempt {attempt}/{max_attempts}). Repairing...")
            if attempt < max_attempts:
                messages.append({
                    "role": "user",
                    "content": f"Lint / typecheck failed on modified files:\n{lint_report}\n\nFix these issues using edit_file.",
                })
                if not run_turn(messages, TOOL_DEFINITIONS, edited=edited):
                    return False
                continue

        needs_test = any(m in ("run_tests", "test") for m in task.verification_methods) or HOLDER.auto_test
        if needs_test:
            UI.status(f"Running verification tests (attempt {attempt}/{max_attempts})...")
            test_res = run_tests(".", target_files=edited)
            if test_res["ran"] and not test_res["success"]:
                UI.warn(f"Test verification failed (attempt {attempt}/{max_attempts}). Initiating repair...")
                if attempt == max_attempts:
                    console.print(format_results(test_res))
                    return False

                failing = [r for r in test_res["results"] if not r["success"]]
                fail_detail = "\n\n".join(f"[{r['lang']}] {' '.join(r['cmd'])}\n{r['output']}" for r in failing)
                repair_prompt = (
                    f"Tests failed during task verification:\n{fail_detail[:3000]}\n\n"
                    f"Diagnose the failure and repair the code using your tools. Keep commentary concise."
                )
                messages.append({"role": "user", "content": repair_prompt})
                if not run_turn(messages, TOOL_DEFINITIONS, edited=edited):
                    return False
                continue

        return True

    return False


def execute_structured_plan(plan: StructuredPlan, messages: List[dict]) -> bool:
    """Executes an approved plan step-by-step with dependency checking, verification, re-planning, and diff review."""
    if not plan or not plan.tasks:
        console.print("[yellow]No tasks found in plan to execute.[/yellow]")
        return False

    create_shadow_checkpoint(f"plan: {plan.objective}")
    console.print(f"\n[bold cyan][•] Executing Plan:[/bold cyan] {plan.objective}\n")

    all_edited: List[str] = []
    idx = 0
    replan_count = 0

    while idx < len(plan.tasks):
        task = plan.tasks[idx]
        deps_ok = True
        for dep_id in task.dependencies:
            dep_task = plan.get_task(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                deps_ok = False
                break

        if not deps_ok:
            task.status = TaskStatus.BLOCKED
            UI.warn(f"Task {task.id} BLOCKED (prerequisite tasks incomplete)")
            idx += 1
            continue

        task.status = TaskStatus.RUNNING
        console.print(f"[bold white][•] Task {task.id}/{len(plan.tasks)}:[/bold white] {task.description}")

        task_prompt = (
            f"Step {task.id} of {len(plan.tasks)}: {task.description}\n"
            f"Target file(s): {', '.join(task.affected_files) if task.affected_files else 'As needed'}\n"
            f"Expected changes: {task.expected_changes or 'Complete the task as specified'}\n"
            f"Verification: {', '.join(task.verification_methods)}\n\n"
            f"Inspect before editing, make minimal targeted changes using write_file/edit_file, and verify your changes. "
            f"Keep commentary concise."
        )
        messages.append({"role": "user", "content": task_prompt})

        edited: List[str] = []
        turn_ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
        all_edited.extend(edited)
        if not turn_ok:
            task.status = TaskStatus.FAILED
            UI.error(f"Task {task.id} execution aborted.")
            return False

        task_ok = verify_and_repair_task(task, edited, messages, max_attempts=3)
        if not task_ok:
            task.status = TaskStatus.FAILED
            if replan_count < 1:
                replan_count += 1
                UI.warn(f"Task {task.id} failed verification. Attempting automatic re-plan of remaining work...")
                replan_prompt = (
                    f"Task {task.id} ('{task.description}') failed verification. "
                    "Re-evaluate the current codebase state and output an updated JSON plan for the remaining tasks."
                )
                messages.append({"role": "user", "content": replan_prompt})
                if run_turn(messages, PLANNING_TOOL_DEFINITIONS, planning_restricted=True):
                    new_plan = parse_plan_response(messages[-1].get("content") or "", default_task=plan.objective)
                    if new_plan.tasks:
                        plan.tasks = plan.tasks[:idx] + new_plan.tasks
                        UI.success(f"Re-planned {len(new_plan.tasks)} remaining task(s). Continuing execution...")
                        continue
            UI.error(f"Task {task.id} failed verification. Halting execution.")
            render_plan_summary(plan)
            return False

        task.status = TaskStatus.COMPLETED
        UI.success(f"Task {task.id} verified: COMPLETED\n")
        idx += 1

    review_git_diff_pass(messages, all_edited)
    UI.success(f"Plan Execution Complete ({len(plan.tasks)}/{len(plan.tasks)} tasks successful).\n")
    render_plan_summary(plan)
    PLAN_STATE.clear()
    return True


def run_test_fix_loop(messages, initial_result=None, max_retries=3):
    result = initial_result
    try:
        for attempt in range(1, max_retries + 1):
            if result is None:
                UI.status(f"Re-running tests (attempt {attempt}/{max_retries})...")
                result = run_tests(".")
            if not result["ran"]:
                console.print(f"[dim]{result['output']}[/dim]")
                return
            if result["success"]:
                UI.success(f"Tests passing (attempt {attempt}/{max_retries}).")
                add_xp(20, "green build")
                return
            UI.warn(f"Tests failing — attempt {attempt}/{max_retries}")
            console.print(format_results(result))
            if attempt == max_retries:
                UI.error("Max retries reached. Leaving for manual review.")
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
                UI.error("Fix attempt aborted.")
                return
            result = None
    except KeyboardInterrupt:
        console.print("\n[yellow]▲ Test-fix loop cancelled by user (Ctrl+C).[/yellow]")


def maybe_autotest(messages, edited):
    if not edited:
        return
    lint_ok, lint_msg = run_lint_and_typecheck(edited)
    if not lint_ok:
        UI.warn("Lint / typecheck warnings on modified files:")
        console.print(f"[dim yellow]{lint_msg[:1500]}[/dim yellow]")

    if not HOLDER.auto_test:
        return
    UI.status("auto-test: running suite after edits...")
    result = run_tests(".", target_files=edited)
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


# ---------------------------------------------------------------------------
# Project Memory (`GHOST.md`) & `/init` + Session Persistence (Tier 2)
# ---------------------------------------------------------------------------
def load_ghost_md() -> str:
    """Load GHOST.md if present in the project root."""
    if GHOST_MD_FILE.exists():
        try:
            content = GHOST_MD_FILE.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return f"\n\n--- Project Memory (GHOST.md) ---\n{content[:6000]}\n---"
        except Exception:
            pass
    return ""


def generate_ghost_md() -> str:
    """Generate a GHOST.md project memory file from codebase scan and test detection."""
    UI.status("Analyzing project to generate GHOST.md...")
    scan_res, _ = get_cached_scan(".", scan_project, force=True)
    test_cmds = detect_test_commands(".")
    langs = ", ".join(f"{k} ({v} files)" for k, v in scan_res["languages"].items()) or "Unknown"
    tests_str = "\n".join(f"- **{lang}**: `{' '.join(cmd)}`" for lang, cmd in test_cmds) or "- No automated test suite detected yet."

    linters = []
    for tool in ("ruff", "mypy", "pytest", "eslint", "tsc"):
        if shutil.which(tool):
            linters.append(tool)

    template = f"""# GHOST.md — Project Memory & Conventions

## Overview
- **Root**: `{scan_res['root']}`
- **Primary Languages**: {langs}
- **Top-Level Structure**: `{', '.join(scan_res['top_level'][:25])}`

## Commands
### Testing
{tests_str}

### Linting & Type Checking
- Available on PATH: `{', '.join(linters) if linters else 'none detected'}`

## Code Style & Conventions
- Read target files before applying edits.
- Keep changes minimal, modular, and well-tested.
- Preserve existing line endings and formatting conventions.

## Architecture & Gotchas
- Add project-specific notes, domain invariants, or tricky edge cases here.
"""
    GHOST_MD_FILE.write_text(template, encoding="utf-8")
    UI.success(f"Generated {GHOST_MD_FILE.resolve()}")
    return template


class SessionManager:
    def __init__(self):
        ensure_data_dir()
        self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        self.filepath = SESSIONS_DIR / f"session_{self.session_id}.json"

    def save(self, messages: List[dict]):
        try:
            ensure_data_dir()
            payload = {
                "id": self.session_id,
                "cwd": os.getcwd(),
                "updated_at": time.time(),
                "model": HOLDER.model_name,
                "todos": TODO_STATE.items,
                "messages": messages,
            }
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        ensure_data_dir()
        files = sorted(SESSIONS_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                user_msgs = [m.get("content", "") for m in data.get("messages", []) if m.get("role") == "user" and not str(m.get("content", "")).startswith("[")]
                preview = (user_msgs[0][:60] + "...") if user_msgs else "(empty session)"
                out.append({
                    "id": data.get("id", f.stem),
                    "file": f,
                    "updated_at": data.get("updated_at", f.stat().st_mtime),
                    "cwd": data.get("cwd", ""),
                    "preview": preview,
                    "msg_count": len(data.get("messages", [])),
                })
            except Exception:
                continue
        return out

    def load(self, target_id: Optional[str] = None) -> Optional[List[dict]]:
        sessions = self.list_sessions(limit=20)
        if not sessions:
            console.print("[yellow]No saved sessions found.[/yellow]")
            return None

        chosen = None
        if target_id:
            for s in sessions:
                if target_id in s["id"]:
                    chosen = s
                    break
            if not chosen:
                console.print(f"[red]Session '{target_id}' not found.[/red]")
                return None
        else:
            table = Table(title="Saved Sessions", border_style="grey50")
            table.add_column("#", width=4)
            table.add_column("Session ID", style="cyan")
            table.add_column("Messages", width=10)
            table.add_column("First Prompt", style="white")
            for idx, s in enumerate(sessions, 1):
                table.add_row(str(idx), s["id"], str(s["msg_count"]), s["preview"])
            console.print(table)
            choice = Prompt.ask("Select session # to resume (or 'c' to cancel)", default="1").strip()
            if choice.lower() == "c" or not choice.isdigit():
                return None
            c_idx = int(choice) - 1
            if 0 <= c_idx < len(sessions):
                chosen = sessions[c_idx]

        if not chosen:
            return None

        data = json.loads(chosen["file"].read_text(encoding="utf-8"))
        self.session_id = data.get("id", self.session_id)
        self.filepath = chosen["file"]
        if data.get("todos"):
            TODO_STATE.items = data["todos"]
        UI.success(f"Resumed session {self.session_id} ({len(data.get('messages', []))} messages).")
        return data.get("messages")


SESSION_MGR = SessionManager()


def rewind_messages(messages: List[dict], n_turns: int = 1) -> List[dict]:
    """Rewind conversation by n user turns."""
    if len(messages) <= 1:
        return messages
    user_indices = [
        i for i, m in enumerate(messages)
        if i > 0 and m.get("role") == "user" and not str(m.get("content", "")).startswith("[")
    ]
    if not user_indices:
        return messages[:1]
    cut_idx = user_indices[max(0, len(user_indices) - max(1, n_turns))]
    UI.success(f"Rewound conversation to before turn #{len(user_indices) - max(1, n_turns) + 1} (kept {cut_idx} message(s)).")
    return messages[:cut_idx]


def handle_slash_command(user_input, messages):
    """Returns 'exit' to quit, True if the input was a handled command, False otherwise."""
    low = user_input.lower().strip()

    if low in ("exit", "quit", "q", "/exit"):
        flush_stats()
        SESSION_MGR.save(messages)
        console.print("[dim]Ghost fades out.[/dim]")
        return "exit"

    if low == "/upgrade":
        upgrade_ghost()
        return True

    if low == "/stats":
        render_stats(load_stats(), HOLDER.model_name, HOLDER.permission_mode, HOLDER.auto_test, HOLDER.verbose)
        return True

    if low == "/rank":
        s = load_stats()
        console.print(f"[cyan]{get_rank(s['level'])}[/cyan]")
        return True

    if low == "/model" or low.startswith("/model "):
        parts = user_input.split(maxsplit=1)
        HOLDER.switch_model(parts[1].strip() if len(parts) > 1 else None, fast=False)
        return True

    if low == "/fastmodel" or low.startswith("/fastmodel "):
        parts = user_input.split(maxsplit=1)
        HOLDER.switch_model(parts[1].strip() if len(parts) > 1 else None, fast=True)
        return True

    if low == "/config":
        HOLDER.reconfigure()
        render_header(load_stats(), HOLDER.model_name)
        return True

    if low == "/init":
        generate_ghost_md()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = SYSTEM_PROMPT + _build_env_context() + load_ghost_md()
        return True

    if low == "/todo":
        console.print(TODO_STATE.format_text())
        return True

    if low == "/compact":
        compacted = compact_context_now(messages)
        messages[:] = compacted
        SESSION_MGR.save(messages)
        return True

    if low == "/resume" or low.startswith("/resume "):
        parts = user_input.split(maxsplit=1)
        loaded = SESSION_MGR.load(parts[1].strip() if len(parts) > 1 else None)
        if loaded:
            messages[:] = loaded
        return True

    if low == "/rewind" or low.startswith("/rewind "):
        parts = user_input.split()
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        messages[:] = rewind_messages(messages, n)
        SESSION_MGR.save(messages)
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

    if low.startswith("/verbose"):
        parts = user_input.split()
        if len(parts) < 2:
            console.print(f"[dim]Verbose mode: {'on' if HOLDER.verbose else 'off'}. Usage: /verbose on|off[/dim]")
        elif parts[1].lower() in ("on", "true", "1"):
            HOLDER.set_verbose(True)
            console.print("[green]Verbose mode turned on. Detailed tool outputs and diffs will be displayed.[/green]")
        elif parts[1].lower() in ("off", "false", "0"):
            HOLDER.set_verbose(False)
            console.print("[green]Verbose mode turned off. Quiet terminal UI active.[/green]")
        else:
            console.print("[red]Invalid option. Usage: /verbose on|off[/red]")
        return True

    if low.startswith("/plan"):
        task = user_input[len("/plan"):].strip()
        if not task:
            console.print("[yellow]Usage: /plan <task description>[/yellow]")
            return True
        PLAN_STATE.start(task)
        INDEX.ensure_indexed()
        UI.status(f"Drafting structured plan for: {task}")
        messages.append({"role": "system", "content": PLANNING_SYSTEM_SUFFIX})
        messages.append({"role": "user", "content": f"Create a structured execution plan for: {task}"})
        ok = run_turn(messages, PLANNING_TOOL_DEFINITIONS, planning_restricted=True)
        if ok:
            plan_text = messages[-1].get("content") or ""
            structured_plan = parse_plan_response(plan_text, default_task=task)
            structured_plan = critique_plan(structured_plan, messages)
            PLAN_STATE.finish_drafting(structured_plan)
            render_structured_plan(structured_plan)
        else:
            PLAN_STATE.clear()
            UI.error("Planning cancelled or aborted.")
        flush_stats()
        SESSION_MGR.save(messages)
        return True

    if low == "/execute":
        if not PLAN_STATE.has_plan():
            console.print("[yellow]No structured plan to execute. Use /plan <task> first.[/yellow]")
            return True
        plan = PLAN_STATE.structured_plan
        execute_structured_plan(plan, messages)
        flush_stats()
        SESSION_MGR.save(messages)
        return True

    if low == "/cancelplan":
        if PLAN_STATE.has_plan() or PLAN_STATE.active:
            PLAN_STATE.clear()
            UI.success("Plan discarded.")
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
            console.print(f"[cyan]Reverted: {', '.join(reverted)}[/cyan]")
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


def _build_env_context() -> str:
    """Build a one-time environment context block injected after the system prompt."""
    lines = ["\n\n--- Environment ---"]
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    shell_env = os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"
    lines.append(f"Shell: {shell_env}")
    lines.append(f"CWD: {os.getcwd()}")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M %Z')}")
    if shutil.which("git"):
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=5, text=True
            ).strip()
            lines.append(f"Git branch: {branch}")
        except Exception:
            pass
    shell_name = os.path.basename(shell_env).lower()
    lines.append(f"Use {'PowerShell/cmd' if platform.system() == 'Windows' else shell_name} syntax for shell commands.")
    lines.append("---")
    return "\n".join(lines)


def run_agent():
    HOLDER.init()
    INDEX.ensure_indexed()
    render_header(load_stats(), HOLDER.model_name)
    system_content = SYSTEM_PROMPT + _build_env_context() + load_ghost_md()
    messages = [{"role": "system", "content": system_content}]

    while True:
        messages = trim_messages(messages)
        try:
            user_input = Prompt.ask("\n[bold white]you[/bold white]").strip()
        except KeyboardInterrupt:
            console.print("\n[dim](Press Ctrl+D or type /exit to quit)[/dim]")
            continue
        except EOFError:
            flush_stats()
            SESSION_MGR.save(messages)
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

        create_shadow_checkpoint(user_input[:60])
        messages.append({"role": "user", "content": user_input})

        edited: List[str] = []
        ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
        if ok:
            maybe_autotest(messages, edited)

        SESSION_MGR.save(messages)
        flush_stats()


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\nGhost fades out.")
