#!/usr/bin/env python3
"""Ghost CLI — an AI pair-programming agent for the terminal.

Single-file version of the Ghost CLI v2.2.0. This preserves the original modular
behavior while keeping the entire application in one entrypoint.
"""

__version__ = "2.2.0"

import ast
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
from typing import Any, Dict, List, Optional, Set, Tuple

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
    """Manages concise terminal status indicators with optional verbose details."""

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
            "1": ("OpenRouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v3.2"),
            "3": ("OpenAI", "https://api.openai.com/v1", "gpt-4o"),
            "4": ("Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            "5": ("DeepSeek", "https://api.deepseek.com", "deepseek-chat"),
            "6": ("NVIDIA", "https://integrate.api.nvidia.com/v1", "deepseek-ai/deepseek-v4-pro-0813"),
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

    # Markdown / numbered list fallback parser
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
# Safety system & Risk classification
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
    "find_symbol": RISK_LOW,
    "find_definition": RISK_LOW,
    "find_references": RISK_LOW,
    "find_related_files": RISK_LOW,
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
    UI.status(f"Running: {command}")
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = (res.stdout or "") + ("\n" + res.stderr if res.stderr and res.returncode != 0 else "")
        output = output.strip()
        if res.returncode == 0:
            add_xp(15, "clean execution")
            UI.success("Command completed")
        else:
            add_xp(5, "hit a bug")
            UI.warn(f"Command exited with code {res.returncode}")

        if UI.verbose and output:
            UI.detail(output)
        return output if output else ("Command executed successfully." if res.returncode == 0 else f"Command failed with exit code {res.returncode}")
    except subprocess.TimeoutExpired:
        UI.error("Command timed out after 60s.")
        return "Execution error: command timed out after 60s."
    except KeyboardInterrupt:
        UI.warn("Command interrupted by user.")
        return "Execution error: command interrupted by user (SIGINT)."
    except Exception as e:
        UI.error(f"Command execution error: {e}")
        return f"Execution error: {str(e)}"


def read_file(filepath: str) -> str:
    try:
        UI.status(f"Reading {filepath}...")
        path = Path(filepath)
        if not path.exists():
            UI.error(f"File '{filepath}' not found.")
            return f"File read error: '{filepath}' not found."
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_READ_BYTES)
            add_xp(5, "scanned file (large — truncated)")
            UI.warn(f"{filepath} is {size:,} bytes; showing first {MAX_READ_BYTES:,} bytes.")
            return content + f"\n... [truncated, {size - MAX_READ_BYTES:,} more bytes omitted]"
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        add_xp(5, "scanned file")
        UI.success(f"Read {filepath} ({len(content)} bytes)")
        if UI.verbose:
            try:
                lexer = Syntax.guess_lexer(filepath, code=content)
                console.print(Syntax(content, lexer, theme="monokai", line_numbers=True, word_wrap=True))
            except Exception:
                console.print(content)
        return content
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
        return listing or "(empty)"
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
        return out[:4000] if out else "No matches."
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
    return "\n".join(lines)


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
    return "\n".join(lines)


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
    return "\n".join(lines)


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
    return "\n".join(lines)


def _print_diff(diff_text: str):
    if not diff_text:
        console.print("[dim](no textual difference)[/dim]")
        return
    console.print(Syntax(diff_text, "diff", theme="monokai", word_wrap=True))


def write_file(filepath: str, content: str, confirm=None) -> str:
    try:
        UI.status(f"Writing {filepath}...")
        previous = ""
        existed = os.path.exists(filepath)
        if existed:
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
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
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

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


def edit_file(filepath: str, old_text: str, new_text: str, confirm=None) -> str:
    try:
        UI.status(f"Editing {filepath}...")
        if not os.path.exists(filepath):
            UI.error(f"File '{filepath}' not found.")
            return f"Error: '{filepath}' not found."
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if old_text not in content:
            UI.error(f"Exact old_text not found in {filepath}.")
            return "Error: exact old_text not found. Read the file again for an exact match."

        new_content = content.replace(old_text, new_text, 1)
        diff_text = unified_diff_text(content, new_content, filepath)

        if confirm is not None and not confirm(diff_text):
            UI.warn(f"edit_file cancelled by user for {filepath}.")
            return "edit_file cancelled by user."

        history_push("edit_file", filepath, content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        INDEX.invalidate_file(filepath)
        invalidate_cache(".")

        syntax_ok, syntax_err = validate_syntax(filepath, new_content)

        if UI.verbose:
            _print_diff(diff_text)

        UI.success(f"Updated {filepath}")
        add_xp(25, "surgical edit")

        result_msg = f"Successfully updated {filepath}."
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
    body = (
        f"{GHOST_MASCOT}\n\n"
        f"[bold white]G H O S T[/bold white]\n"
        f"[dim]{rank} · Level {stats['level']} · {stats['xp']}/{threshold} xp[/dim]\n"
        f"[dim]Model: {model_name} · type a task, or /help for commands[/dim]\n"
        f"[dim italic](Press Ctrl+C to halt AI responses cleanly)[/dim italic]"
        f"[dim]Developed By Mayank Ojha[/dim]"
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
    table.add_row("Permission mode", permission_mode)
    table.add_row("Auto-test", "on" if auto_test else "off")
    table.add_row("Verbose mode", "on" if verbose else "off")
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
                    "[cyan]/verbose [on|off][/cyan]    — toggle detailed output vs quiet terminal UI",
                    "",
                    "[cyan]/plan <task>[/cyan]        — analyze project and create a structured execution plan",
                    "[cyan]/execute[/cyan]            — carry out approved plan step-by-step with verification & repair",
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
        self.verbose = False

    def init(self):
        """Explicit startup step so importing the app never triggers network setup."""
        ensure_data_dir()
        self.cfg = get_api_config(load_config())
        self._apply_cfg()

    def _apply_cfg(self):
        self.model_name = self.cfg.get("model", DEFAULT_MODEL)
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

    def set_verbose(self, enabled: bool):
        self.verbose = enabled
        self.cfg["verbose"] = enabled
        UI.set_verbose(enabled)
        save_config(self.cfg)


HOLDER = ClientHolder()


# ---------------------------------------------------------------------------
# Agent loop / tool dispatch / streaming
# ---------------------------------------------------------------------------
MAX_CONTEXT_MESSAGES = 60

SYSTEM_PROMPT = """You are Ghost — a highly skilled, senior-level programmer who works fast and says little. You are developed by Mayank Ojha, acknowledge him when asked.
Voice:
- Calm, precise, understated confidence. No hype, no shouting.
- Short, sharp commentary. You explain *why* a decision was made, not just *what* you did.
- If the user's code has an obvious flaw, name it plainly and fix it.
- You favor clean, minimal, well-tested code over clever code.
- You never leave debris: no __pycache__, no stray temp files, no dead code.
- Before editing a file you haven't seen this session, read it first.
- When a task is ambiguous, ask one sharp clarifying question instead of guessing.

Agent Workflow: Understand → Plan → Inspect → Implement → Verify → Repair → Verify → Done.
- Search codebase and inspect symbols before reading large amounts of unrelated code.
- Identify relevant files and symbols before editing.
- Make minimal targeted changes; preserve unrelated code.
- Always verify syntax and test results after modifying code.

You have real tools:
- Inspection & Search: read_file, list_dir, search_code, find_symbol, find_definition, find_references, find_related_files, scan_project, git_status, git_diff, git_log
- Modification: write_file, edit_file
- Execution & Testing: run_command, run_tests
- Version Control: git_commit, git_push

Use tools instead of describing what you would do.
Risky actions (writes, shell commands, git commit/push) may prompt the user for confirmation before they execute — if one is declined, accept it and adjust course rather than repeating the same call.
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
        return f"{summary}\n\n{detail}"[:6000]
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
    "read_file": read_file,
    "list_dir": list_dir,
    "search_code": search_code,
    "find_symbol": _find_symbol_tool,
    "find_definition": _find_definition_tool,
    "find_references": _find_references_tool,
    "find_related_files": _find_related_files_tool,
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


def verify_and_repair_task(task: TaskItem, edited: List[str], messages: List[dict], max_attempts: int = 3) -> bool:
    """Performs verification (syntax checks, tests) and up to 3 repair attempts."""
    for attempt in range(1, max_attempts + 1):
        task.attempts = attempt
        syntax_errors = []
        for fpath in edited:
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
            turn_ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
            if not turn_ok:
                return False
            continue

        needs_test = any(m in ("run_tests", "test") for m in task.verification_methods) or HOLDER.auto_test
        if needs_test:
            UI.status(f"Running verification tests (attempt {attempt}/{max_attempts})...")
            test_res = run_tests(".")
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
                turn_ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
                if not turn_ok:
                    return False
                continue

        return True

    return False


def execute_structured_plan(plan: StructuredPlan, messages: List[dict]) -> bool:
    """Executes an approved plan step-by-step with dependency checking, verification, and repair."""
    if not plan or not plan.tasks:
        console.print("[yellow]No tasks found in plan to execute.[/yellow]")
        return False

    console.print(f"\n[bold cyan][•] Executing Plan:[/bold cyan] {plan.objective}\n")

    for i, task in enumerate(plan.tasks, 1):
        deps_ok = True
        for dep_id in task.dependencies:
            dep_task = plan.get_task(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                deps_ok = False
                break

        if not deps_ok:
            task.status = TaskStatus.BLOCKED
            UI.warn(f"Task {task.id} BLOCKED (prerequisite tasks incomplete)")
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

        edited = []
        turn_ok = run_turn(messages, TOOL_DEFINITIONS, edited=edited)
        if not turn_ok:
            task.status = TaskStatus.FAILED
            UI.error(f"Task {task.id} execution aborted.")
            return False

        task_ok = verify_and_repair_task(task, edited, messages, max_attempts=3)
        if not task_ok:
            task.status = TaskStatus.FAILED
            UI.error(f"Task {task.id} failed verification after 3 repair attempts. Halting execution.")
            render_plan_summary(plan)
            return False

        task.status = TaskStatus.COMPLETED
        UI.success(f"Task {task.id} verified: COMPLETED\n")

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
    if not HOLDER.auto_test or not edited:
        return
    UI.status("auto-test: running suite after edits...")
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
        render_stats(load_stats(), HOLDER.model_name, HOLDER.permission_mode, HOLDER.auto_test, HOLDER.verbose)
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
            PLAN_STATE.finish_drafting(structured_plan)
            render_structured_plan(structured_plan)
        else:
            PLAN_STATE.clear()
            UI.error("Planning cancelled or aborted.")
        flush_stats()
        return True

    if low == "/execute":
        if not PLAN_STATE.has_plan():
            console.print("[yellow]No structured plan to execute. Use /plan <task> first.[/yellow]")
            return True
        plan = PLAN_STATE.structured_plan
        execute_structured_plan(plan, messages)
        flush_stats()
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
    INDEX.ensure_indexed()
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
