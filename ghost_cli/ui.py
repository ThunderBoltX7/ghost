"""Terminal UI management, Rich rendering, diff viewers, and error diagnostics for Ghost CLI."""

import re
import os
import sys
from contextlib import contextmanager
from typing import Any, Optional, Tuple, List

from rich import box
from rich.align import Align
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

DOT_SYM = "[•]"
CHECK_SYM = "[✓]"
CROSS_SYM = "[✗]"
WARN_SYM = "[!]"


def diagnose_error(error_message: str) -> Tuple[str, str, str]:
    """Diagnose an error message into (what_failed, why_failed, actionable_fix)."""
    msg = str(error_message or "").strip()
    msg_lower = msg.lower()

    # 1. API Authentication & Keys
    if any(k in msg_lower for k in ("api_key", "api key", "unauthorized", "401", "authentication", "invalid_api_key", "bearer")):
        return (
            "AI model API authentication failed.",
            "The configured API key is missing, empty, or rejected by the provider.",
            "Set your OPENAI_API_KEY environment variable, check ~/.ghost/config.json, or run /config. Run /doctor to test connectivity.",
        )

    # 2. Rate Limits & Quotas
    if any(k in msg_lower for k in ("rate limit", "429", "quota", "tokens per min", "insufficient_quota", "too many requests")):
        return (
            "API request rate limit or quota exceeded.",
            "The provider temporarily throttled requests or account credits have been exhausted.",
            "Wait a moment before retrying, switch to a fallback model via /model, or verify your billing dashboard.",
        )

    # 3. Network & Connection Timeouts
    if any(k in msg_lower for k in ("timeout", "connection error", "connect", "unreachable", "dns", "socket", "502", "503", "504", "529")):
        return (
            "Network connection to the AI provider endpoint failed or timed out.",
            "Internet connection dropped, a proxy or firewall is blocking traffic, or the remote provider is down.",
            "Verify your internet connection and proxy settings. Run /doctor to diagnose endpoint reachability.",
        )

    # 4. File Not Found
    if any(k in msg_lower for k in ("not found", "no such file", "filenotfounderror", "file read error")):
        return (
            "Target file or directory was not found on the filesystem.",
            "The specified path does not exist, has been deleted, or the filename was misspelled.",
            "Use list_dir or search_code to locate the correct file path before reading or modifying it.",
        )

    # 5. Permission & Sandbox Restrictions
    if any(k in msg_lower for k in ("permission denied", "permissionerror", "access is denied", "outside project root", "blocked by project root sandbox")):
        return (
            "File or directory access was denied by permissions or security sandbox.",
            "Attempted to access files outside the current project root, or filesystem permissions prohibit access.",
            "Approve the directory in the sandbox prompt if intended, or ensure terminal process has sufficient permissions.",
        )

    # 6. Syntax Validation Errors
    if any(k in msg_lower for k in ("syntaxerror", "syntax error", "invalid syntax", "syntax validation failed")):
        return (
            "Code syntax validation failed after file modifications.",
            "The edited code contains invalid syntax, missing closing parentheses/brackets, or indentation errors.",
            "Inspect the syntax error line and use edit_file to immediately correct the code structure.",
        )

    # 7. Git & Version Control Errors
    if any(k in msg_lower for k in ("merge conflict", "fatal: not a git", "git error", "uncommitted changes", "failed to push")):
        return (
            "Git version control operation encountered a failure.",
            "Working tree has uncommitted conflicts, merge markers, or is not an active git repository.",
            "Run /status or /diff to review working tree state, resolve merge conflicts, or run git init.",
        )

    # 8. General / Fallback
    summary = msg.splitlines()[0] if msg else "An unexpected error occurred."
    return (
        f"Operation failed: {summary}",
        "The operation encountered an unhandled condition or unexpected runtime error.",
        "Check your input arguments, inspect project configuration with /status, or run /doctor for system health.",
    )


def format_status_bar(
    model: str,
    mode: str,
    used_tokens: int,
    budget: int,
    pct: float,
    git_branch: str,
    level: int,
    title: str,
) -> str:
    """Format status bar string matching Phase 2.1 specification."""
    return (
        f"[Model: {model}] "
        f"[Mode: {mode}] "
        f"[Tokens: {used_tokens}/{budget} ({pct:.1f}%)] "
        f"[Branch: {git_branch}] "
        f"[XP: Lv.{level} {title}]"
    )


def render_diff_viewer(diff_text: str, filepath: str = "", console_instance: Optional[Console] = None):
    """Render syntax-highlighted diff: side-by-side when cols > 120, unified otherwise."""
    c = console_instance or console
    if not diff_text or not diff_text.strip():
        c.print("[dim](no textual difference)[/dim]")
        return

    term_width = c.width or 80
    lines = diff_text.splitlines()

    if term_width > 120:
        # Side-by-side diff table
        table = Table(
            title=f"Diff Preview: {filepath}" if filepath else "Diff Preview",
            box=box.ROUNDED,
            border_style="grey50",
            expand=True,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Old #", justify="right", style="dim", width=6)
        table.add_column("Original", style="white", ratio=1)
        table.add_column("New #", justify="right", style="dim", width=6)
        table.add_column("Modified", style="white", ratio=1)

        old_lineno = 0
        new_lineno = 0
        hunk_re = re.compile(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("---") or line.startswith("+++"):
                i += 1
                continue

            hunk_m = hunk_re.match(line)
            if hunk_m:
                old_lineno = int(hunk_m.group(1))
                new_lineno = int(hunk_m.group(2))
                table.add_row(
                    "",
                    f"[dim cyan]{escape(line)}[/dim cyan]",
                    "",
                    f"[dim cyan]{escape(line)}[/dim cyan]",
                )
                i += 1
                continue

            if line.startswith("-"):
                del_lines = []
                while i < len(lines) and lines[i].startswith("-") and not lines[i].startswith("---"):
                    del_lines.append(lines[i][1:])
                    i += 1
                add_lines = []
                while i < len(lines) and lines[i].startswith("+") and not lines[i].startswith("+++"):
                    add_lines.append(lines[i][1:])
                    i += 1

                max_len = max(len(del_lines), len(add_lines))
                for idx in range(max_len):
                    if idx < len(del_lines):
                        d_no = str(old_lineno)
                        d_text = f"[bold red]- {escape(del_lines[idx])}[/bold red]"
                        old_lineno += 1
                    else:
                        d_no = ""
                        d_text = ""

                    if idx < len(add_lines):
                        a_no = str(new_lineno)
                        a_text = f"[bold green]+ {escape(add_lines[idx])}[/bold green]"
                        new_lineno += 1
                    else:
                        a_no = ""
                        a_text = ""

                    table.add_row(d_no, d_text, a_no, a_text)
                continue

            elif line.startswith("+"):
                table.add_row(
                    "",
                    "",
                    str(new_lineno),
                    f"[bold green]+ {escape(line[1:])}[/bold green]",
                )
                new_lineno += 1
                i += 1
                continue

            else:
                text = line[1:] if line.startswith(" ") else line
                table.add_row(
                    str(old_lineno) if old_lineno else "",
                    f"[dim]  {escape(text)}[/dim]",
                    str(new_lineno) if new_lineno else "",
                    f"[dim]  {escape(text)}[/dim]",
                )
                if old_lineno:
                    old_lineno += 1
                if new_lineno:
                    new_lineno += 1
                i += 1

        c.print(table)
    else:
        # Unified diff panel
        rendered_lines = []
        for line in lines:
            if line.startswith("+++") or line.startswith("---"):
                rendered_lines.append(f"[bold white]{escape(line)}[/bold white]")
            elif line.startswith("@@"):
                rendered_lines.append(f"[bold cyan]{escape(line)}[/bold cyan]")
            elif line.startswith("+"):
                rendered_lines.append(f"[bold green]{escape(line)}[/bold green]")
            elif line.startswith("-"):
                rendered_lines.append(f"[bold red]{escape(line)}[/bold red]")
            else:
                text = line[1:] if line.startswith(" ") else line
                rendered_lines.append(f"[dim]  {escape(text)}[/dim]")

        body = "\n".join(rendered_lines)
        title = f"[bold cyan]Diff Preview: {filepath}[/bold cyan]" if filepath else "[bold cyan]Diff Preview[/bold cyan]"
        c.print(Panel(body, title=title, border_style="grey50"))


class TerminalUI:

    def __init__(self, console_instance: Console):
        self.console = console_instance
        self.verbose = False
        self._current_status = None

    def set_verbose(self, enabled: bool):
        self.verbose = bool(enabled)

    @contextmanager
    def spinner(self, message: str):
        """Rich Live spinner context manager during tool execution and thinking phases."""
        prev = self._current_status
        status = self.console.status(f"[cyan]{message}[/cyan]", spinner="dots")
        self._current_status = status
        status.start()
        try:
            yield status
        finally:
            try:
                status.stop()
            except Exception:
                pass
            self._current_status = prev

    def pause_spinner(self):
        """Temporarily pause live spinner for interactive user prompts."""
        if self._current_status:
            try:
                self._current_status.stop()
            except Exception:
                pass

    def resume_spinner(self):
        """Resume live spinner after interactive user prompts."""
        if self._current_status:
            try:
                self._current_status.start()
            except Exception:
                pass

    def status(self, message: str):
        self.console.print(f"[dim cyan]{DOT_SYM}[/dim cyan] {message}")

    def success(self, message: str):
        self.console.print(f"[bold green]{CHECK_SYM}[/bold green] {message}")

    def warn(self, message: str, why: Optional[str] = None, fix: Optional[str] = None):
        """Render actionable warning inside a yellow border Rich Panel."""
        body = f"[bold white]{message}[/bold white]"
        if why:
            body += f"\n[dim yellow]Cause:[/dim yellow] {why}"
        if fix:
            body += f"\n[dim cyan]Tip:[/dim cyan] {fix}"
        self.console.print(Panel(body, title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))

    def error(self, message: str, why: Optional[str] = None, fix: Optional[str] = None):
        """Render diagnostic error message inside a red border Rich Panel."""
        what, auto_why, auto_fix = diagnose_error(message)
        actual_why = why or auto_why
        actual_fix = fix or auto_fix

        body = (
            f"[bold white]What failed:[/bold white] {what}\n"
            f"[bold yellow]Why it likely failed:[/bold yellow] {actual_why}\n"
            f"[bold cyan]Actionable fix:[/bold cyan] {actual_fix}"
        )
        self.console.print(Panel(body, title="[bold red]Error[/bold red]", border_style="red"))

    def detail(self, renderable: Any):
        if self.verbose:
            self.console.print(renderable)

    def print_tool_card(self, fn_name: str, arg_summary: str, ok: bool, elapsed: float, detail: str = ""):
        """Print a compact one-line tool result card."""
        status_icon = "[bold green]✓[/bold green]" if ok else "[bold red]✗[/bold red]"
        elapsed_str = f"{elapsed:.1f}s"
        arg_part = f"[dim]{arg_summary[:60]}[/dim]" if arg_summary else ""
        detail_part = f"  [dim]{detail[:80]}[/dim]" if detail and self.verbose else ""
        self.console.print(
            f"  {status_icon} [cyan]{fn_name:<18}[/cyan] {arg_part} [dim grey50]{elapsed_str}[/dim grey50]{detail_part}"
        )

    def render_diff(self, diff_text: str, filepath: str = ""):
        render_diff_viewer(diff_text, filepath, console_instance=self.console)

    def display_mascot_banner(self):
        from .config import HOLDER, STATS_FILE
        from .utils import load_stats
        stats = load_stats(STATS_FILE)
        render_header(stats, HOLDER.model_name, HOLDER.fast_model_name)

    def display_status_bar(self):
        """Render dynamic status bar at bottom or between conversation turns."""
        try:
            from .config import HOLDER, STATS_FILE
            from .utils import load_stats, get_rank
            from .context import _total_tokens
            from .git_manager import get_git_branch
            from .agent import AGENT

            stats = load_stats(STATS_FILE)
            level = stats.get("level", 1)
            title = get_rank(level)
            used_tokens = _total_tokens(AGENT.messages)
            budget = HOLDER.cfg.get("context_limit", 64_000)
            pct = (used_tokens / budget * 100) if budget else 0.0
            branch = get_git_branch()

            bar_text = format_status_bar(
                model=HOLDER.model_name,
                mode=HOLDER.permission_mode,
                used_tokens=used_tokens,
                budget=budget,
                pct=pct,
                git_branch=branch,
                level=level,
                title=title,
            )
            self.console.print(Panel(bar_text, border_style="grey50", padding=(0, 1)))
        except Exception:
            pass


UI = TerminalUI(console)


GHOST_MASCOT = r"""[bold white]  .▄▄▄▄▄▄▄▄▄.
 ▐█  ◕   ◕  █▌
 ▐█     ▾    █▌
 ▐█▄▄▄▄▄▄▄▄▄█▌
  ╲╱ ╲╱ ╲╱ ╲╱[/bold white]"""


def render_header(stats, model_name, fast_model_name):
    console.clear()
    from .utils import get_rank
    rank = get_rank(stats["level"])
    threshold = stats["level"] * 100
    fast_info = f" (fast: {fast_model_name})" if fast_model_name != model_name else ""
    body = (
        f"{GHOST_MASCOT}\n\n"
        f"[bold white]G H O S T[/bold white]\n"
        f"[dim]{rank} · Level {stats['level']} · {stats['xp']}/{threshold} xp[/dim]\n"
        f"[dim]Model: {model_name}{fast_info} · type a task, or /help for commands[/dim]\n"
        f"[dim italic](Press Ctrl+C to halt AI responses cleanly)[/dim italic]"
    )
    console.print(Panel(Align.center(body), border_style="grey50", padding=(1, 4)))


def render_stats(stats, model_name, fast_model_name, permission_mode, auto_test, verbose):
    from .utils import get_rank
    table = Table(title="Ghost — Status", border_style="grey50", show_header=False)
    table.add_row("Rank", get_rank(stats["level"]))
    table.add_row("Level", str(stats["level"]))
    table.add_row("XP", f"{stats['xp']} / {stats['level']*100}")
    table.add_row("Edits made", str(stats.get("edits", 0)))
    table.add_row("Commands run", str(stats.get("commands", 0)))
    table.add_row("Active Model", str(model_name))
    table.add_row("Fast / Explorer Model", str(fast_model_name))
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
                    "[cyan]/mode[/cyan]               — show or set permission mode: [cyan]/mode plan|safe|normal|autonomous[/cyan]",
                    "[cyan]/verbose [on|off][/cyan]    — toggle detailed output vs quiet terminal UI",
                    "",
                    "[cyan]/init[/cyan]               — generate or refresh GHOST.md project memory file",
                    "[cyan]/plan <task>[/cyan]        — analyze, draft, and critique a structured execution plan",
                    "[cyan]/execute[/cyan]            — carry out approved plan with verification, lint & re-planning",
                    "[cyan]/cancelplan[/cyan]         — discard the current plan",
                    "[cyan]/todo[/cyan]               — display current live checklist",
                    "",
                    "[cyan]/compact[/cyan]            — summarize older turns into a working-state note",
                    "[cyan]/sessions [id][/cyan]      — list or resume a saved conversation session",
                    "[cyan]/rewind [n][/cyan]         — rewind conversation by n user turns (default 1)",
                    "",
                    "[cyan]/status[/cyan]              — show git status and active workspace state",
                    "[cyan]/diff[/cyan]                — show git diff for workspace or specific path",
                    "[cyan]/commit[/cyan]              — stage changes and commit with message",
                    "[cyan]/push[/cyan]                — push commits to remote repository",
                    "[cyan]/test[/cyan]                — detect and run the project's test suite",
                    "[cyan]/scan[/cyan]                — quick project scan (languages, markers, file count)",
                    "[cyan]/jobs[/cyan]                — list and monitor running background processes",
                    "[cyan]/tokens[/cyan]              — show conversation and cumulative token usage",
                    "[cyan]/undo[/cyan]                — revert last file edits or restore shadow git checkpoint",
                    "[cyan]/checkpoint[/cyan]          — create a workspace shadow checkpoint",
                    "[cyan]/checkpoints[/cyan]         — list all recorded shadow checkpoints",
                    "[cyan]/restore[/cyan]             — restore workspace to most recent checkpoint",
                    "[cyan]/clean[/cyan]               — clean up checkpoints, undo stack, and index caches",
                    "[cyan]/export[/cyan]              — export conversation transcript to Markdown",
                    "[cyan]/doctor[/cyan]              — run system diagnostics and environment health check",
                    "",
                    "[cyan]/clear[/cyan]               — clear the screen",
                    "[cyan]/help[/cyan]                — this menu",
                    "[cyan]/exit[/cyan]                — quit",
                ]
            ),
            title="Commands",
            border_style="grey50",
        )
    )


def _print_diff(diff_text: str, filepath: str = ""):
    render_diff_viewer(diff_text, filepath, console_instance=console)


def _print_scan(result, cached):
    table = Table(title=f"Project scan{' (cached)' if cached else ''}", border_style="grey50", show_header=False)
    table.add_row("Root", result["root"])
    table.add_row("Files", str(result["file_count"]))
    table.add_row("Languages", ", ".join(f"{k} ({v})" for k, v in result["languages"].items()) or "(none detected)")
    markers = [k for k, v in result["markers"].items() if v]
    table.add_row("Markers", ", ".join(markers) or "(none)")
    console.print(table)
