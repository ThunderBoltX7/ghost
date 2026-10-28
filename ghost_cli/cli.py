"""Interactive REPL and slash command router for Ghost CLI."""

import sys
from typing import Dict, Any

from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel

from .ui import console, UI
from .config import HOLDER, CONFIG_FILE
from .sessions import SESSION_MGR, TODO_STATE
from .git_manager import (
    git_status,
    git_diff,
    git_commit,
    git_push,
    undo_last_edit,
    checkpoint_create,
    checkpoint_restore,
    checkpoint_list,
)
from .verification import _run_tests_tool
from .planning import generate_structured_plan, execute_structured_plan
from .tools.search import _scan_project_tool
from .tools.execution import BG_MANAGER
from .agent import AGENT

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.key_binding import KeyBindings
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    Completer = object
    Completion = None
    KeyBindings = None



def _cmd_help(arg: str):
    table = Table(title="Ghost AI CLI — Slash Commands", border_style="grey50")
    table.add_column("Command", style="cyan", width=22)
    table.add_column("Description", style="white")
    for name, info in COMMAND_REGISTRY.items():
        aliases_str = f", {', '.join(info['aliases'])}" if info.get("aliases") else ""
        table.add_row(f"{name}{aliases_str}", info["desc"])
    console.print(table)


def _cmd_exit(arg: str):
    console.print("[cyan]Goodbye! Keep building awesome things with Ghost.[/cyan]")
    sys.exit(0)


def _cmd_plan(arg: str):
    if not arg:
        arg = Prompt.ask("[bold cyan]What is your goal for this plan?[/bold cyan]").strip()
    if arg:
        plan = generate_structured_plan(arg)
        if plan:
            execute_structured_plan(plan, AGENT.messages)


def _cmd_sessions(arg: str):
    history = SESSION_MGR.load(arg if arg else None)
    if history:
        AGENT.messages = history
        UI.success(f"Loaded session with {len(history)} messages.")


def _cmd_status(arg: str):
    console.print(Panel(git_status(), title="[bold cyan]Git Status & Workspace[/bold cyan]", border_style="grey50"))


def _cmd_diff(arg: str):
    console.print(Panel(git_diff(arg if arg else None), title="[bold cyan]Git Diff[/bold cyan]", border_style="grey50"))


def _cmd_commit(arg: str):
    if not arg:
        arg = Prompt.ask("[bold cyan]Commit message?[/bold cyan]").strip()
    if arg:
        console.print(git_commit(arg))


def _cmd_push(arg: str):
    console.print(git_push())


def _cmd_undo(arg: str):
    ok, msg = undo_last_edit()
    if ok:
        UI.success(msg)
    else:
        UI.warn(msg)


def _cmd_checkpoint(arg: str):
    label = arg.strip() or "manual"
    ref = checkpoint_create(label)
    if ref:
        UI.success(f"Created shadow checkpoint ({label}) [{ref[:8]}].")
    else:
        UI.warn("Could not create checkpoint (not a git repository or no changes).")


def _cmd_checkpoints(arg: str):
    cps = checkpoint_list()
    if not cps:
        console.print("[yellow]No shadow checkpoints found.[/yellow]")
        return
    table = Table(title="Shadow Checkpoints", border_style="grey50")
    table.add_column("#", width=4)
    table.add_column("Ref", style="cyan")
    table.add_column("Label", style="white")
    table.add_column("Workspace", style="dim")
    for idx, cp in enumerate(reversed(cps[-15:]), 1):
        table.add_row(str(idx), cp.get("ref", "")[:8], cp.get("label", ""), cp.get("cwd", ""))
    console.print(table)


def _cmd_restore(arg: str):
    ok, msg = checkpoint_restore()
    if ok:
        UI.success(msg)
    else:
        UI.warn(msg)


def _cmd_rewind(arg: str):
    from .context import rewind_messages
    n = int(arg.strip()) if arg.strip().isdigit() else 1
    AGENT.messages = rewind_messages(AGENT.messages, n_turns=n)
    SESSION_MGR.save(AGENT.messages)


def _cmd_test(arg: str):
    console.print(Panel(_run_tests_tool(), title="[bold cyan]Test Suite Runner[/bold cyan]", border_style="grey50"))


def _cmd_scan(arg: str):
    console.print(Panel(_scan_project_tool(), title="[bold cyan]Project Scan[/bold cyan]", border_style="grey50"))


def _cmd_jobs(arg: str):
    jobs_str = BG_MANAGER.status(arg if arg else None)
    console.print(Panel(jobs_str, title="[bold cyan]Background Jobs[/bold cyan]", border_style="grey50"))


def _cmd_tokens(arg: str):
    from .context import _total_tokens, TOKEN_TRACKER
    total = _total_tokens(AGENT.messages)
    limit = HOLDER.cfg.get("context_limit", 64_000)
    thresh = int(limit * 0.70)
    pct = (total / limit) * 100 if limit else 0
    console.print(f"[bold cyan]Context Tokens (Active Window):[/bold cyan] ~{total:,} / {limit:,} ({pct:.1f}%) [dim](compaction threshold: {thresh:,})[/dim]")
    console.print(f"[bold cyan]Session Cumulative Usage:[/bold cyan] {TOKEN_TRACKER.total_tokens:,} tokens (Input: {TOKEN_TRACKER.input_tokens:,}, Output: {TOKEN_TRACKER.output_tokens:,})")


def _cmd_compact(arg: str):
    from .context import compact_context_now
    AGENT.messages = compact_context_now(AGENT.messages)
    SESSION_MGR.save(AGENT.messages)


def _cmd_clean(arg: str):
    from .git_manager import history_clear, CHECKPOINT_FILE
    from .tools.search import clear_cache
    history_clear()
    clear_cache(".")
    if CHECKPOINT_FILE.exists():
        try:
            CHECKPOINT_FILE.unlink()
        except Exception:
            pass
    UI.success("Cleaned checkpoints, undo history, and index caches.")


def _cmd_export(arg: str):
    out_path = arg.strip() or f"ghost_export_{SESSION_MGR.session_id}.md"
    try:
        lines = [f"# Ghost CLI Session Export - {SESSION_MGR.session_id}\n\n"]
        for msg in AGENT.messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            if role == "System":
                continue
            lines.append(f"### {role}\n\n{content}\n\n")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        UI.success(f"Session transcript exported to {out_path}")
    except Exception as e:
        UI.error(f"Failed to export transcript: {e}")


def _cmd_init(arg: str):
    from .tools.search import generate_ghost_md
    generate_ghost_md()


def _cmd_model(arg: str):
    if arg:
        HOLDER.switch_model(arg.strip())
    else:
        HOLDER.switch_model()


def _cmd_mode(arg: str):
    if arg:
        m = arg.strip().lower()
        if HOLDER.set_permission_mode(m):
            UI.success(f"Permission mode set to '{m}'.")
        else:
            UI.warn(f"Invalid mode '{m}'. Valid modes: plan, safe, normal, autonomous")
    else:
        console.print(f"Current permission mode: [bold cyan]{HOLDER.permission_mode}[/bold cyan]")
        console.print("[dim]Options: plan, safe, normal, autonomous[/dim]")


def _cmd_doctor(arg: str):
    UI.status("Running Ghost Doctor health check...")
    api_ok = False
    try:
        from .models import call_completion
        res = call_completion([{"role": "user", "content": "ping"}], _model_override=HOLDER.fast_model_name)
        api_ok = bool(res)
    except Exception:
        pass

    doc_table = Table(title="Ghost Doctor Diagnostics", border_style="grey50")
    doc_table.add_column("Component", style="cyan")
    doc_table.add_column("Status", style="white")
    doc_table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    doc_table.add_row("API Key Configured", "Yes" if HOLDER.cfg.get("api_key") else "Missing / Invalid")
    doc_table.add_row("API Connection", "Healthy" if api_ok else "Failed / Unreachable")
    doc_table.add_row("Active Model", HOLDER.model_name)
    doc_table.add_row("Fast Model", HOLDER.fast_model_name)
    doc_table.add_row("Permission Mode", HOLDER.permission_mode)
    doc_table.add_row("Config Path", str(CONFIG_FILE))
    console.print(doc_table)


def _cmd_clear(arg: str):
    AGENT.messages = [AGENT.messages[0]]
    TODO_STATE.items = []
    UI.success("Conversation history cleared.")


COMMAND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "/help": {"fn": _cmd_help, "desc": "Show this help menu with all available commands.", "aliases": ["/h", "/?"]},
    "/plan": {"fn": _cmd_plan, "desc": "Draft and execute a structured multi-step plan.", "aliases": []},
    "/sessions": {"fn": _cmd_sessions, "desc": "List and resume saved sessions.", "aliases": []},
    "/status": {"fn": _cmd_status, "desc": "Show git status and active workspace state.", "aliases": ["/git"]},
    "/diff": {"fn": _cmd_diff, "desc": "Show git diff for workspace or specific path.", "aliases": []},
    "/commit": {"fn": _cmd_commit, "desc": "Stage changes and commit with message.", "aliases": []},
    "/push": {"fn": _cmd_push, "desc": "Push commits to remote repository.", "aliases": []},
    "/undo": {"fn": _cmd_undo, "desc": "Undo last file edit or restore last checkpoint.", "aliases": []},
    "/checkpoint": {"fn": _cmd_checkpoint, "desc": "Create a workspace shadow checkpoint.", "aliases": []},
    "/checkpoints": {"fn": _cmd_checkpoints, "desc": "List all recorded shadow checkpoints.", "aliases": []},
    "/restore": {"fn": _cmd_restore, "desc": "Restore workspace to most recent checkpoint.", "aliases": []},
    "/rewind": {"fn": _cmd_rewind, "desc": "Rewind conversation history by n turns (default 1).", "aliases": []},
    "/test": {"fn": _cmd_test, "desc": "Run detected test suite (pytest/unittest/npm).", "aliases": []},
    "/scan": {"fn": _cmd_scan, "desc": "Scan project structure and codebase index.", "aliases": []},
    "/jobs": {"fn": _cmd_jobs, "desc": "List and monitor running background processes.", "aliases": ["/bg"]},
    "/tokens": {"fn": _cmd_tokens, "desc": "Show current conversation token usage and budget.", "aliases": []},
    "/compact": {"fn": _cmd_compact, "desc": "Manually compact conversation context into summary.", "aliases": []},
    "/clean": {"fn": _cmd_clean, "desc": "Clean up checkpoints, undo stack, and index caches.", "aliases": []},
    "/export": {"fn": _cmd_export, "desc": "Export conversation transcript to Markdown.", "aliases": []},
    "/init": {"fn": _cmd_init, "desc": "Generate or refresh GHOST.md project conventions.", "aliases": []},
    "/model": {"fn": _cmd_model, "desc": "Show or switch primary AI model.", "aliases": []},
    "/mode": {"fn": _cmd_mode, "desc": "Show or switch permission mode (plan, safe, normal, autonomous).", "aliases": []},
    "/doctor": {"fn": _cmd_doctor, "desc": "Run system diagnostics and environment health check.", "aliases": []},
    "/clear": {"fn": _cmd_clear, "desc": "Clear active conversation history.", "aliases": []},
    "/exit": {"fn": _cmd_exit, "desc": "Exit Ghost CLI.", "aliases": ["/quit", "/q"]},
}


class SlashCommandCompleter(Completer):
    """Autocomplete provider for registered slash commands with descriptions."""

    def get_completions(self, document, complete_event=None):
        text = getattr(document, "text_before_cursor", str(document))
        if not text.startswith("/"):
            return

        word = text.strip()
        for cmd in sorted(COMMAND_REGISTRY.keys()):
            info = COMMAND_REGISTRY[cmd]
            if cmd.startswith(word):
                if Completion is not None:
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display=cmd,
                        display_meta=info.get("desc", ""),
                    )
                else:
                    yield cmd

        for cmd, info in sorted(COMMAND_REGISTRY.items()):
            for alias in info.get("aliases", []):
                if alias.startswith(word):
                    if Completion is not None:
                        yield Completion(
                            alias,
                            start_position=-len(word),
                            display=alias,
                            display_meta=f"Alias for {cmd}: {info.get('desc', '')}",
                        )
                    else:
                        yield alias


def handle_slash_command(cmd: str) -> bool:
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in COMMAND_REGISTRY:
        COMMAND_REGISTRY[command]["fn"](arg)
        return True

    for _, info in COMMAND_REGISTRY.items():
        if command in info.get("aliases", []):
            info["fn"](arg)
            return True

    UI.warn(f"Unknown slash command: {command}. Type /help for available commands.")
    return True


def _build_toolbar():
    """Called by prompt_toolkit on every keypress to refresh the toolbar."""
    try:
        from .config import HOLDER, STATS_FILE
        from .utils import load_stats, get_rank
        from .context import _total_tokens, TOKEN_TRACKER
        from .git_manager import get_git_branch
        from .agent import AGENT

        stats = load_stats(STATS_FILE)
        level = stats.get("level", 1)
        rank = get_rank(level)
        used = _total_tokens(AGENT.messages)
        limit = HOLDER.cfg.get("context_limit", 64_000)
        pct = (used / limit * 100) if limit else 0.0

        if pct >= 90:
            ctx_color = "ansired"
        elif pct >= 70:
            ctx_color = "ansiyellow"
        else:
            ctx_color = "ansigreen"

        branch = get_git_branch()
        mode = HOLDER.permission_mode

        from prompt_toolkit.formatted_text import FormattedText
        return FormattedText([
            ("ansibrightblack", " 👻 ghost "),
            ("ansibrightcyan bold", f"{HOLDER.model_name} "),
            ("ansibrightblack", "│ "),
            ("ansibrightwhite", f"mode:{mode} "),
            ("ansibrightblack", "│ "),
            (ctx_color, f"ctx:{used:,}/{limit:,} ({pct:.0f}%) "),
            ("ansibrightblack", "│ "),
            ("ansicyan", f"branch:{branch} "),
            ("ansibrightblack", "│ "),
            ("ansiyellow", f"Lv.{level} {rank} "),
        ])
    except Exception:
        return " ghost "


MODES = ["plan", "normal", "autonomous"]


def interactive_repl() -> None:
    UI.display_mascot_banner()
    UI.display_status_bar()

    session = None
    if PROMPT_TOOLKIT_AVAILABLE:
        try:
            from pathlib import Path
            history_file = Path.home() / ".ghost" / "ghost_history.txt"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            style = Style.from_dict({
                "prompt": "bold #00d7ff",
                "completion-menu.completion": "bg:#262626 #ffffff",
                "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
                "completion-menu.meta.completion": "bg:#262626 #888888 italic",
                "completion-menu.meta.completion.current": "bg:#005f87 #e0e0e0 italic",
            })
            kb = KeyBindings()

            @kb.add("s-tab")
            def _cycle_mode(event):
                current = HOLDER.permission_mode
                try:
                    idx = MODES.index(current)
                except ValueError:
                    idx = 1
                new_mode = MODES[(idx + 1) % len(MODES)]
                HOLDER.set_permission_mode(new_mode)
                event.app.invalidate()   # forces toolbar redraw

            session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=SlashCommandCompleter(),
                style=style,
                key_bindings=kb,
                bottom_toolbar=_build_toolbar,
                refresh_interval=2.0,
            )
        except Exception:
            session = None

    while True:
        try:
            if session is not None:
                user_input = session.prompt([("class:prompt", "\nghost> ")]).strip()
            else:
                user_input = Prompt.ask("\n[bold cyan]ghost>[/bold cyan] ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                handled = handle_slash_command(user_input)
                if handled:
                    continue

            AGENT.run_turn(user_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Session interrupted. Type /exit to quit.[/cyan]")
        except Exception as e:
            UI.error(f"REPL Error: {e}")
