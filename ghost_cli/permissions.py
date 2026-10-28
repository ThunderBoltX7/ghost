"""Safety, sandbox validation, and command permission classification for Ghost CLI."""

import re
import shlex
from pathlib import Path
from typing import Set, Tuple, Optional, List
from rich.prompt import Prompt
from .ui import console

RISK_LOW = "low"
RISK_RISKY = "risky"
RISK_CRITICAL = "critical"

PERMISSION_MODES = ("plan", "safe", "normal", "autonomous")
DEFAULT_PERMISSION_MODE = "normal"

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

CRITICAL_COMMAND_PATTERNS = [
    r"\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*f\b",
    r"\brm\s+.*-[a-zA-Z]*f[a-zA-Z]*r\b",
    r"\brm\s+.*--recursive",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"mkfs(\.\w+)?\s",
    r"dd\s+.*of=/dev/",
    r">\s*/dev/sd[a-z]",
    r">\s*/dev/nvme",
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
    r"curl\s+.*\|\s*(sudo\s+)?(sh|bash|zsh|python[23]?|perl|ruby|powershell|pwsh)",
    r"wget\s+.*-O\s*-\s*\|\s*(sudo\s+)?(sh|bash|zsh|python[23]?|perl|ruby|powershell|pwsh)",
    r"wget\s+.*\|\s*(sudo\s+)?(sh|bash|zsh|python[23]?|perl|ruby|powershell|pwsh)",
    r"powershell\s+.*-(?:enc|encodedcommand)\b",
    r"pwsh\s+.*-(?:enc|encodedcommand)\b",
    r"\bInvoke-Expression\b",
    r"\biex\b",
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
    r"pnpm\s+(add|remove|install)\b",
    r"git\s+(checkout|reset|rebase|merge|stash\s+drop)",
    r"\bkill\b",
    r"\bpkill\b",
    r"docker\s+(rm|rmi|system\s+prune)",
    r"sed\s+.*-i",
    r"tee\s+",
    r"python[23]?\s+-c\b",
]

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
    root = Path.cwd().resolve()
    try:
        resolved = Path(filepath).resolve()
        resolved.relative_to(root)
        return True, resolved
    except ValueError:
        return False, Path(filepath).resolve()


def check_path_sandbox(filepath: str) -> bool:
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


def _split_compound_command(command: str) -> List[str]:
    segments = []
    # Extract subshells $(...) and `...`
    subshells = re.findall(r"\$\((.*?)\)", command) + re.findall(r"`(.*?)`", command)
    for sub in subshells:
        if sub.strip():
            segments.extend(_split_compound_command(sub))

    # Split compound command on &&, ||, ;, |, and newlines respecting quotes
    try:
        lexer = shlex.shlex(command, posix=False, punctuation_chars="|;&\n")
        lexer.whitespace_split = True
        curr: List[str] = []
        for token in lexer:
            if token in ("&&", "||", ";", "|", "\n", "&"):
                if curr:
                    segments.append(" ".join(curr))
                    curr = []
            else:
                curr.append(token)
        if curr:
            segments.append(" ".join(curr))
    except Exception:
        parts = re.split(r"(?:&&|\|\||[;\n|])", command)
        segments.extend(p.strip() for p in parts if p.strip())

    if not segments:
        segments = [command.strip()]
    return segments


def _classify_single_segment(segment: str) -> str:
    # Check for direct critical patterns
    for pat in CRITICAL_COMMAND_PATTERNS:
        if re.search(pat, segment, re.IGNORECASE):
            return RISK_CRITICAL

    # Redirection check
    redir_match = re.search(r"(?:^|[^&|21])>{1,2}\s*(\S+)", segment)
    if redir_match:
        target = redir_match.group(1).strip()
        if re.search(r"^/dev/sd|^/dev/nvme", target, re.IGNORECASE):
            return RISK_CRITICAL
        if not re.search(r"^(?:/dev/null|nul|&1|&2)", target, re.IGNORECASE):
            return RISK_RISKY

    for pat in RISKY_COMMAND_PATTERNS:
        if re.search(pat, segment, re.IGNORECASE):
            if is_command_allowed_by_session(segment):
                return RISK_LOW
            return RISK_RISKY

    return RISK_LOW


def classify_command(command: str) -> str:
    cmd_strip = command.strip()
    if not cmd_strip:
        return RISK_LOW

    # Check whole command string for multi-token critical patterns (e.g. curl | sh)
    for pat in CRITICAL_COMMAND_PATTERNS:
        if re.search(pat, cmd_strip, re.IGNORECASE):
            return RISK_CRITICAL

    segments = _split_compound_command(cmd_strip)
    has_risky = False

    for seg in segments:
        risk = _classify_single_segment(seg)
        if risk == RISK_CRITICAL:
            return RISK_CRITICAL
        if risk == RISK_RISKY:
            has_risky = True

    if has_risky:
        return RISK_LOW if is_command_allowed_by_session(cmd_strip) else RISK_RISKY

    return RISK_LOW


def assess_tool_call(fn_name: str, args: dict) -> tuple:
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


def needs_confirmation(mode: str, risk: str, fn_name: Optional[str] = None) -> bool:
    mode = (mode or "normal").lower().strip()
    if mode == "ask":
        mode = "normal"
    elif mode == "strict":
        mode = "safe"

    if mode == "autonomous":
        return False

    if mode == "plan":
        if fn_name is not None and fn_name not in READ_ONLY_TOOLS:
            return True
        return risk != RISK_LOW

    if mode == "safe":
        if fn_name is not None and fn_name not in READ_ONLY_TOOLS:
            return True
        return risk != RISK_LOW

    # mode == "normal" (default): confirms only critical / destructive actions
    return risk == RISK_CRITICAL


def _risk_tag(risk: str) -> str:
    return {
        RISK_CRITICAL: "[bold red]CRITICAL[/bold red]",
        RISK_RISKY: "[yellow]RISKY[/yellow]",
        RISK_LOW: "[dim]routine[/dim]",
    }[risk]


def _ask(risk: str, command_for_prefix: Optional[str] = None) -> bool:
    try:
        try:
            from .ui import UI
            UI.pause_spinner()
        except Exception:
            pass
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
    finally:
        try:
            from .ui import UI
            UI.resume_spinner()
        except Exception:
            pass


def prompt_confirm(description: str, risk: str, command_for_prefix: Optional[str] = None) -> bool:
    console.print(f"\n[bold]⚠ confirmation required[/bold] ({_risk_tag(risk)}) — {description}")
    return _ask(risk, command_for_prefix=command_for_prefix)


def confirm_with_preview(description: str, risk: str, show_preview) -> bool:
    console.print(f"\n[bold]⚠ confirmation required[/bold] ({_risk_tag(risk)}) — {description}")
    if show_preview is not None:
        show_preview()
    return _ask(risk)
