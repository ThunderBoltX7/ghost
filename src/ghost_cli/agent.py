"""Agent loop, conversation manager, and core execution engine for Ghost CLI."""

import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Any

from .ui import UI
from .config import HOLDER, GHOST_MD_FILE
from .models import stream_completion
from .utils import add_xp
from .context import compact_messages, repair_messages
from .sessions import SESSION_MGR
from .tools.registry import TOOL_DEFINITIONS, dispatch_tool_call, _parse_text_tool_calls
from .git_manager import get_git_branch


def get_git_context() -> Dict[str, str]:
    branch = get_git_branch()
    ctx = {"branch": branch, "dirty": "0", "recent_commit": "none"}
    if branch == "no-git":
        return ctx
    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if status_res.returncode == 0:
            dirty_count = len([l for l in status_res.stdout.splitlines() if l.strip()])
            ctx["dirty"] = str(dirty_count)
        log_res = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if log_res.returncode == 0:
            ctx["recent_commit"] = log_res.stdout.strip() or "none"
    except Exception:
        pass
    return ctx


def _mode_safety_rule(mode: str) -> str:
    rules = {
        "plan": "Planning mode active: read-only tools execute without prompt; modifying tools/commands are blocked or require confirmation.",
        "safe": "Safe mode active: confirmation is required for all mutating actions (edits, writes, git commits, shell commands).",
        "normal": "Normal mode active: standard operations execute automatically; destructive actions (rm -rf, git push --force, drop table) require confirmation.",
        "autonomous": "Autonomous mode active: automated operations execute without interactive prompts.",
    }
    return rules.get(mode, rules["normal"])


def build_system_prompt() -> str:
    git_ctx = get_git_context()
    os_info = f"{platform.system()} {platform.release()}"
    shell_name = os.environ.get("SHELL") or os.environ.get("COMSPEC") or ("cmd.exe" if sys.platform.startswith("win") else "/bin/sh")
    py_ver = platform.python_version()
    cwd = os.getcwd()
    perm_mode = HOLDER.permission_mode
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")

    tool_summary = (
        "Available Tools:\n"
        "- File Operations: read_file (chunked line reading), edit_file (exact string replacement), write_file (create/overwrite), list_dir\n"
        "- Code Navigation: search_code (ripgrep/python), find_symbol, find_definition, find_references, find_related_files, scan_project\n"
        "- Execution & Background: run_command (streaming shell runner), bg_status (poll bg jobs), bg_kill (terminate bg jobs)\n"
        "- Research Subagent: spawn_explorer (isolated read-only subagent for complex codebase research)\n"
        "- Git & Checkpoints: git_status, git_diff, git_log, git_commit, git_push\n"
        "- Verification & Checklist: run_tests (auto test runner), todo_write, todo_read"
    )

    ghost_memory = ""
    if GHOST_MD_FILE.exists():
        try:
            content = GHOST_MD_FILE.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                ghost_memory = f"\n\n--- Project Memory (GHOST.md) ---\n{content[:6000]}\n---"
        except Exception:
            pass
    if not ghost_memory:
        ghost_memory = "\n\nTip: run /init to generate project conventions in GHOST.md"

    sys_prompt = (
        "You are Ghost — an elite, highly reliable, autonomous AI coding assistant CLI.\n"
        "You are operating inside the user's workspace repository. You have direct access to tools for reading, "
        "searching, editing files, running shell commands, managing git checkpoints, and spawning read-only explorers.\n\n"
        f"Operating Environment:\n"
        f"- Working Directory: {cwd}\n"
        f"- OS: {os_info}\n"
        f"- Shell: {shell_name}\n"
        f"- Python: {py_ver}\n"
        f"- Local Time: {local_time}\n\n"
        f"Git Context:\n"
        f"- Branch: {git_ctx['branch']}\n"
        f"- Dirty Files: {git_ctx['dirty']}\n"
        f"- Recent Commit: {git_ctx['recent_commit']}\n\n"
        f"Safety & Permissions:\n"
        f"- Mode: {perm_mode}\n"
        f"- Rule: {_mode_safety_rule(perm_mode)}\n\n"
        f"{tool_summary}\n\n"
        "Guidelines:\n"
        "1. Always inspect the codebase before modifying it. Use search_code, find_symbol, or read_file first.\n"
        "2. Make precise, minimal, correct edits using edit_file. Verify syntax and tests after making changes.\n"
        "3. For complex multi-step goals, use todo_write to track progress on the live checklist.\n"
        "4. Be concise, direct, and professional in your responses. Avoid filler text."
        f"{ghost_memory}"
    )
    return sys_prompt


class AgentLoop:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(),
            }
        ]
        self.edited_files: List[str] = []

    def run_turn(self, user_prompt: str) -> bool:
        sys_content = build_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = sys_content
        else:
            self.messages.insert(0, {"role": "system", "content": sys_content})

        self.messages.append({"role": "user", "content": user_prompt})
        SESSION_MGR.save(self.messages)

        turn_clean = False
        try:
            # Main agentic loop
            for _ in range(1, 16):
                self.messages = compact_messages(self.messages)

                try:
                    resp = stream_completion(self.messages, TOOL_DEFINITIONS)
                except KeyboardInterrupt:
                    UI.warn("Turn interrupted by user.")
                    self.messages = repair_messages(self.messages)
                    SESSION_MGR.save(self.messages)
                    return False
                except Exception as e:
                    UI.error(f"LLM API error: {e}")
                    return False

                if resp is None:
                    UI.error("No response received from model.")
                    return False

                content = resp.content or ""
                tool_calls = resp.tool_calls or []

                # Fallback parsing if text contains pseudo-tool calls
                if not tool_calls and content:
                    cleaned_content, parsed_calls = _parse_text_tool_calls(content)
                    if parsed_calls:
                        content = cleaned_content or ""
                        tool_calls = parsed_calls

                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ]
                self.messages.append(assistant_msg)
                SESSION_MGR.save(self.messages)

                if not tool_calls:
                    if content:
                        add_xp(10, "successful explanation/answer")
                    turn_clean = True
                    break

                # Execute tool calls with interruption handling
                executed_call_ids = set()
                try:
                    for tc in tool_calls:
                        fn_name = tc.function.name
                        UI.status(f"Executing tool: [bold cyan]{fn_name}[/bold cyan]...")

                        tool_output = dispatch_tool_call(tc, edited=self.edited_files)

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_output),
                        })
                        executed_call_ids.add(tc.id)
                        SESSION_MGR.save(self.messages)
                except KeyboardInterrupt:
                    # Item 0.8: inject synthetic tool responses for unfulfilled tool calls
                    for tc in tool_calls:
                        if tc.id not in executed_call_ids:
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "(cancelled by user)",
                            })
                    SESSION_MGR.save(self.messages)
                    UI.warn("Tool execution cancelled by user.")
                    return False
            else:
                UI.warn("Reached maximum steps (15) for this prompt turn.")
                turn_clean = False

            if self.edited_files:
                try:
                    from .verification import maybe_autotest
                    maybe_autotest(self.messages, list(self.edited_files))
                except Exception as e:
                    if UI.verbose:
                        UI.warn(f"Autotest check encountered an issue: {e}")
                finally:
                    self.edited_files.clear()

            UI.display_status_bar()
            return turn_clean
        except KeyboardInterrupt:
            self.messages = repair_messages(self.messages)
            SESSION_MGR.save(self.messages)
            UI.warn("Turn interrupted by user.")
            return False


AGENT = AgentLoop()
