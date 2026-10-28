"""Agent loop, conversation manager, and core execution engine for Ghost CLI."""

import os
import platform
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

from .ui import UI
from .config import HOLDER, GHOST_MD_FILE
from .models import stream_completion, _tool_call_hash
from .utils import add_xp
from .context import compact_messages, repair_messages
from .sessions import SESSION_MGR
from .tools.registry import TOOL_DEFINITIONS, dispatch_tool_call, _parse_text_tool_calls, READ_ONLY_TOOLS
from .git_manager import get_git_branch, create_shadow_checkpoint


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

    def _run_tools_parallel(self, tool_calls, edited_files):
        """Run read-only tools concurrently. Returns list of (tc.id, output) in original order."""
        import time as _time
        import json as _json
        results = {}
        print_lock = threading.Lock()

        def _run_one(tc):
            fn_name = tc.function.name
            t0 = _time.monotonic()
            output = dispatch_tool_call(tc, edited=edited_files)
            elapsed = _time.monotonic() - t0
            ok = not str(output).startswith("Error")
            try:
                args_preview = _json.loads(tc.function.arguments or "{}")
                arg_summary = next(iter(args_preview.values()), "") if args_preview else ""
                arg_summary = str(arg_summary)[:60]
            except Exception:
                arg_summary = ""
            with print_lock:
                UI.print_tool_card(fn_name, arg_summary, ok, elapsed, detail=str(output)[:80])
            return tc.id, output

        with ThreadPoolExecutor(max_workers=min(6, len(tool_calls))) as pool:
            futures = {pool.submit(_run_one, tc): tc for tc in tool_calls}
            for future in as_completed(futures):
                tc_id, output = future.result()
                results[tc_id] = output

        return [(tc.id, results[tc.id]) for tc in tool_calls]

    def run_turn(self, user_prompt: str) -> bool:
        from .git_manager import create_shadow_checkpoint
        try:
            create_shadow_checkpoint(label=user_prompt[:60])
        except Exception:
            pass

        sys_content = build_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = sys_content
        else:
            self.messages.insert(0, {"role": "system", "content": sys_content})

        self.messages.append({"role": "user", "content": user_prompt})
        SESSION_MGR.save(self.messages)

        turn_clean = False
        try:
            call_counts: Counter = Counter()
            repeat_nudges: int = 0
            max_steps = int(HOLDER.cfg.get("max_steps", 40))
            # Main agentic loop
            for _ in range(1, max_steps + 1):
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
                all_readonly = all(
                    tc.function.name in READ_ONLY_TOOLS for tc in tool_calls
                )

                if all_readonly and len(tool_calls) > 1:
                    try:
                        id_output_pairs = self._run_tools_parallel(tool_calls, self.edited_files)
                        for tc_id, tool_output in id_output_pairs:
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": str(tool_output),
                            })
                            executed_call_ids.add(tc_id)
                    except KeyboardInterrupt:
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
                    try:
                        for tc in tool_calls:
                            fn_name = tc.function.name
                            h = _tool_call_hash(tc)
                            call_counts[h] += 1
                            if call_counts[h] >= 3:
                                # Don't execute; inject a nudge tool message instead
                                self.messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": (
                                        f"[loop detected] Tool '{fn_name}' with identical arguments has been called "
                                        f"{call_counts[h]} times in this turn. You are stuck in a loop. "
                                        "Stop repeating this call. Try a completely different approach, a different tool, "
                                        "or ask the user for clarification."
                                    ),
                                })
                                repeat_nudges += 1
                                if repeat_nudges >= 2:
                                    UI.warn("Loop detected: Ghost repeated the same tool call too many times. Stopping turn.")
                                    SESSION_MGR.save(self.messages)
                                    return False
                                continue  # skip dispatch, go to next tc

                            UI.status(f"Executing tool: [bold cyan]{fn_name}[/bold cyan]...")

                            import time as _time

                            t0 = _time.monotonic()
                            tool_output = dispatch_tool_call(tc, edited=self.edited_files)
                            elapsed = _time.monotonic() - t0

                            ok = not str(tool_output).startswith("Error")
                            # Build a short arg summary
                            try:
                                import json as _json
                                args_preview = _json.loads(tc.function.arguments or "{}")
                                arg_summary = next(iter(args_preview.values()), "") if args_preview else ""
                                arg_summary = str(arg_summary)[:60]
                            except Exception:
                                arg_summary = ""

                            UI.print_tool_card(fn_name, arg_summary, ok, elapsed, detail=str(tool_output)[:80])

                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(tool_output),
                            })
                            executed_call_ids.add(tc.id)
                            SESSION_MGR.save(self.messages)
                    except KeyboardInterrupt:
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
                UI.warn(f"Reached maximum steps ({max_steps}) for this prompt turn.")
                turn_clean = False

            if self.edited_files:
                try:
                    from .verification import maybe_autotest
                    maybe_autotest(self.messages, list(self.edited_files))
                except Exception as e:
                    if UI.verbose:
                        UI.warn(f"Autotest check encountered an issue: {e}")

                from .context import _total_tokens
                from .git_manager import git_diff
                files_str = ", ".join(self.edited_files[:5])
                extra = f" (+{len(self.edited_files)-5} more)" if len(self.edited_files) > 5 else ""
                tok = _total_tokens(self.messages)
                limit = HOLDER.cfg.get("context_limit", 64_000)
                pct = (tok / limit * 100) if limit else 0
                UI.console.print(
                    f"\n[dim grey50]─── turn summary ───[/dim grey50]\n"
                    f"[dim]  Files changed : {files_str}{extra}[/dim]\n"
                    f"[dim]  Context usage : ~{tok:,} / {limit:,} ({pct:.0f}%)[/dim]\n"
                    f"[dim]  Type /undo to revert edits · /diff to review changes[/dim]\n"
                    f"[dim grey50]────────────────────[/dim grey50]"
                )
                self.edited_files.clear()

            UI.display_status_bar()
            return turn_clean
        except KeyboardInterrupt:
            self.messages = repair_messages(self.messages)
            SESSION_MGR.save(self.messages)
            UI.warn("Turn interrupted by user.")
            return False


AGENT = AgentLoop()
