"""Tool definitions, routing, and dispatch mechanism for Ghost CLI."""

import json
import re
import uuid
from types import SimpleNamespace
from typing import List, Tuple, Optional

from rich.syntax import Syntax

from ..ui import console, UI
from ..config import HOLDER
from ..utils import truncate_output, add_xp, bump
from ..permissions import assess_tool_call, needs_confirmation, prompt_confirm, confirm_with_preview
from ..git_manager import git_status, git_diff, git_log, git_commit, git_push
from ..sessions import todo_write, todo_read
from ..verification import _run_tests_tool
from .file_ops import read_file, list_dir, write_file, edit_file
from .search import INDEX, _scan_project_tool
from .execution import run_command, bg_status, bg_kill
from .explorer import spawn_explorer

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


def search_code(pattern: str, path: str = ".") -> str:
    from .search import _python_search_code
    import shutil, subprocess
    try:
        UI.status(f"Searching code for '{pattern}'...")
        if shutil.which("rg"):
            cmd = ["rg", "-n", "--no-heading", "--color=never", "-g", "!.git", "-g", "!__pycache__", "-g", "!node_modules", "--", pattern, path]
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


def _build_confirm(risk, description, filepath=""):
    def _confirm(diff_text=""):
        if not needs_confirmation(HOLDER.permission_mode, risk):
            return True
        preview = None
        if diff_text:
            preview = lambda: UI.render_diff(diff_text, filepath=filepath)
        return confirm_with_preview(description, risk, preview)
    return _confirm


def _get_required_params(fn_name: str) -> List[str]:
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
        fp = str(args.get("filepath", ""))
        args["confirm"] = _build_confirm(risk, description, filepath=fp)
    elif needs_confirmation(HOLDER.permission_mode, risk):
        cmd_for_prefix = str(args.get("command", "")) if fn_name == "run_command" else None
        if not prompt_confirm(description, risk, command_for_prefix=cmd_for_prefix):
            return f"{fn_name} cancelled by user."

    fn = TOOL_MAP.get(fn_name)
    if fn is None:
        return f"Unknown tool: {fn_name}"

    try:
        with UI.spinner(f"Executing {fn_name}..."):
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

    output_str = str(result) if result is not None and str(result).strip() != "" else f"Tool '{fn_name}' completed with no output."
    return truncate_output(output_str, max_chars=16000)


def _parse_text_tool_calls(text: str) -> Tuple[Optional[str], Optional[List[SimpleNamespace]]]:
    if not text:
        return text, None

    calls: List[SimpleNamespace] = []
    cleaned = text

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
