"""Read-only codebase research subagent for Ghost CLI."""

from typing import List, Dict, Any

from ..ui import UI
from ..config import HOLDER
from ..models import stream_completion
from ..utils import truncate_output

EXPLORER_READ_ONLY_TOOLS = {
    "read_file",
    "list_dir",
    "search_code",
    "find_symbol",
    "find_definition",
    "find_references",
    "find_related_files",
}


def spawn_explorer(question: str, max_steps: int = 8) -> str:
    UI.status(f"Subagent exploring: {question[:70]}...")
    sub_system = (
        "You are Ghost Explorer — a fast, read-only codebase research subagent.\n"
        "You run in an isolated environment with strictly read-only access.\n"
        "Use read_file, search_code, list_dir, find_symbol, find_definition, find_references, and find_related_files "
        "to answer the question with exact file paths, line numbers, and concise architectural findings.\n"
        "Do NOT propose or execute edits; return a dense, factual markdown summary of findings."
    )
    sub_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sub_system},
        {"role": "user", "content": question},
    ]
    from .registry import TOOL_DEFINITIONS, dispatch_tool_call
    sub_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in EXPLORER_READ_ONLY_TOOLS]

    try:
        steps = max(1, min(int(max_steps), 15))
    except (TypeError, ValueError):
        steps = 8

    try:
        for _ in range(steps):
            try:
                resp = stream_completion(
                    sub_messages,
                    sub_tools,
                    _model_override=HOLDER.fast_model_name,
                    quiet=not UI.verbose,
                )
            except KeyboardInterrupt:
                UI.warn("Explorer subagent cancelled by user.")
                return "Explorer subagent cancelled by user."
            except Exception as e:
                UI.warn(f"Explorer subagent model error: {e}")
                return f"Explorer subagent encountered an error: {e}"

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
                    fn_name = tc.function.name
                    if fn_name not in EXPLORER_READ_ONLY_TOOLS:
                        out = f"Error: '{fn_name}' is not permitted in read-only explorer subagent."
                    else:
                        out = dispatch_tool_call(tc, planning_restricted=True)
                    sub_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": truncate_output(str(out), max_chars=16000),
                    })
            else:
                summary = resp.content or "(no findings reported)"
                UI.success("Explorer subagent finished.")
                return summary

        UI.warn("Explorer subagent hit step limit; synthesizing summary from gathered findings.")
        for m in reversed(sub_messages):
            if m.get("role") == "assistant" and m.get("content"):
                return m["content"]
        return "Explorer subagent finished without a final summary."
    except KeyboardInterrupt:
        UI.warn("Explorer subagent cancelled by user.")
        return "Explorer subagent cancelled by user."
    except Exception as e:
        UI.warn(f"Explorer subagent error: {e}")
        return f"Explorer subagent error: {e}"
