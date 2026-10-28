"""Context management, token counting, and intelligent compaction for Ghost CLI."""

from typing import List, Dict, Any
from .utils import _count_tokens
from .config import HOLDER
from .ui import UI

MAX_CONTEXT_MESSAGES = 60


class SessionTokenTracker:
    def __init__(self):
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0

    def record(self, input_tok: int, output_tok: int):
        self.input_tokens += max(0, int(input_tok))
        self.output_tokens += max(0, int(output_tok))
        self.total_tokens += max(0, int(input_tok)) + max(0, int(output_tok))

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0


TOKEN_TRACKER = SessionTokenTracker()


def _msg_tokens(msg: dict) -> int:
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
    if len(messages) < 4:
        return messages
    system = messages[0]
    # Keep the last N turns (at least the last 2 user-assistant exchanges) verbatim
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) >= 2:
        tail_idx = user_indices[-2]
    else:
        tail_len = min(6, len(messages) - 1)
        tail_idx = max(1, len(messages) - tail_len)

    # Move cut point backwards until tail begins on a user message
    # or an assistant message with no pending tool results
    while tail_idx > 1:
        msg = messages[tail_idx]
        role = msg.get("role")
        if role == "user":
            break
        if role == "assistant" and not msg.get("tool_calls"):
            break
        tail_idx -= 1

    keep_tail = messages[tail_idx:]
    to_summarize = messages[1:tail_idx]
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
        "- Goal\n- Decisions Made\n- Files Touched\n- Open Issues\n"
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
    return repair_messages([system, summary_msg] + keep_tail)


def _fallback_trim(messages: List[dict]) -> List[dict]:
    if len(messages) <= MAX_CONTEXT_MESSAGES + 1:
        return messages
    system = messages[0]
    tail = messages[1:][-MAX_CONTEXT_MESSAGES:]
    for i, m in enumerate(tail):
        if m.get("role") == "user":
            tail = tail[i:]
            break
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            tail = tail[i:]
            break
    return repair_messages([system] + tail)


def repair_messages(messages: List[dict]) -> List[dict]:
    """Ensure message sequences are valid for OpenAI API.

    Drops orphan tool messages without preceding assistant tool_calls,
    and inserts synthetic tool results for any unfulfilled tool_calls.
    """
    if not messages:
        return []

    repaired: List[dict] = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]
        role = msg.get("role")

        if role == "tool":
            # Isolated tool message without active preceding assistant tool call: drop it
            i += 1
            continue

        repaired.append(msg)
        i += 1

        if role == "assistant" and msg.get("tool_calls"):
            expected_ids = []
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    expected_ids.append(tc_id)

            seen_tool_msgs: Dict[str, dict] = {}
            while i < n and messages[i].get("role") == "tool":
                t_msg = messages[i]
                t_id = t_msg.get("tool_call_id")
                if t_id and t_id in expected_ids and t_id not in seen_tool_msgs:
                    seen_tool_msgs[t_id] = t_msg
                i += 1

            for tid in expected_ids:
                if tid in seen_tool_msgs:
                    repaired.append(seen_tool_msgs[tid])
                else:
                    repaired.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": "(interrupted: no result)",
                    })

    return repaired


def _microcompact_messages(messages: List[dict]) -> List[dict]:
    """Stage 1: Replace old tool results with stubs. No API call needed."""
    if len(messages) < 6:
        return messages

    # Find all tool message indices, keep the last 8 untouched
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    to_stub = tool_indices[:-8] if len(tool_indices) > 8 else []

    if not to_stub:
        return messages

    result = []
    for i, m in enumerate(messages):
        if i in to_stub:
            content = m.get("content", "")
            stub = content[:80].replace("\n", " ") if isinstance(content, str) else "(tool result)"
            result.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": f"[cleared: {stub}…]",
            })
        else:
            result.append(m)
    return result


def compact_messages(messages: List[dict]) -> List[dict]:
    context_limit = HOLDER.cfg.get("context_limit", 64_000)
    total = _total_tokens(messages)

    stage1_threshold = int(context_limit * 0.60)
    stage2_threshold = int(context_limit * 0.75)

    if total <= stage1_threshold:
        return messages

    if total <= stage2_threshold:
        UI.warn(f"Context at ~{total:,} tokens (60% threshold). Running microcompaction...")
        return _microcompact_messages(messages)

    UI.warn(f"Context at ~{total:,} tokens (75% threshold). Running full compaction...")
    microcompacted = _microcompact_messages(messages)
    return _compact_messages(microcompacted)


def compact_context_now(messages: List[dict]) -> List[dict]:
    before = _total_tokens(messages)
    compacted = _compact_messages(messages)
    after = _total_tokens(compacted)
    UI.console.print(f"[dim]Context compacted: ~{before:,} → ~{after:,} tokens.[/dim]")
    return compacted


def rewind_messages(messages: List[dict], n_turns: int = 1) -> List[dict]:
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
