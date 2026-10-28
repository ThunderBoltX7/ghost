"""Session persistence and conversation management for Ghost CLI."""

import json
import os
import time
import uuid
from typing import List, Dict, Any, Optional

from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel

from .config import SESSIONS_DIR, ensure_data_dir, HOLDER
from .ui import console, UI


def _format_relative_time(ts: float) -> str:
    now = time.time()
    diff = int(now - ts)
    if diff < 60:
        return "just now"
    if diff < 3600:
        mins = diff // 60
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if diff < 86400:
        hours = diff // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = diff // 86400
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


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

    def update_status(self, item_id: str, new_status: str) -> None:
        """Update single task status dynamically and re-render."""
        status = str(new_status).lower()
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        found = False
        for item in self.items:
            if str(item.get("id")) == str(item_id):
                item["status"] = status
                found = True
                break
        if found:
            self.render()

    def format_text(self) -> str:
        if not self.items:
            return "Todo list is empty."
        lines = ["Current Todo Checklist:"]
        for item in self.items:
            sym = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(item["status"], "[ ]")
            lines.append(f"  {sym} #{item['id']}: {item['content']} ({item['status']})")
        return "\n".join(lines)

    def render(self):
        if not self.items:
            return
        from rich.markup import escape
        lines = []
        for item in self.items:
            st = item["status"]
            cid = item.get("id", "")
            content = escape(item.get("content", ""))
            prefix = f"#{cid} " if cid else ""
            if st == "completed":
                lines.append(f"[green][x][/green] [dim strike]{prefix}{content}[/dim strike]")
            elif st == "in_progress":
                lines.append(f"[bold cyan][>][/bold cyan] [bold white]{prefix}{content}[/bold white]")
            else:
                lines.append(f"[grey50][ ][/grey50] [white]{prefix}{content}[/white]")
        console.print(Panel("\n".join(lines), title="[bold cyan]Live Checklist[/bold cyan]", border_style="cyan", padding=(0, 1)))



TODO_STATE = TodoManager()


def todo_write(todos: List[Dict[str, Any]]) -> str:
    return TODO_STATE.update(todos)


def todo_read() -> str:
    return TODO_STATE.format_text()


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
        files = list(SESSIONS_DIR.glob("session_*.json"))
        curr_cwd = os.getcwd()
        parsed = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                user_msgs = [m.get("content", "") for m in data.get("messages", []) if m.get("role") == "user" and not str(m.get("content", "")).startswith("[")]
                preview = (user_msgs[0][:60] + "...") if user_msgs else "(empty session)"
                up_at = float(data.get("updated_at", f.stat().st_mtime))
                sess_cwd = data.get("cwd", "")
                is_curr_cwd = (os.path.abspath(sess_cwd) == os.path.abspath(curr_cwd)) if sess_cwd else False
                parsed.append({
                    "id": data.get("id", f.stem),
                    "file": f,
                    "updated_at": up_at,
                    "cwd": sess_cwd,
                    "is_current_cwd": is_curr_cwd,
                    "relative_time": _format_relative_time(up_at),
                    "preview": preview,
                    "msg_count": len(data.get("messages", [])),
                })
            except Exception:
                continue

        parsed.sort(key=lambda s: (not s["is_current_cwd"], -s["updated_at"]))
        return parsed[:limit]

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
            table.add_column("Updated", style="green")
            table.add_column("Current Dir?", style="yellow")
            table.add_column("Messages", width=10)
            table.add_column("First Prompt", style="white")
            for idx, s in enumerate(sessions, 1):
                match_str = "yes" if s.get("is_current_cwd") else "no"
                table.add_row(str(idx), s["id"], s.get("relative_time", ""), match_str, str(s["msg_count"]), s["preview"])
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
        from .context import repair_messages
        msgs = data.get("messages", [])
        return repair_messages(msgs)


SESSION_MGR = SessionManager()
