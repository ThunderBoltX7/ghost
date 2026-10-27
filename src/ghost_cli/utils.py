"""Shared utilities, token counting, stats, and leveling for Ghost CLI."""

import atexit
import json
import os
import time
from pathlib import Path

# Optional tiktoken for accurate token counting
try:
    import tiktoken as _tiktoken
    _TK_ENC = _tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        try:
            return len(_TK_ENC.encode(text))
        except Exception:
            return max(1, len(text) // 4)
except ImportError:
    _tiktoken = None
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def truncate_output(text: str, max_chars: int = 16000) -> str:
    """Keep head and tail with a truncation marker when output exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    removed_lines = text[half:-half].count("\n")
    removed_chars = len(text) - max_chars
    marker = f"\n... [truncated {removed_lines} lines / {removed_chars:,} chars] ...\n"
    return head + marker + tail


def _detect_line_ending(raw: bytes) -> str:
    """Return '\r\n' if CRLF dominates, else '\n'."""
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_ws(text: str) -> str:
    """Strip trailing whitespace per line and normalize CRLF→LF for fuzzy matching."""
    return "\n".join(line.rstrip().expandtabs(4) for line in text.replace("\r\n", "\n").splitlines())


# Stats / XP / leveling
_stats_cache = None
_stats_dirty = False


def _read_stats_from_disk(stats_file: Path):
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"level": 1, "xp": 0, "edits": 0, "commands": 0}


def load_stats(stats_file: Path):
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = _read_stats_from_disk(stats_file)
    return _stats_cache


def mark_dirty():
    global _stats_dirty
    _stats_dirty = True


def flush_stats(stats_file: Path = None):
    global _stats_dirty
    if stats_file is None:
        stats_file = Path.home() / ".ghost" / "ghost_stats.json"
    if _stats_dirty and _stats_cache is not None:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(_stats_cache, f)
        _stats_dirty = False


atexit.register(flush_stats)


def add_xp(amount, reason="", stats_file: Path = None, console=None, verbose=False):
    global _stats_dirty
    if stats_file is None:
        stats_file = Path.home() / ".ghost" / "ghost_stats.json"
    stats = load_stats(stats_file)
    stats["xp"] += amount
    threshold = stats["level"] * 100
    leveled = False
    while stats["xp"] >= threshold:
        stats["level"] += 1
        stats["xp"] -= threshold
        threshold = stats["level"] * 100
        leveled = True
    _stats_dirty = True
    if leveled:
        flush_stats(stats_file)
        try:
            from .ui import UI
            UI.success(f"▲ Level up! Ghost is now Level {stats['level']} ({get_rank(stats['level'])}).")
        except Exception:
            if console:
                console.print(f"[bold yellow]▲ Level up. Ghost is now Level {stats['level']}.[/bold yellow]")
    if verbose and reason and console:
        console.print(f"[dim green]+{amount} xp — {reason}[/dim green]")
    return stats


def bump(field, amount=1, stats_file: Path = None):
    global _stats_dirty
    if stats_file is None:
        stats_file = Path.home() / ".ghost" / "ghost_stats.json"
    stats = load_stats(stats_file)
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
