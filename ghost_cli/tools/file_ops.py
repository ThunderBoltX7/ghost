"""File read, write, and edit operations for Ghost CLI."""

import os
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.syntax import Syntax

from ..ui import console, UI
from ..config import MAX_READ_BYTES
from ..utils import truncate_output, add_xp, _detect_line_ending, _normalize_ws
from ..permissions import check_path_sandbox
from ..git_manager import history_push
from ..verification import validate_syntax
from .search import INDEX, invalidate_cache


def read_file(filepath: str, offset: int = 1, limit: int = 2000) -> str:
    try:
        UI.status(f"Reading {filepath}...")
        path = Path(filepath)
        if not path.exists():
            UI.error(f"File '{filepath}' not found.")
            return f"File read error: '{filepath}' not found."

        raw_bytes = path.read_bytes()
        if len(raw_bytes) > MAX_READ_BYTES:
            UI.warn(f"{filepath} is {len(raw_bytes):,} bytes (large file, line-based read applied)")

        with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
            all_lines = f.read().splitlines()

        total_lines = len(all_lines)
        offset = max(1, int(offset))
        limit = max(1, int(limit))
        end = min(offset + limit - 1, total_lines)
        selected = all_lines[offset - 1:end]

        width = len(str(total_lines))
        numbered = "\n".join(f"{str(offset + i).rjust(width)}\t{line}" for i, line in enumerate(selected))

        hint = ""
        if end < total_lines:
            hint = f"\n[file has {total_lines} lines; showing {offset}–{end} ({total_lines - end} lines remaining). Use offset={end + 1} to read more]"
        elif offset > 1:
            hint = f"\n[file has {total_lines} lines; showing {offset}–{end}]"

        add_xp(5, "scanned file")
        UI.success(f"Read {filepath} (lines {offset}–{end} of {total_lines})")

        if UI.verbose:
            try:
                lexer = Syntax.guess_lexer(filepath, code=numbered)
                console.print(Syntax(numbered, lexer, theme="monokai", line_numbers=False, word_wrap=True))
            except Exception:
                console.print(numbered[:3000])

        return truncate_output(numbered + hint)
    except UnicodeDecodeError:
        UI.error(f"'{filepath}' does not look like a text file.")
        return f"File read error: '{filepath}' does not look like a text file."
    except Exception as e:
        UI.error(f"File read error: {e}")
        return f"File read error: {str(e)}"


def list_dir(path: str = ".") -> str:
    from ..git_manager import IGNORE_DIRS
    try:
        UI.status(f"Listing {path}...")
        entries = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
            depth = root.replace(path, "").count(os.sep)
            if depth > 2:
                continue
            for f in files:
                entries.append(os.path.relpath(os.path.join(root, f), path))
        entries_sorted = sorted(entries)[:200]
        listing = "\n".join(entries_sorted)
        add_xp(3, "surveyed the terrain")
        UI.success(f"Listed {len(entries_sorted)} items in {path}")
        if UI.verbose and listing:
            UI.detail(listing)
        return truncate_output(listing or "(empty)")
    except Exception as e:
        UI.error(f"List error: {e}")
        return f"List error: {str(e)}"


def unified_diff_text(old_content: str, new_content: str, filepath: str, max_lines: int = 200) -> str:
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=filepath,
        tofile=filepath,
        lineterm="",
    )
    lines = list(diff)
    if not lines:
        return ""
    truncated = lines[:max_lines]
    text = "".join(line if line.endswith("\n") else line + "\n" for line in truncated)
    if len(lines) > max_lines:
        text += f"... ({len(lines) - max_lines} more lines omitted)\n"
    return text


def write_file(filepath: str, content: str, confirm=None) -> str:
    try:
        if not check_path_sandbox(filepath):
            UI.warn(f"write_file blocked by project root sandbox: {filepath}")
            return f"Error: write_file to '{filepath}' outside project root was denied by user."

        UI.status(f"Writing {filepath}...")
        previous = ""
        existed = os.path.exists(filepath)
        orig_ending = "\n"
        if existed:
            try:
                raw = Path(filepath).read_bytes()
                orig_ending = _detect_line_ending(raw)
                with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
                    previous = f.read()
            except Exception:
                previous = ""

        diff_text = unified_diff_text(previous, content, filepath)
        if confirm is not None and not confirm(diff_text):
            UI.warn(f"write_file cancelled by user for {filepath}.")
            return "write_file cancelled by user."

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        write_content = content
        if existed and orig_ending == "\r\n":
            write_content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(write_content)

        history_push("write_file", filepath, previous if existed else "")
        INDEX.invalidate_file(filepath)
        invalidate_cache(".")

        syntax_ok, syntax_err = validate_syntax(filepath, content)

        if UI.verbose:
            from ..ui import _print_diff
            _print_diff(diff_text)

        action = "Created" if not existed else "Overwrote"
        UI.success(f"{action} {filepath} ({len(content)} bytes)")
        add_xp(20, "forged a new file" if not existed else "overwrote a file")

        result_msg = f"{action} {filepath} ({len(content)} bytes)."
        if not syntax_ok:
            UI.error(f"Syntax error in {filepath}: {syntax_err}")
            result_msg += f"\nERROR: Syntax validation failed after write:\n{syntax_err}\nPlease immediately inspect and fix this syntax error."
        return result_msg
    except Exception as e:
        UI.error(f"write_file failed on {filepath}: {e}")
        return str(e)


def _find_closest_region(content: str, old_text: str, n: int = 5) -> str:
    content_lines = content.splitlines()
    old_lines = old_text.splitlines()
    if not content_lines:
        return "File is empty."
    best_ratio = 0.0
    best_start = 0
    window = max(1, min(len(old_lines), len(content_lines)))
    for i in range(max(1, len(content_lines) - window + 1)):
        block = "\n".join(content_lines[i:i + window])
        ratio = difflib.SequenceMatcher(None, old_text, block).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
    snippet_lines = content_lines[max(0, best_start - 2):best_start + window + 2]
    start_num = max(1, best_start - 1)
    numbered = "\n".join(f"{start_num + i}\t{l}" for i, l in enumerate(snippet_lines))
    return f"Closest region (lines {start_num}–{start_num + len(snippet_lines) - 1}, similarity {best_ratio:.0%}):\n{numbered}"


def _apply_single_edit(content: str, old_text: str, new_text: str, replace_all: bool = False) -> Tuple[str, str]:
    if not old_text:
        return content, "old_text cannot be empty."

    work_content = content.replace("\r\n", "\n")
    work_old = old_text.replace("\r\n", "\n")
    work_new = new_text.replace("\r\n", "\n")

    count = work_content.count(work_old)

    if count == 1 or (count > 1 and replace_all):
        if replace_all:
            return work_content.replace(work_old, work_new), ""
        return work_content.replace(work_old, work_new, 1), ""

    if count > 1:
        return content, (
            f"old_text matched {count} locations. Add more surrounding context lines to make it unique, "
            f"or pass replace_all=true to replace all occurrences."
        )

    content_lines = work_content.splitlines()
    old_lines = work_old.splitlines()
    norm_content_lines = [l.rstrip().expandtabs(4) for l in content_lines]
    norm_old_lines = [l.rstrip().expandtabs(4) for l in old_lines]

    matches = []
    w = len(norm_old_lines)
    if w > 0 and w <= len(norm_content_lines):
        for i in range(len(norm_content_lines) - w + 1):
            if norm_content_lines[i:i + w] == norm_old_lines:
                matches.append(i)

    if len(matches) == 1 or (len(matches) > 1 and replace_all):
        result_lines = list(content_lines)
        new_lines = work_new.splitlines()
        for idx in reversed(matches):
            result_lines[idx:idx + w] = new_lines
        rebuilt = "\n".join(result_lines)
        if work_content.endswith("\n"):
            rebuilt += "\n"
        return rebuilt, ""

    if len(matches) > 1:
        return content, (
            f"Whitespace-normalized old_text matched {len(matches)} locations. "
            f"Include more context lines or set replace_all=true."
        )

    closest = _find_closest_region(work_content, work_old)
    return content, f"old_text not found (exact or whitespace-normalized). {closest}"


def edit_file(
    filepath: str,
    old_text: str = "",
    new_text: str = "",
    confirm=None,
    replace_all: bool = False,
    edits: Optional[List[Dict[str, str]]] = None,
) -> str:
    try:
        if not check_path_sandbox(filepath):
            UI.warn(f"edit_file blocked by project root sandbox: {filepath}")
            return f"Error: edit_file on '{filepath}' outside project root was denied by user."

        UI.status(f"Editing {filepath}...")
        if not os.path.exists(filepath):
            UI.error(f"File '{filepath}' not found.")
            return f"Error: '{filepath}' not found."

        raw = Path(filepath).read_bytes()
        orig_ending = _detect_line_ending(raw)
        with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
            content = f.read()

        if edits:
            edit_list = [(e.get("old_text", ""), e.get("new_text", ""), bool(e.get("replace_all", replace_all))) for e in edits]
        else:
            edit_list = [(old_text, new_text, replace_all)]

        working = content
        for i, (ot, nt, ra) in enumerate(edit_list):
            new_working, err = _apply_single_edit(working, ot, nt, ra)
            if err:
                prefix = f"Edit {i + 1}/{len(edit_list)}: " if len(edit_list) > 1 else ""
                UI.error(f"edit_file: {prefix}{err}")
                return f"Error: {prefix}{err}"
            working = new_working

        new_content = working
        diff_text = unified_diff_text(content.replace("\r\n", "\n"), new_content.replace("\r\n", "\n"), filepath)

        if confirm is not None and not confirm(diff_text):
            UI.warn(f"edit_file cancelled by user for {filepath}.")
            return "edit_file cancelled by user."

        history_push("edit_file", filepath, content)

        write_content = new_content.replace("\r\n", "\n")
        if orig_ending == "\r\n":
            write_content = write_content.replace("\n", "\r\n")
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            f.write(write_content)

        INDEX.invalidate_file(filepath)
        invalidate_cache(".")

        syntax_ok, syntax_err = validate_syntax(filepath, new_content)

        if UI.verbose:
            from ..ui import _print_diff
            _print_diff(diff_text)

        n_edits = len(edit_list)
        UI.success(f"Updated {filepath} ({n_edits} edit(s))")
        add_xp(25, "surgical edit")

        result_msg = f"Successfully updated {filepath} ({n_edits} edit(s))."
        if not syntax_ok:
            UI.error(f"Syntax error in {filepath}: {syntax_err}")
            result_msg += f"\nERROR: Syntax validation failed after edit:\n{syntax_err}\nPlease immediately inspect and fix this syntax error."
        return result_msg
    except Exception as e:
        UI.error(f"edit_file failed on {filepath}: {e}")
        return str(e)
