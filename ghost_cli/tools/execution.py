"""Background process management and live streaming command runner for Ghost CLI."""

import os
import signal
import subprocess
import sys
import threading
import time
from typing import List, Dict, Any, Optional

from ..ui import console, UI
from ..utils import truncate_output, add_xp

IS_WIN = sys.platform.startswith("win")
MAX_STREAM_BYTES = 1_000_000  # 1 MB per stream


def _get_process_group_kwargs() -> dict:
    if IS_WIN:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WIN:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        else:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class BackgroundProcessManager:
    def __init__(self):
        self.procs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._counter = 0

    def start(self, command: str) -> str:
        with self.lock:
            self._counter += 1
            job_id = f"bg_{self._counter}"

        buf: List[str] = []
        buf_lock = threading.Lock()
        total_bytes = 0

        kwargs = _get_process_group_kwargs()
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )

        def _reader():
            nonlocal total_bytes
            try:
                for line in iter(proc.stdout.readline, ""):
                    line_bytes = len(line.encode("utf-8", errors="replace"))
                    with buf_lock:
                        if total_bytes + line_bytes <= MAX_STREAM_BYTES:
                            buf.append(line)
                            total_bytes += line_bytes
                        elif total_bytes < MAX_STREAM_BYTES:
                            buf.append("\n... [output capped at 1MB] ...\n")
                            total_bytes = MAX_STREAM_BYTES
                        if len(buf) > 2000:
                            del buf[:500]
            except Exception:
                pass

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        with self.lock:
            self.procs[job_id] = {
                "id": job_id,
                "command": command,
                "proc": proc,
                "buffer": buf,
                "lock": buf_lock,
                "started_at": time.time(),
            }

        time.sleep(0.4)
        rc = proc.poll()
        with buf_lock:
            initial_logs = "".join(buf[-30:]).strip()
        status_str = "running" if rc is None else f"exited ({rc})"
        UI.success(f"Background process {job_id} ({status_str}) — PID {proc.pid}")
        return (
            f"Started background process '{job_id}' (PID {proc.pid}, status: {status_str}).\n"
            f"Initial output:\n{initial_logs or '(no output yet)'}"
        )

    def status(self, job_id: Optional[str] = None, tail_lines: int = 50) -> str:
        with self.lock:
            if not self.procs:
                return "No background processes."
            if job_id:
                info = self.procs.get(job_id)
                if not info:
                    return f"Background job '{job_id}' not found. Active IDs: {', '.join(self.procs.keys())}"
                targets = [info]
            else:
                targets = list(self.procs.values())

        out_blocks = []
        for info in targets:
            proc = info["proc"]
            rc = proc.poll()
            state = "running" if rc is None else f"exited ({rc})"
            with info["lock"]:
                logs = "".join(info["buffer"][-max(1, int(tail_lines)):]).strip()
            out_blocks.append(
                f"[{info['id']}] PID {proc.pid} | {state} | cmd: {info['command']}\n"
                f"--- last {tail_lines} lines ---\n{logs or '(empty)'}"
            )
        return truncate_output("\n\n".join(out_blocks))

    def kill(self, job_id: str) -> str:
        with self.lock:
            info = self.procs.get(job_id)
        if not info:
            return f"Background job '{job_id}' not found."
        proc = info["proc"]
        if proc.poll() is not None:
            return f"Job '{job_id}' already exited with code {proc.returncode}."
        try:
            _kill_process_tree(proc)
            UI.success(f"Terminated background job {job_id} (PID {proc.pid})")
            return f"Terminated background job '{job_id}' (PID {proc.pid})."
        except Exception as e:
            return f"Error killing job '{job_id}': {e}"


BG_MANAGER = BackgroundProcessManager()


def bg_status(job_id: Optional[str] = None, tail_lines: int = 50) -> str:
    return BG_MANAGER.status(job_id=job_id, tail_lines=tail_lines)


def bg_kill(job_id: str) -> str:
    return BG_MANAGER.kill(job_id)


def run_command(command: str, timeout: int = 60, background: bool = False) -> str:
    if background:
        UI.status(f"Starting background process: {command}")
        return BG_MANAGER.start(command)

    timeout = max(1, min(int(timeout), 600))
    UI.status(f"Running: {command}")
    kwargs = _get_process_group_kwargs()
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def _read_stream(stream, collector, is_err=False):
            total_bytes = 0
            try:
                for line in iter(stream.readline, ""):
                    line_bytes = len(line.encode("utf-8", errors="replace"))
                    if total_bytes + line_bytes <= MAX_STREAM_BYTES:
                        collector.append(line)
                        total_bytes += line_bytes
                    elif total_bytes < MAX_STREAM_BYTES:
                        collector.append("\n... [output stream capped at 1MB] ...\n")
                        total_bytes = MAX_STREAM_BYTES
                    if UI.verbose:
                        prefix = "[dim red]│ [/dim red]" if is_err else "[dim]│ [/dim]"
                        console.print(prefix + line.rstrip("\r\n"))
            except Exception:
                pass

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, False), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, True), daemon=True)
        t_out.start()
        t_err.start()

        start_time = time.time()
        while proc.poll() is None:
            if time.time() - start_time > timeout:
                _kill_process_tree(proc)
                UI.error(f"Command timed out after {timeout}s.")
                partial_out = "".join(stdout_lines).strip()
                partial_err = "".join(stderr_lines).strip()
                combined = "\n".join(filter(None, [partial_out, f"[stderr]\n{partial_err}" if partial_err else ""]))
                return truncate_output(f"Execution error: command timed out after {timeout}s.\n{combined}".strip())
            time.sleep(0.05)

        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

        out_str = "".join(stdout_lines).strip()
        err_str = "".join(stderr_lines).strip()

        parts = []
        if out_str:
            parts.append(out_str)
        if err_str:
            label = "stderr" if proc.returncode == 0 else "stderr (error)"
            parts.append(f"[{label}]\n{err_str}")
        output = "\n".join(parts).strip()

        if proc.returncode == 0:
            add_xp(15, "clean execution")
            UI.success("Command completed")
        else:
            add_xp(5, "hit a bug")
            UI.warn(f"Command exited with code {proc.returncode}")

        if not output:
            return "Command executed successfully." if proc.returncode == 0 else f"Command failed with exit code {proc.returncode}"
        result = output if proc.returncode == 0 else f"Exit code {proc.returncode}:\n{output}"
        return truncate_output(result)

    except KeyboardInterrupt:
        try:
            _kill_process_tree(proc)
        except Exception:
            pass
        UI.warn("Command interrupted by user.")
        return "Execution error: command interrupted by user (SIGINT)."
    except Exception as e:
        UI.error(f"Command execution error: {e}")
        return f"Execution error: {str(e)}"
