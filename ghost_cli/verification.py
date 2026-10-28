"""Syntax validation, test runners, linting, and verification/repair loops for Ghost CLI."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .ui import console, UI
from .config import HOLDER
from .utils import add_xp, truncate_output


def validate_syntax(filepath: str, content: Optional[str] = None) -> Tuple[bool, str]:
    path = Path(filepath)
    if content is None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return False, f"Could not read file for validation: {e}"

    ext = path.suffix.lower()

    if ext == ".py":
        try:
            ast.parse(content, filename=filepath)
            compile(content, filepath, "exec")
            return True, ""
        except SyntaxError as e:
            line_txt = e.text.strip() if e.text else ""
            return False, f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  Line {e.lineno}: {line_txt}"
        except Exception as e:
            return False, f"Python compile error: {e}"

    if ext == ".json":
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSONDecodeError at line {e.lineno}, col {e.colno}: {e.msg}"

    return True, ""


def detect_python_test_cmd(root="."):
    root = Path(root)
    has_pytest = shutil.which("pytest") is not None
    markers = (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() or (root / "setup.cfg").exists()
    has_test_files = any(root.rglob("test_*.py")) or any(root.rglob("*_test.py"))
    if not markers and not has_test_files:
        return None
    if has_pytest:
        return ["pytest", "-q"]
    if has_test_files:
        return ["python3", "-m", "unittest", "discover", "-q"]
    return None


def detect_js_test_cmd(root="."):
    pkg = Path(root) / "package.json"
    if not pkg.exists():
        return None
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except Exception:
        return None
    scripts = data.get("scripts", {})
    test_script = scripts.get("test", "")
    if not test_script or "no test specified" in test_script.lower():
        return None
    if (Path(root) / "yarn.lock").exists() and shutil.which("yarn"):
        return ["yarn", "test"]
    if (Path(root) / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
        return ["pnpm", "test"]
    if shutil.which("npm"):
        return ["npm", "test", "--silent"]
    return None


def detect_test_commands(root="."):
    cmds = []
    py = detect_python_test_cmd(root)
    if py:
        cmds.append(("python", py))
    js = detect_js_test_cmd(root)
    if js:
        cmds.append(("javascript", js))
    return cmds


def _find_relevant_pytest_targets(edited_files: List[str], root: str = ".") -> List[str]:
    from .git_manager import IGNORE_DIRS
    targets = []
    root_path = Path(root)
    for fp in edited_files:
        if not fp or not fp.endswith(".py"):
            continue
        p = Path(fp)
        if p.name.startswith("test_") or p.name.endswith("_test.py"):
            if p.exists():
                targets.append(str(p))
            continue
        stem = p.stem
        candidates = list(root_path.rglob(f"test_{stem}.py")) + list(root_path.rglob(f"{stem}_test.py"))
        for c in candidates:
            if not any(part in IGNORE_DIRS for part in c.parts):
                targets.append(str(c))
    return sorted(set(targets))


def _run_one(lang, cmd, root, timeout):
    try:
        res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
        ok = res.returncode == 0
        output = (res.stdout + "\n" + res.stderr).strip()
        return {"lang": lang, "cmd": cmd, "success": ok, "output": output[-4000:]}
    except FileNotFoundError:
        return {"lang": lang, "cmd": cmd, "success": False, "output": f"{cmd[0]} not found on PATH."}
    except subprocess.TimeoutExpired:
        return {"lang": lang, "cmd": cmd, "success": False, "output": f"Test run timed out after {timeout}s."}
    except KeyboardInterrupt:
        return {"lang": lang, "cmd": cmd, "success": False, "output": "Test run interrupted by user."}
    except Exception as e:
        return {"lang": lang, "cmd": cmd, "success": False, "output": f"Error running tests: {e}"}


def run_tests(root=".", timeout=120, target_files: Optional[List[str]] = None):
    cmds = detect_test_commands(root)
    if not cmds:
        return {"ran": False, "success": None, "results": [], "output": "No Python or JavaScript test framework detected."}

    if target_files and shutil.which("pytest"):
        relevant = _find_relevant_pytest_targets(target_files, root)
        if relevant:
            cmds = [("python (scoped)", ["pytest", "-q"] + relevant)] + [c for c in cmds if c[0] != "python"]

    results = []
    if len(cmds) > 1:
        with ThreadPoolExecutor(max_workers=len(cmds)) as pool:
            futures = [pool.submit(_run_one, lang, cmd, root, timeout) for lang, cmd in cmds]
            results = [f.result() for f in futures]
    else:
        lang, cmd = cmds[0]
        results = [_run_one(lang, cmd, root, timeout)]

    overall_ok = all(r["success"] for r in results)
    return {"ran": True, "success": overall_ok, "results": results}


def run_lint_and_typecheck(edited_files: Optional[List[str]] = None) -> Tuple[bool, str]:
    edited_files = [f for f in (edited_files or []) if f and os.path.exists(f)]
    py_files = [f for f in edited_files if f.endswith(".py")]
    ts_files = [f for f in edited_files if f.endswith((".ts", ".tsx", ".js", ".jsx"))]

    checks = []
    if py_files:
        if shutil.which("ruff"):
            checks.append(("ruff", ["ruff", "check", "--quiet"] + py_files))
        if shutil.which("mypy"):
            checks.append(("mypy", ["mypy", "--ignore-missing-imports", "--no-error-summary"] + py_files))
    if ts_files or (Path("tsconfig.json").exists() and edited_files):
        if Path("tsconfig.json").exists() and shutil.which("tsc"):
            checks.append(("tsc", ["tsc", "--noEmit"]))

    if not checks:
        return True, "No linter/typechecker issues (or tools not installed)."

    failures = []
    for name, cmd in checks:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if res.returncode != 0:
                out = (res.stdout + "\n" + res.stderr).strip()
                failures.append(f"[{name}] failed:\n{out[:2000]}")
        except Exception:
            continue

    if failures:
        return False, "\n\n".join(failures)
    return True, f"Passed checks: {', '.join(n for n, _ in checks)}"


def format_results(result: dict) -> str:
    if not result["ran"]:
        return result["output"]
    lines = []
    for r in result["results"]:
        status = "PASS" if r["success"] else "FAIL"
        lines.append(f"[{r['lang']}] {status} — {' '.join(r['cmd'])}")
    return "\n".join(lines)


def _run_tests_tool() -> str:
    UI.status("Running tests...")
    result = run_tests(".")
    if not result["ran"]:
        UI.status(result["output"])
        return result["output"]
    summary = format_results(result)
    if result["success"]:
        UI.success("Tests passed")
    else:
        UI.error("Tests failed")
    if UI.verbose:
        UI.detail(summary)
    if not result["success"]:
        failing = [r for r in result["results"] if not r["success"]]
        detail = "\n\n".join(f"--- {r['lang']} output ---\n{r['output']}" for r in failing)
        return truncate_output(f"{summary}\n\n{detail}")
    return summary


def run_test_fix_loop(messages, initial_result=None, max_retries=3):
    from .agent import AGENT
    result = initial_result
    try:
        for attempt in range(1, max_retries + 1):
            if result is None:
                UI.status(f"Re-running tests (attempt {attempt}/{max_retries})...")
                result = run_tests(".")
            if not result["ran"]:
                console.print(f"[dim]{result['output']}[/dim]")
                return
            if result["success"]:
                UI.success(f"Tests passing (attempt {attempt}/{max_retries}).")
                add_xp(20, "green build")
                return
            UI.warn(f"Tests failing — attempt {attempt}/{max_retries}")
            console.print(format_results(result))
            if attempt == max_retries:
                UI.error("Max retries reached. Leaving for manual review.")
                return
            failing = [r for r in result["results"] if not r["success"]]
            failure_text = "\n\n".join(f"[{r['lang']}] {' '.join(r['cmd'])}\n{r['output']}" for r in failing)
            fix_prompt = (
                "The test suite is failing:\n\n" + failure_text[:4000] +
                "\n\nAnalyze the failure and fix the code using your tools. "
                "Make the fix directly; keep commentary short."
            )
            ok = AGENT.run_turn(fix_prompt)
            if not ok:
                UI.error("Fix attempt aborted.")
                return
            result = None
    except KeyboardInterrupt:
        console.print("\n[yellow]▲ Test-fix loop cancelled by user (Ctrl+C).[/yellow]")


def maybe_autotest(messages, edited):
    if not edited:
        return
    lint_ok, lint_msg = run_lint_and_typecheck(edited)
    if not lint_ok:
        UI.warn("Lint / typecheck warnings on modified files:")
        console.print(f"[dim yellow]{lint_msg[:1500]}[/dim yellow]")

    if not HOLDER.auto_test:
        return
    UI.status("auto-test: running suite after edits...")
    result = run_tests(".", target_files=edited)
    if not result["ran"]:
        console.print(f"[dim]{result['output']}[/dim]")
        return
    console.print(format_results(result))
    if result["success"]:
        add_xp(20, "clean test run")
        return
    try:
        from rich.prompt import Prompt
        ans = Prompt.ask("[yellow]Tests are failing. Attempt an automatic fix loop (up to 3 tries)? [y/N][/yellow]", default="n")
        if ans.strip().lower() in ("y", "yes"):
            run_test_fix_loop(messages, initial_result=result)
    except KeyboardInterrupt:
        console.print("\n[yellow]Auto-fix cancelled.[/yellow]")
