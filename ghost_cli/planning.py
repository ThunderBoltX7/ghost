"""Structured planning, task items, plan parsing, critique, and step execution engine for Ghost CLI."""

import json
import os
import re
from typing import List, Dict, Any, Optional

from rich.panel import Panel
from rich.table import Table

from .ui import console, UI
from .config import HOLDER
from .git_manager import create_shadow_checkpoint, git_diff, _is_git_repo
from .verification import validate_syntax, run_lint_and_typecheck, run_tests, format_results
from .models import stream_completion

PLANNING_SYSTEM_SUFFIX = """

PLANNING MODE is active. You may only use read-only inspection tools:
read_file, list_dir, search_code, find_symbol, find_definition, find_references, find_related_files, git_status, git_diff, git_log, scan_project.
Do NOT call write_file, edit_file, run_command, git_commit, or git_push right now.

Investigate the codebase as needed, then respond with a structured execution plan enclosed in a ```json ``` block matching:
{
  "objective": "Concise summary of what will be achieved",
  "affected_files": ["path/to/file1.py", "path/to/file2.py"],
  "risks": "Potential edge cases or risks",
  "tasks": [
    {
      "id": 1,
      "description": "Concise action description",
      "affected_files": ["path/to/file1.py"],
      "dependencies": [],
      "expected_changes": "What will be created or modified",
      "verification_methods": ["syntax_check", "run_tests"],
      "risks": "Risks for this step"
    }
  ]
}

End your response once the JSON plan is formulated. Do not perform the work yet.
"""


class TaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class TaskItem:
    def __init__(
        self,
        task_id: Any,
        description: str,
        affected_files: Optional[List[str]] = None,
        dependencies: Optional[List[Any]] = None,
        expected_changes: str = "",
        verification_methods: Optional[List[str]] = None,
        risks: str = "",
    ):
        self.id = task_id
        self.description = description
        self.affected_files = affected_files or []
        self.dependencies = dependencies or []
        self.expected_changes = expected_changes
        self.verification_methods = verification_methods or ["syntax_check"]
        self.risks = risks
        self.status = TaskStatus.PENDING
        self.result = ""
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "affected_files": self.affected_files,
            "dependencies": self.dependencies,
            "expected_changes": self.expected_changes,
            "verification_methods": self.verification_methods,
            "risks": self.risks,
            "status": self.status,
            "attempts": self.attempts,
        }


class StructuredPlan:
    def __init__(
        self,
        objective: str = "",
        affected_files: Optional[List[str]] = None,
        tasks: Optional[List[TaskItem]] = None,
        risks: str = "",
    ):
        self.objective = objective
        self.affected_files = affected_files or []
        self.tasks = tasks or []
        self.risks = risks
        self.raw_text = ""
        self.critique = ""

    def get_task(self, task_id: Any) -> Optional[TaskItem]:
        for t in self.tasks:
            if str(t.id) == str(task_id):
                return t
        return None

    def is_complete(self) -> bool:
        return len(self.tasks) > 0 and all(t.status == TaskStatus.COMPLETED for t in self.tasks)


class PlanState:
    def __init__(self):
        self.active = False
        self.task_description = None
        self.structured_plan: Optional[StructuredPlan] = None

    def start(self, task: str):
        self.active = True
        self.task_description = task
        self.structured_plan = None

    def finish_drafting(self, plan: StructuredPlan):
        self.active = False
        self.structured_plan = plan

    def clear(self):
        self.active = False
        self.task_description = None
        self.structured_plan = None

    def has_plan(self) -> bool:
        return self.structured_plan is not None and len(self.structured_plan.tasks) > 0


PLAN_STATE = PlanState()


def parse_plan_response(text: str, default_task: str = "") -> StructuredPlan:
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    data = None
    if json_match:
        try:
            data = json.loads(json_match.group(1))
        except Exception:
            data = None

    if not data:
        raw_match = re.search(r"(\{[\s\S]*\"tasks\"[\s\S]*\})", text)
        if raw_match:
            try:
                data = json.loads(raw_match.group(1))
            except Exception:
                data = None

    if isinstance(data, dict) and "tasks" in data:
        objective = str(data.get("objective", default_task or "Execution plan"))
        affected_files = list(data.get("affected_files", []))
        risks = str(data.get("risks", ""))
        tasks = []
        for idx, t in enumerate(data.get("tasks", []), 1):
            if isinstance(t, dict):
                tid = t.get("id", idx)
                desc = t.get("description", f"Step {idx}")
                afiles = t.get("affected_files", [])
                deps = t.get("dependencies", [])
                changes = t.get("expected_changes", "")
                verif = t.get("verification_methods", ["syntax_check"])
                risk = t.get("risks", "")
                tasks.append(
                    TaskItem(
                        task_id=tid,
                        description=desc,
                        affected_files=afiles,
                        dependencies=deps,
                        expected_changes=changes,
                        verification_methods=verif,
                        risks=risk,
                    )
                )
            elif isinstance(t, str):
                tasks.append(TaskItem(task_id=idx, description=t))

        plan = StructuredPlan(objective=objective, affected_files=affected_files, tasks=tasks, risks=risks)
        plan.raw_text = text
        return plan

    tasks = []
    objective = default_task or "Execution plan"
    affected_files = []
    risks = ""

    lines = text.splitlines()
    step_pattern = re.compile(r"^\s*(?:###?\s*)?(?:Step\s*)?(\d+)[\.\:\)]\s*(.+)", re.IGNORECASE)
    file_pattern = re.compile(r"([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)")

    for line in lines:
        line_clean = line.strip()
        if line_clean.lower().startswith(("objective:", "**objective:**", "# objective")):
            objective = re.sub(r"^(?:#+\s*)?(?:\*\*)?objective:(?:\*\*)?\s*", "", line_clean, flags=re.IGNORECASE).strip()
            continue
        if line_clean.lower().startswith(("risks:", "**risks:**", "# risks")):
            risks = re.sub(r"^(?:#+\s*)?(?:\*\*)?risks:(?:\*\*)?\s*", "", line_clean, flags=re.IGNORECASE).strip()
            continue
        m = step_pattern.match(line_clean)
        if m:
            tid = int(m.group(1))
            desc = m.group(2).strip()
            found_files = file_pattern.findall(desc)
            affected_files.extend(found_files)
            tasks.append(
                TaskItem(
                    task_id=tid,
                    description=desc,
                    affected_files=found_files,
                    dependencies=[tid - 1] if tid > 1 else [],
                    expected_changes="",
                    verification_methods=["syntax_check"],
                    risks="",
                )
            )

    if not tasks:
        tasks.append(
            TaskItem(
                task_id=1,
                description=default_task or "Perform planned updates",
                affected_files=file_pattern.findall(text),
                dependencies=[],
                expected_changes="",
                verification_methods=["syntax_check"],
                risks="",
            )
        )

    plan = StructuredPlan(
        objective=objective,
        affected_files=sorted(list(set(affected_files))),
        tasks=tasks,
        risks=risks,
    )
    plan.raw_text = text
    return plan


def render_structured_plan(plan: StructuredPlan):
    body_lines = [
        f"[bold white]Objective:[/bold white] {plan.objective}",
    ]
    if plan.affected_files:
        body_lines.append(f"[bold white]Affected Files:[/bold white] {', '.join(plan.affected_files)}")
    if plan.risks:
        body_lines.append(f"[bold yellow]Risks:[/bold yellow] {plan.risks}")
    if plan.critique:
        body_lines.append(f"[bold cyan]Critique Notes:[/bold cyan] {plan.critique}")

    console.print(Panel("\n".join(body_lines), title="Structured Plan", border_style="cyan"))

    table = Table(title="Execution Tasks", border_style="grey50", header_style="bold cyan")
    table.add_column("#", style="bold white", width=4)
    table.add_column("Task Description", style="white")
    table.add_column("Dependencies", style="dim", width=14)
    table.add_column("Verification", style="dim", width=16)
    table.add_column("Status", width=12)

    status_styles = {
        TaskStatus.PENDING: "[dim]PENDING[/dim]",
        TaskStatus.RUNNING: "[bold cyan]RUNNING[/bold cyan]",
        TaskStatus.COMPLETED: "[bold green]COMPLETED[/bold green]",
        TaskStatus.FAILED: "[bold red]FAILED[/bold red]",
        TaskStatus.BLOCKED: "[yellow]BLOCKED[/yellow]",
        TaskStatus.SKIPPED: "[dim]SKIPPED[/dim]",
    }

    for t in plan.tasks:
        deps_str = ", ".join(str(d) for d in t.dependencies) if t.dependencies else "None"
        verif_str = ", ".join(t.verification_methods) if t.verification_methods else "syntax_check"
        status_str = status_styles.get(t.status, t.status)
        table.add_row(str(t.id), t.description, deps_str, verif_str, status_str)

    console.print(table)
    console.print("[dim]Type /execute to carry out this plan step-by-step, or /cancelplan to discard.[/dim]")


def render_plan_summary(plan: StructuredPlan):
    table = Table(title=f"Plan Summary: {plan.objective}", border_style="grey50")
    table.add_column("#", width=4)
    table.add_column("Task", style="white")
    table.add_column("Status", width=12)
    table.add_column("Attempts", width=10)

    status_styles = {
        TaskStatus.PENDING: "[dim]PENDING[/dim]",
        TaskStatus.RUNNING: "[bold cyan]RUNNING[/bold cyan]",
        TaskStatus.COMPLETED: "[bold green]COMPLETED[/bold green]",
        TaskStatus.FAILED: "[bold red]FAILED[/bold red]",
        TaskStatus.BLOCKED: "[yellow]BLOCKED[/yellow]",
        TaskStatus.SKIPPED: "[dim]SKIPPED[/dim]",
    }
    for t in plan.tasks:
        table.add_row(str(t.id), t.description, status_styles.get(t.status, t.status), str(t.attempts or 1))
    console.print(table)


def generate_structured_plan(task_goal: str) -> Optional[StructuredPlan]:
    UI.status(f"Drafting plan for: {task_goal}...")
    plan_system = (
        "You are Ghost Planner — an elite software architect.\n"
        "Draft a robust, structured execution plan for the user's goal.\n"
        + PLANNING_SYSTEM_SUFFIX
    )
    plan_messages = [
        {"role": "system", "content": plan_system},
        {"role": "user", "content": f"Goal: {task_goal}"},
    ]
    from .tools.registry import PLANNING_TOOL_DEFINITIONS
    for _ in range(5):
        try:
            resp = stream_completion(
                plan_messages,
                PLANNING_TOOL_DEFINITIONS,
                _model_override=HOLDER.fast_model_name,
                quiet=True,  # Suppress raw JSON streaming output during plan drafting
            )
            if resp and resp.tool_calls:
                plan_messages.append({
                    "role": "assistant",
                    "content": resp.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in resp.tool_calls
                    ],
                })
                from .tools.registry import dispatch_tool_call
                for tc in resp.tool_calls:
                    out = dispatch_tool_call(tc, planning_restricted=True)
                    plan_messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)})
                continue

            if resp and resp.content:
                plan = parse_plan_response(resp.content, default_task=task_goal)
                if plan.tasks:
                    plan = critique_plan(plan, plan_messages)
                    PLAN_STATE.finish_drafting(plan)
                    UI.success("Plan formulated successfully.")
                    render_structured_plan(plan)
                    return plan
        except Exception as e:
            UI.error(f"Planning error: {e}")
            break
    UI.error("Failed to formulate a valid structured plan.")
    return None


def critique_plan(plan: StructuredPlan, messages: List[dict]) -> StructuredPlan:
    UI.status("Critiquing plan against codebase...")
    critique_prompt = (
        "Review the drafted execution plan below for missing edge cases, wrong file paths, "
        "missing dependencies, or verification gaps. Return a refined JSON plan block in the exact same schema, "
        "plus a one-sentence 'critique' field summarizing any improvements made.\n\n"
        f"Draft Plan:\n{plan.raw_text}"
    )
    crit_messages = list(messages) + [{"role": "user", "content": critique_prompt}]
    from .tools.registry import PLANNING_TOOL_DEFINITIONS
    try:
        resp = stream_completion(
            crit_messages,
            PLANNING_TOOL_DEFINITIONS,
            _model_override=HOLDER.fast_model_name,
            quiet=True,
        )
        if resp and resp.content:
            refined = parse_plan_response(resp.content, default_task=plan.objective)
            if refined.tasks:
                json_match = re.search(r"\"critique\"\s*:\s*\"([^\"]+)\"", resp.content)
                refined.critique = json_match.group(1) if json_match else "Verified & refined by critique pass."
                UI.success("Plan critique complete.")
                return refined
    except Exception as e:
        UI.warn(f"Critique pass skipped ({e}).")
    return plan


def review_git_diff_pass(messages: List[dict], edited: List[str]) -> bool:
    if not edited or not _is_git_repo():
        return True
    diff = git_diff()
    if not diff or diff == "No changes.":
        return True
    UI.status("Running final reviewer pass over git diff...")
    review_prompt = (
        "Review the following `git diff` of your recent changes. Check for unintended edits, "
        "leftover debug statements, broken imports, or obvious bugs.\n"
        "If everything looks clean, reply starting with 'LGTM:'. "
        "If you spot a bug, fix it immediately using edit_file.\n\n"
        f"```diff\n{diff[:5000]}\n```"
    )
    messages.append({"role": "user", "content": review_prompt})
    from .agent import AgentLoop
    agent_inst = AgentLoop()
    agent_inst.messages = messages
    agent_inst.run_turn(review_prompt)
    return True


def verify_and_repair_task(task: TaskItem, edited: List[str], messages: List[dict], max_attempts: int = 3) -> bool:
    for attempt in range(1, max_attempts + 1):
        task.attempts = attempt
        syntax_errors = []
        for fpath in set(edited):
            if fpath and os.path.exists(fpath):
                ok, err = validate_syntax(fpath)
                if not ok:
                    syntax_errors.append(err)

        if syntax_errors:
            error_report = "Syntax verification failed:\n" + "\n".join(syntax_errors)
            UI.warn(f"Syntax error detected (attempt {attempt}/{max_attempts}). Initiating repair...")
            if attempt == max_attempts:
                console.print(f"[red]{error_report}[/red]")
                return False

            repair_prompt = f"{error_report}\n\nAnalyze and fix the syntax error immediately using edit_file/write_file."
            messages.append({"role": "user", "content": repair_prompt})
            from .agent import AgentLoop
            agent_inst = AgentLoop()
            agent_inst.messages = messages
            agent_inst.run_turn(repair_prompt)
            continue

        lint_ok, lint_report = run_lint_and_typecheck(edited)
        if not lint_ok:
            UI.warn(f"Lint/typecheck issue detected (attempt {attempt}/{max_attempts}). Repairing...")
            if attempt < max_attempts:
                messages.append({
                    "role": "user",
                    "content": f"Lint / typecheck failed on modified files:\n{lint_report}\n\nFix these issues using edit_file.",
                })
                from .agent import AgentLoop
                agent_inst = AgentLoop()
                agent_inst.messages = messages
                agent_inst.run_turn(messages[-1]["content"])
                continue

        needs_test = any(m in ("run_tests", "test") for m in task.verification_methods) or HOLDER.auto_test
        if needs_test:
            UI.status(f"Running verification tests (attempt {attempt}/{max_attempts})...")
            test_res = run_tests(".", target_files=edited)
            if test_res["ran"] and not test_res["success"]:
                UI.warn(f"Test verification failed (attempt {attempt}/{max_attempts}). Initiating repair...")
                if attempt == max_attempts:
                    console.print(format_results(test_res))
                    return False

                failing = [r for r in test_res["results"] if not r["success"]]
                fail_detail = "\n\n".join(f"[{r['lang']}] {' '.join(r['cmd'])}\n{r['output']}" for r in failing)
                repair_prompt = (
                    f"Tests failed during task verification:\n{fail_detail[:3000]}\n\n"
                    f"Diagnose the failure and repair the code using your tools. Keep commentary concise."
                )
                messages.append({"role": "user", "content": repair_prompt})
                from .agent import AgentLoop
                agent_inst = AgentLoop()
                agent_inst.messages = messages
                agent_inst.run_turn(repair_prompt)
                continue

        return True

    return False


def execute_structured_plan(plan: StructuredPlan, messages: List[dict]) -> bool:
    if not plan or not plan.tasks:
        console.print("[yellow]No tasks found in plan to execute.[/yellow]")
        return False

    create_shadow_checkpoint(f"plan: {plan.objective}")
    console.print(f"\n[bold cyan][•] Executing Plan:[/bold cyan] {plan.objective}\n")

    from .sessions import TODO_STATE
    todo_items = [
        {
            "id": str(t.id),
            "content": t.description,
            "status": "completed" if t.status == TaskStatus.COMPLETED else ("in_progress" if t.status == TaskStatus.RUNNING else "pending"),
        }
        for t in plan.tasks
    ]
    TODO_STATE.update(todo_items)

    all_edited: List[str] = []
    idx = 0
    replan_count = 0

    while idx < len(plan.tasks):
        task = plan.tasks[idx]
        deps_ok = True
        for dep_id in task.dependencies:
            dep_task = plan.get_task(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                deps_ok = False
                break

        if not deps_ok:
            task.status = TaskStatus.BLOCKED
            UI.warn(f"Task {task.id} BLOCKED (prerequisite tasks incomplete)")
            idx += 1
            continue

        task.status = TaskStatus.RUNNING
        TODO_STATE.update_status(str(task.id), "in_progress")
        console.print(f"[bold white][•] Task {task.id}/{len(plan.tasks)}:[/bold white] {task.description}")

        task_prompt = (
            f"Step {task.id} of {len(plan.tasks)}: {task.description}\n"
            f"Target file(s): {', '.join(task.affected_files) if task.affected_files else 'As needed'}\n"
            f"Expected changes: {task.expected_changes or 'Complete the task as specified'}\n"
            f"Verification: {', '.join(task.verification_methods)}\n\n"
            f"Inspect before editing, make minimal targeted changes using write_file/edit_file, and verify your changes. "
            f"Keep commentary concise."
        )
        messages.append({"role": "user", "content": task_prompt})

        edited: List[str] = []
        from .agent import AgentLoop
        agent_inst = AgentLoop()
        agent_inst.messages = messages
        agent_inst.run_turn(task_prompt)
        edited.extend(agent_inst.edited_files)
        all_edited.extend(edited)

        task_ok = verify_and_repair_task(task, edited, messages, max_attempts=3)
        if not task_ok:
            task.status = TaskStatus.FAILED
            if replan_count < 1:
                replan_count += 1
                UI.warn(f"Task {task.id} failed verification. Attempting automatic re-plan of remaining work...")
                replan_prompt = (
                    f"Task {task.id} ('{task.description}') failed verification. "
                    "Re-evaluate the current codebase state and output an updated JSON plan for the remaining tasks."
                )
                messages.append({"role": "user", "content": replan_prompt})
                replan_agent = AgentLoop()
                replan_agent.messages = messages
                replan_agent.run_turn(replan_prompt)
                new_plan = parse_plan_response(messages[-1].get("content") or "", default_task=plan.objective)
                if new_plan.tasks:
                    plan.tasks = plan.tasks[:idx] + new_plan.tasks
                    UI.success(f"Re-planned {len(new_plan.tasks)} remaining task(s). Continuing execution...")
                    continue
            UI.error(f"Task {task.id} failed verification. Halting execution.")
            render_plan_summary(plan)
            return False

        task.status = TaskStatus.COMPLETED
        TODO_STATE.update_status(str(task.id), "completed")
        UI.success(f"Task {task.id} verified: COMPLETED\n")
        idx += 1

    review_git_diff_pass(messages, all_edited)
    UI.success(f"Plan Execution Complete ({len(plan.tasks)}/{len(plan.tasks)} tasks successful).\n")
    render_plan_summary(plan)
    PLAN_STATE.clear()
    return True
