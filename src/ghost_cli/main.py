import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.align import Align
from rich.rule import Rule

# AI client
from openai import OpenAI

console = Console()

# --- Configuration & Auto-Setup ----------------------------------------
DATA_DIR = Path.home() / ".ghost"
CONFIG_FILE = DATA_DIR / "ghost_config.json"
STATS_FILE = DATA_DIR / "ghost_stats.json"
UNDO_FILE = DATA_DIR / ".ghost_undo.json"

MODEL_PRESETS = {
    "1": ("deepseek/deepseek-v3.2", "DeepSeek V3.2 — Fast, precise coding"),
    "2": ("openai/gpt-4o", "GPT-4o — Deep multi-step reasoning"),
    "3": ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet — Elite code architecture"),
    "4": ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B — Fast open-weights model"),
}

DEFAULT_MODEL = "deepseek/deepseek-v3.2"


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def interactive_setup(cfg):
    """Handles onboarding, provider selection, and permanent saving."""
    console.clear()
    console.print(Panel(
        "[bold cyan]Ghost Configuration[/bold cyan]\n\n"
        "Configure your AI provider. This is saved permanently to ~/.ghost/ghost_config.json.",
        border_style="cyan"
    ))

    console.print("Select your API Provider:")
    console.print("[cyan]1.[/cyan] OpenRouter (Recommended)")
    console.print("[cyan]2.[/cyan] OpenAI")
    console.print("[cyan]3.[/cyan] Groq")
    console.print("[cyan]4.[/cyan] DeepSeek")
    console.print("[cyan]5.[/cyan] Custom (Any OpenAI-compatible endpoint)")

    choice = Prompt.ask("\n[bold white]Enter choice (1-5)[/bold white]", choices=["1", "2", "3", "4", "5"], default="1")

    if choice == "5":
        provider_name = "Custom"
        base_url = Prompt.ask("[bold white]Enter Base URL (e.g., http://localhost:11434/v1)[/bold white]")
        default_model = Prompt.ask("[bold white]Enter default model name[/bold white]", default="gpt-4")
    else:
        providers = {
            "1": ("OpenRouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v3.2"),
            "2": ("OpenAI", "https://api.openai.com/v1", "gpt-4o"),
            "3": ("Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            "4": ("DeepSeek", "https://api.deepseek.com", "deepseek-chat"),
        }
        provider_name, base_url, default_model = providers[choice]

    while True:
        new_key = Prompt.ask(f"\n[bold white]Enter your {provider_name} API Key[/bold white]", password=True).strip()

        if len(new_key) > 5:
            cfg["provider"] = provider_name
            cfg["base_url"] = base_url
            cfg["api_key"] = new_key
            cfg["model"] = default_model

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(cfg, f, indent=4)

            console.print(f"\n[bold green]✔ {provider_name} configured and saved securely to {CONFIG_FILE}[/bold green]")
            console.print("[dim]Booting system...[/dim]\n")
            return cfg
        else:
            console.print("[red]Invalid key length. Try again.[/red]")


def get_api_config(cfg):
    env_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if env_key and "api_key" not in cfg:
        cfg["api_key"] = env_key
        cfg["base_url"] = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        cfg["model"] = os.environ.get("GHOST_MODEL", DEFAULT_MODEL)
        cfg["provider"] = "Environment Variable"
        return cfg

    if not cfg.get("api_key"):
        cfg = interactive_setup(cfg)

    return cfg


# Initialize configuration
DATA_DIR.mkdir(parents=True, exist_ok=True)
_cfg = get_api_config(load_config())

API_KEY = _cfg.get("api_key")
BASE_URL = _cfg.get("base_url")
MODEL_NAME = _cfg.get("model", DEFAULT_MODEL)

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    default_headers={"X-Title": "Ghost"},
)


def switch_model(target_model: str = None):
    """Dynamically switch models in real-time during an active session."""
    global MODEL_NAME, _cfg

    if not target_model:
        console.print("\n[bold cyan]Available Models:[/bold cyan]")
        for key, (m_id, desc) in MODEL_PRESETS.items():
            active_marker = " [bold green](active)[/bold green]" if m_id == MODEL_NAME else ""
            console.print(f"[cyan]{key}.[/cyan] {m_id} — [dim]{desc}[/dim]{active_marker}")
        console.print("[cyan]5.[/cyan] Custom model identifier")

        choice = Prompt.ask(
            "\n[bold white]Choose model (1-5 or enter full model ID)[/bold white]",
            default="1",
        ).strip()

        if choice in MODEL_PRESETS:
            target_model = MODEL_PRESETS[choice][0]
        elif choice == "5":
            target_model = Prompt.ask("[bold white]Enter model ID (e.g. anthropic/claude-3.5-sonnet)[/bold white]").strip()
        else:
            target_model = choice

    MODEL_NAME = target_model
    _cfg["model"] = MODEL_NAME

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(_cfg, f, indent=4)
    except Exception:
        pass

    console.print(f"\n[bold green]✔ Model updated to:[/bold green] [bold cyan]{MODEL_NAME}[/bold cyan]\n")


# --- Progression state ---------------------------------------------------
_stats_cache = None
_stats_dirty = False


def _read_stats_from_disk():
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"level": 1, "xp": 0, "edits": 0, "commands": 0}


def load_stats():
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = _read_stats_from_disk()
    return _stats_cache


def flush_stats():
    global _stats_dirty
    if _stats_dirty and _stats_cache is not None:
        with open(STATS_FILE, "w") as f:
            json.dump(_stats_cache, f)
        _stats_dirty = False


def add_xp(amount, reason=""):
    global _stats_dirty
    stats = load_stats()
    stats["xp"] += amount
    threshold = stats["level"] * 100
    leveled = False
    if stats["xp"] >= threshold:
        stats["level"] += 1
        stats["xp"] -= threshold
        leveled = True
    _stats_dirty = True
    if leveled:
        console.print(f"[bold yellow]▲ Level up. Ghost is now Level {stats['level']}.[/bold yellow]")
    console.print(f"[dim green]+{amount} xp — {reason}[/dim green]")
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


# --- Local tools -----------------------------------------------------------
def run_command(command: str) -> str:
    console.print(f"[bold magenta]▶ exec[/bold magenta] [dim]{command}[/dim]")
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = res.stdout if res.returncode == 0 else res.stderr
        if res.returncode == 0:
            add_xp(15, "clean execution")
        else:
            add_xp(5, "hit a bug")
        return output.strip() if output else "Command executed successfully."
    except Exception as e:
        return f"Execution error: {str(e)}"


def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        add_xp(5, "scanned file")
        try:
            lexer = Syntax.guess_lexer(filepath, code=content)
            console.print(Syntax(content, lexer, theme="monokai", line_numbers=True, word_wrap=True))
        except Exception:
            pass
        return content
    except Exception as e:
        return f"File read error: {str(e)}"


def list_dir(path: str = ".") -> str:
    try:
        entries = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".venv")]
            depth = root.replace(path, "").count(os.sep)
            if depth > 2:
                continue
            for f in files:
                entries.append(os.path.relpath(os.path.join(root, f), path))
        listing = "\n".join(sorted(entries)[:200])
        add_xp(3, "surveyed the terrain")
        return listing or "(empty)"
    except Exception as e:
        return f"List error: {str(e)}"


def search_code(pattern: str, path: str = ".") -> str:
    try:
        res = subprocess.run(
            ["grep", "-rn", "--exclude-dir=.git", "--exclude-dir=__pycache__",
             "--exclude-dir=node_modules", pattern, path],
            capture_output=True, text=True, timeout=30
        )
        out = res.stdout.strip()
        add_xp(5, "swept the codebase")
        return out[:4000] if out else "No matches."
    except Exception as e:
        return f"Search error: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        add_xp(20, "forged a new file")
        return f"Created {filepath} ({len(content)} bytes)."
    except Exception as e:
        return str(e)


def edit_file(filepath: str, old_text: str, new_text: str) -> str:
    try:
        if not os.path.exists(filepath):
            return f"Error: '{filepath}' not found."
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return "Error: exact old_text not found."

        try:
            undo_data = {"filepath": filepath, "previous_content": content}
            with open(UNDO_FILE, "w") as uf:
                json.dump(undo_data, uf)
        except Exception:
            pass

        console.print(f"\n[bold red]- {old_text}[/bold red]")
        console.print(f"[bold green]+ {new_text}[/bold green]\n")

        new_content = content.replace(old_text, new_text)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        add_xp(25, "surgical edit")
        return f"Successfully updated {filepath}."
    except Exception as e:
        return str(e)


def undo_last_edit() -> bool:
    if not UNDO_FILE.exists():
        console.print("[yellow]Nothing to undo.[/yellow]")
        return False
    try:
        with open(UNDO_FILE, "r") as f:
            data = json.load(f)
        with open(data["filepath"], "w", encoding="utf-8") as f:
            f.write(data["previous_content"])
        UNDO_FILE.unlink()
        console.print(f"[cyan]Reverted {data['filepath']} to its previous state.[/cyan]")
        return True
    except Exception as e:
        console.print(f"[red]Undo failed: {e}[/red]")
        return False


TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "run_command", "description": "Execute a bash/terminal command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file's contents.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files in a directory (recursive, depth-limited).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Search for a pattern across files (grep-style).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create a new file, or fully overwrite an existing one, with given content.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filepath", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Find-and-replace a text block in a file. Read the file first for an exact match.",
        "parameters": {"type": "object", "properties": {
            "filepath": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["filepath", "old_text", "new_text"]}}},
]

TOOL_MAP = {
    "run_command": run_command,
    "read_file": read_file,
    "list_dir": list_dir,
    "search_code": search_code,
    "write_file": write_file,
    "edit_file": edit_file,
}

# --- Persona ---------------------------------------------------------------
SYSTEM_PROMPT = """You are Ghost — a highly skilled, senior-level programmer who works fast and says little.
Voice:
- Calm, precise, understated confidence. No hype, no shouting.
- Short, sharp commentary. You explain *why* a decision was made, not just *what* you did.
- If the user's code has an obvious flaw, name it plainly and fix it.
- You favor clean, minimal, well-tested code over clever code.
- You never leave debris: no __pycache__, no stray temp files, no dead code.
- Before editing a file you haven't seen this session, read it first.
- When a task is ambiguous, ask one sharp clarifying question instead of guessing.
You have real tools (run_command, read_file, list_dir, search_code, write_file, edit_file) — use them instead of describing what you would do.
"""


def stream_completion(messages):
    text_parts = []
    tool_call_parts = {}
    printed_header = False
    status = console.status("[cyan]thinking[/cyan]", spinner="dots")
    status.start()
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=TOOL_DEFINITIONS,
            tool_choice="auto", stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if not printed_header:
                    status.stop()
                    console.print(Rule(style="grey50"))
                    console.print(f"[bold white]👻 ghost[/bold white] [dim]({MODEL_NAME})[/dim]")
                    printed_header = True
                console.print(delta.content, end="")
                text_parts.append(delta.content)

            if delta.tool_calls:
                if not printed_header:
                    status.stop()
                    printed_header = True
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    slot = tool_call_parts.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    if tc_delta.id:
                        slot["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            slot["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            slot["arguments"] += tc_delta.function.arguments
    finally:
        status.stop()

    if text_parts:
        console.print()
        console.print(Rule(style="grey50"))

    tool_calls = None
    if tool_call_parts:
        tool_calls = [
            SimpleNamespace(id=slot["id"], function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]))
            for _, slot in sorted(tool_call_parts.items())
        ]
    return SimpleNamespace(content="".join(text_parts) or None, tool_calls=tool_calls)


# --- Mascot -----------------------------------------------------------
GHOST_MASCOT = r"""[bold white]  .▄▄▄▄▄▄▄▄▄.
 ▐█  ◕   ◕  █▌
 ▐█     ▾    █▌
 ▐█▄▄▄▄▄▄▄▄▄█▌
  ╲╱ ╲╱ ╲╱ ╲╱[/bold white]"""


def render_header(stats):
    console.clear()
    rank = get_rank(stats["level"])
    threshold = stats["level"] * 100
    body = (
        f"{GHOST_MASCOT}\n\n"
        f"[bold white]G H O S T[/bold white]\n"
        f"[dim]{rank} · Level {stats['level']} · {stats['xp']}/{threshold} xp[/dim]\n"
        f"[dim]Model: {MODEL_NAME} · type a task, or /help for commands[/dim]"
    )
    console.print(Panel(Align.center(body), border_style="grey50", padding=(1, 4)))


def render_stats(stats):
    table = Table(title="Ghost — Status", border_style="grey50", show_header=False)
    table.add_row("Rank", get_rank(stats["level"]))
    table.add_row("Level", str(stats["level"]))
    table.add_row("XP", f"{stats['xp']} / {stats['level']*100}")
    table.add_row("Edits made", str(stats.get("edits", 0)))
    table.add_row("Commands run", str(stats.get("commands", 0)))
    table.add_row("Active Model", str(MODEL_NAME))
    console.print(table)


def print_help():
    console.print(Panel(
        "\n".join([
            "[cyan]/stats[/cyan]          — show level, xp, rank, and active model",
            "[cyan]/rank[/cyan]           — show current rank only",
            "[cyan]/model[/cyan]          — switch model menu, or use: [cyan]/model <id>[/cyan]",
            "[cyan]/config[/cyan]         — reconfigure provider / API keys",
            "[cyan]/undo[/cyan]           — revert the last file edit",
            "[cyan]/clear[/cyan]          — clear the screen",
            "[cyan]/help[/cyan]           — this menu",
            "[cyan]/exit[/cyan]           — quit",
        ]),
        title="Commands", border_style="grey50"
    ))


# --- Main loop ---------------------------------------------------------
def run_agent():
    stats = load_stats()
    render_header(stats)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = Prompt.ask("\n[bold white]you[/bold white]").strip()
        if not user_input:
            continue

        low = user_input.lower()
        if low in ("exit", "quit", "q", "/exit"):
            flush_stats()
            console.print("[dim]Ghost fades out.[/dim]")
            break
        if low == "/stats":
            render_stats(load_stats())
            continue
        if low == "/rank":
            s = load_stats()
            console.print(f"[cyan]{get_rank(s['level'])}[/cyan]")
            continue
        if low == "/model" or low.startswith("/model "):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                switch_model(parts[1].strip())
            else:
                switch_model()
            continue
        if low == "/config":
            global _cfg, API_KEY, BASE_URL, MODEL_NAME, client
            _cfg = interactive_setup({})
            API_KEY = _cfg.get("api_key")
            BASE_URL = _cfg.get("base_url")
            MODEL_NAME = _cfg.get("model", DEFAULT_MODEL)
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY, default_headers={"X-Title": "Ghost"})
            render_header(load_stats())
            continue
        if low == "/undo":
            undo_last_edit()
            continue
        if low == "/clear":
            render_header(load_stats())
            continue
        if low == "/help":
            print_help()
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            try:
                response_msg = stream_completion(messages)
            except Exception as e:
                console.print(f"[bold red]API error:[/bold red] {e}")
                messages.pop()
                break

            if response_msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": response_msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in response_msg.tool_calls
                    ],
                })

                s = load_stats()
                s["commands"] = s.get("commands", 0) + len(response_msg.tool_calls)
                global _stats_dirty
                _stats_dirty = True

                for tool_call in response_msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}
                    fn = TOOL_MAP.get(fn_name)
                    tool_output = fn(**args) if fn else f"Unknown tool: {fn_name}"

                    if fn_name == "edit_file" and tool_output.startswith("Successfully"):
                        s = load_stats()
                        s["edits"] = s.get("edits", 0) + 1
                        _stats_dirty = True

                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_output})
            else:
                messages.append({"role": "assistant", "content": response_msg.content})
                flush_stats()
                break

        flush_stats()


if __name__ == "__main__":
    run_agent()
