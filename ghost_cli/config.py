"""Configuration management, provider setup wizard, and client holder for Ghost CLI."""

__version__ = "3.0.0"

import json
import os
import subprocess
import sys
from pathlib import Path
from openai import OpenAI
from rich.prompt import Prompt
from rich.panel import Panel

from .ui import console, UI

DATA_DIR = Path.home() / ".ghost"
SESSIONS_DIR = DATA_DIR / "sessions"
CONFIG_FILE = DATA_DIR / "ghost_config.json"
STATS_FILE = DATA_DIR / "ghost_stats.json"
UNDO_FILE = DATA_DIR / ".ghost_undo.json"
HISTORY_FILE = DATA_DIR / "ghost_history.json"
CACHE_FILE = DATA_DIR / "ghost_cache.json"
CHECKPOINT_FILE = DATA_DIR / "ghost_checkpoints.json"
GHOST_MD_FILE = Path("GHOST.md")
MAX_READ_BYTES = 1024 * 1024 * 2  # 2MB

MODEL_PRESETS = {
    "1": ("deepseek/deepseek-v3.2", "DeepSeek V3.2 — Fast, precise coding"),
    "2": ("openai/gpt-4o", "GPT-4o — Deep multi-step reasoning"),
    "3": ("deepseek-ai/deepseek-v4-pro-0813", "DeepSeek V4 Pro — NVIDIA NIM"),
    "4": ("nvidia/nemotron-3.5-lightning:free", "Nemotron 3.5 Lightning — Free NIM"),
    "5": ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B — Fast open-weights model"),
}

DEFAULT_MODEL = "deepseek/deepseek-v3.2"
PERMISSION_MODES = ("plan", "safe", "normal", "autonomous")
DEFAULT_PERMISSION_MODE = "normal"
GITHUB_REPO_URL = "git+https://github.com/ThunderBoltX7/ghost"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg):
    try:
        ensure_data_dir()
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        return True
    except Exception:
        return False


def interactive_setup(cfg):
    """Handles onboarding, provider selection, and permanent saving."""
    console.clear()
    console.print(
        Panel(
            "[bold cyan]Ghost Configuration[/bold cyan]\n\n"
            "Configure your AI provider. This is saved permanently to ~/.ghost/ghost_config.json.",
            border_style="cyan",
        )
    )

    console.print("Select your API Provider:")
    console.print("[cyan]1.[/cyan] OpenRouter (Recommended)")
    console.print("[cyan]2.[/cyan] OmniRoute Gateway")
    console.print("[cyan]3.[/cyan] OpenAI")
    console.print("[cyan]4.[/cyan] Groq")
    console.print("[cyan]5.[/cyan] DeepSeek")
    console.print("[cyan]6.[/cyan] NVIDIA NIM")
    console.print("[cyan]7.[/cyan] Custom (Any OpenAI-compatible endpoint)")

    choice = Prompt.ask("\n[bold white]Enter choice (1-7)[/bold white]", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")

    if choice == "2":
        provider_name = "OmniRoute"
        base_url = Prompt.ask(
            "[bold white]Enter OmniRoute Base URL[/bold white]",
            default="http://localhost:20128/v1",
        ).strip().rstrip("/")
        default_model = Prompt.ask(
            "[bold white]Enter default model name[/bold white]",
            default="auto",
        ).strip()
    elif choice == "7":
        provider_name = "Custom"
        base_url = Prompt.ask("[bold white]Enter Base URL (e.g., http://localhost:11434/v1)[/bold white]").strip().rstrip("/")
        default_model = Prompt.ask("[bold white]Enter default model name[/bold white]", default="gpt-4").strip()
    else:
        providers = {
            "1": ("OpenRouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v3.2"),
            "3": ("OpenAI", "https://api.openai.com/v1", "gpt-4o"),
            "4": ("Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            "5": ("DeepSeek", "https://api.deepseek.com", "deepseek-chat"),
            "6": ("NVIDIA", "https://integrate.api.nvidia.com/v1", "deepseek-ai/deepseek-v4-pro-0813"),
        }
        provider_name, base_url, default_model = providers[choice]

    while True:
        new_key = Prompt.ask(
            f"\n[bold white]Enter your {provider_name} API Key[/bold white]",
            password=True,
            default="omniroute-default" if choice == "2" else "",
        ).strip()

        if len(new_key) >= 1:
            cfg["provider"] = provider_name
            cfg["base_url"] = base_url
            cfg["api_key"] = new_key
            cfg["model"] = default_model
            cfg.setdefault("fast_model", default_model)
            cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
            cfg.setdefault("auto_test", False)
            cfg.setdefault("verbose", False)

            save_config(cfg)

            console.print(f"\n[bold green]✔ {provider_name} configured and saved securely to {CONFIG_FILE}[/bold green]")
            console.print("[dim]Booting system...[/dim]\n")
            return cfg
        else:
            console.print("[red]Invalid key length. Try again.[/red]")


def get_api_config(cfg):
    omni_key = os.environ.get("OMNIROUTE_API_KEY")
    if omni_key and "api_key" not in cfg:
        cfg["api_key"] = omni_key
        cfg["base_url"] = os.environ.get("OMNIROUTE_BASE_URL", "http://localhost:20128/v1").rstrip("/")
        cfg["model"] = os.environ.get("GHOST_MODEL", "auto")
        cfg["provider"] = "OmniRoute"
        cfg.setdefault("fast_model", os.environ.get("GHOST_FAST_MODEL", cfg["model"]))
        cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
        cfg.setdefault("auto_test", False)
        cfg.setdefault("verbose", False)
        return cfg

    env_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    if env_key and "api_key" not in cfg:
        cfg["api_key"] = env_key
        if os.environ.get("NVIDIA_API_KEY") and not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            cfg["base_url"] = "https://integrate.api.nvidia.com/v1"
            cfg["model"] = "deepseek-ai/deepseek-v4-pro-0813"
            cfg["provider"] = "NVIDIA NIM"
        else:
            cfg["base_url"] = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            cfg["model"] = os.environ.get("GHOST_MODEL", DEFAULT_MODEL)
            cfg["provider"] = "Environment Variable"

    if not cfg.get("api_key"):
        cfg = interactive_setup(cfg)

    cfg.setdefault("fast_model", os.environ.get("GHOST_FAST_MODEL", cfg.get("model", DEFAULT_MODEL)))
    cfg.setdefault("permission_mode", DEFAULT_PERMISSION_MODE)
    cfg.setdefault("auto_test", False)
    cfg.setdefault("verbose", False)
    return cfg


def upgrade_ghost():
    """Auto-upgrades Ghost CLI from GitHub using the active Python executable."""
    console.print(f"\n[bold cyan]▲ Connecting to repository to upgrade Ghost...[/bold cyan]")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        GITHUB_REPO_URL,
    ]

    status = console.status("[cyan]Running pip install --upgrade...[/cyan]", spinner="dots")
    status.start()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        status.stop()
        console.print("[red]Upgrade timed out after 180s. Check your internet connection.[/red]")
        return
    except KeyboardInterrupt:
        status.stop()
        console.print("\n[yellow]Upgrade interrupted by user.[/yellow]")
        return
    except Exception as e:
        status.stop()
        console.print(f"[red]Upgrade failed to execute:[/red] {e}")
        return
    finally:
        status.stop()

    if res.returncode == 0:
        console.print("\n[bold green]✔ Ghost upgraded successfully![/bold green]")
        console.print("[dim]Type /exit and run `ghost` again to use the new version.[/dim]\n")
    else:
        err_msg = res.stderr.strip() or res.stdout.strip()
        console.print(f"\n[bold red]Upgrade failed (exit code {res.returncode}):[/bold red]\n{err_msg}\n")


class ClientHolder:
    def __init__(self):
        self.cfg = {}
        self.client = None
        self.model_name = DEFAULT_MODEL
        self.fast_model_name = DEFAULT_MODEL
        self.permission_mode = DEFAULT_PERMISSION_MODE
        self.auto_test = False
        self.verbose = False

    def init(self):
        ensure_data_dir()
        self.cfg = get_api_config(load_config())
        self._apply_cfg()

    def _apply_cfg(self):
        self.model_name = self.cfg.get("model", DEFAULT_MODEL)
        self.fast_model_name = self.cfg.get("fast_model", self.model_name)
        mode = self.cfg.get("permission_mode", DEFAULT_PERMISSION_MODE)
        if mode == "ask":
            mode = "normal"
        elif mode == "strict":
            mode = "safe"
        if mode not in PERMISSION_MODES:
            mode = DEFAULT_PERMISSION_MODE
        self.permission_mode = mode
        self.cfg["permission_mode"] = mode
        self.auto_test = bool(self.cfg.get("auto_test", False))
        self.verbose = bool(self.cfg.get("verbose", False))
        UI.set_verbose(self.verbose)
        self.client = OpenAI(
            base_url=self.cfg.get("base_url"),
            api_key=self.cfg.get("api_key"),
            default_headers={"X-Title": "Ghost"},
        )

    def reconfigure(self):
        self.cfg = get_api_config({})
        self._apply_cfg()

    def switch_model(self, target_model: str = None, fast: bool = False):
        label = "Fast / Explorer Model" if fast else "Primary Model"
        current = self.fast_model_name if fast else self.model_name
        if not target_model:
            console.print(f"\n[bold cyan]Available Models ({label}):[/bold cyan]")
            for key, (m_id, desc) in MODEL_PRESETS.items():
                active_marker = " [bold green](active)[/bold green]" if m_id == current else ""
                console.print(f"[cyan]{key}.[/cyan] {m_id} — [dim]{desc}[/dim]{active_marker}")
            console.print("[cyan]6.[/cyan] Custom model identifier")

            choice = Prompt.ask(
                "\n[bold white]Choose model (1-6 or enter full model ID)[/bold white]",
                default="1",
            ).strip()

            if choice in MODEL_PRESETS:
                target_model = MODEL_PRESETS[choice][0]
            elif choice == "6":
                target_model = Prompt.ask("[bold white]Enter model ID[/bold white]").strip()
            else:
                target_model = choice

        if fast:
            self.fast_model_name = target_model
            self.cfg["fast_model"] = target_model
        else:
            self.model_name = target_model
            self.cfg["model"] = target_model
        save_config(self.cfg)
        console.print(f"\n[bold green]✔ {label} updated to:[/bold green] [bold cyan]{target_model}[/bold cyan]\n")

    def set_permission_mode(self, mode: str) -> bool:
        mode = mode.strip().lower()
        if mode == "ask":
            mode = "normal"
        elif mode == "strict":
            mode = "safe"
        if mode not in PERMISSION_MODES:
            return False
        self.permission_mode = mode
        self.cfg["permission_mode"] = mode
        save_config(self.cfg)
        return True

    def set_auto_test(self, enabled: bool):
        self.auto_test = enabled
        self.cfg["auto_test"] = enabled
        save_config(self.cfg)

    def set_verbose(self, enabled: bool):
        self.verbose = enabled
        self.cfg["verbose"] = enabled
        UI.set_verbose(enabled)
        save_config(self.cfg)


HOLDER = ClientHolder()
