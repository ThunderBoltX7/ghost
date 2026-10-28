"""Main entry point for Ghost CLI."""

import sys
import argparse
import os

try:
    from .config import HOLDER, MODEL_PRESETS, save_config, __version__
    from .ui import console, UI
    from .cli import interactive_repl
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ghost_cli.config import HOLDER, MODEL_PRESETS, save_config, __version__
    from ghost_cli.ui import console, UI
    from ghost_cli.cli import interactive_repl


def run_agent() -> None:
    parser = argparse.ArgumentParser(description="Ghost AI CLI — Powerful agentic coding assistant.")
    parser.add_argument("prompt", nargs="*", help="Optional initial prompt to run.")
    parser.add_argument("-p", type=str, dest="prompt_flag", help="Optional initial prompt.")
    parser.add_argument("--model", type=str, help="Specify model name or preset (sonnet, opus, fast).")
    parser.add_argument("--mode", type=str, choices=["plan", "safe", "normal", "autonomous"], help="Set permission mode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose streaming output.")
    parser.add_argument("--version", action="store_true", help="Show version info.")
    args = parser.parse_args()

    if args.version:
        console.print(f"[bold cyan]Ghost AI CLI v{__version__}[/bold cyan]")
        sys.exit(0)

    HOLDER.init()

    if args.model:
        model_map = {
            "sonnet": "anthropic/claude-3.5-sonnet",
            "opus": "anthropic/claude-3-opus",
            "fast": HOLDER.fast_model_name,
            "deepseek": "deepseek/deepseek-v3.2",
            "gpt4o": "openai/gpt-4o",
        }
        target = model_map.get(args.model.lower(), args.model)
        if target in MODEL_PRESETS:
            target = MODEL_PRESETS[target][0]
        HOLDER.model_name = target
        HOLDER.cfg["model"] = target
        save_config(HOLDER.cfg)

    if args.mode:
        HOLDER.set_permission_mode(args.mode)

    if args.verbose:
        HOLDER.set_verbose(True)

    prompt_parts = list(args.prompt or [])
    if args.prompt_flag:
        prompt_parts.append(args.prompt_flag)

    if prompt_parts:
        initial_prompt = " ".join(prompt_parts)
        UI.display_mascot_banner()
        UI.status(f"Running initial prompt: {initial_prompt}")
        from .agent import AGENT
        AGENT.run_turn(initial_prompt)
        interactive_repl()
    else:
        interactive_repl()


def main() -> None:
    run_agent()


if __name__ == "__main__":
    main()
