# Ghost CLI

Ghost is an elite autonomous AI coding assistant CLI designed for codebase reasoning, multi-step structured planning, safe execution, and seamless developer workflows directly from the terminal.

## Requirements

- **Python**: `>=3.10`
- **Git**: Installed and available on `PATH`

## Installation

### From GitHub
```bash
pip install git+https://github.com/ThunderBoltX7/ghost
```

### From Local Source (Editable with fast tokenization)
```bash
git clone https://github.com/ThunderBoltX7/ghost.git
cd ghost
pip install -e ".[fast]"
```

## Quick Start

Start an interactive session:
```bash
ghost
```

Or provide an initial prompt:
```bash
ghost "Analyze this codebase architecture and run the test suite"
```

### CLI Flags
```bash
ghost --help              # Show help menu
ghost --model <model>     # Specify model preset (sonnet, opus, fast, deepseek, gpt4o) or custom model ID
ghost --mode <mode>       # Set permission mode (plan, safe, normal, autonomous)
ghost --verbose           # Enable verbose execution output
ghost --version           # Show version
```

## Provider Setup

On first launch, Ghost opens an interactive setup wizard to configure your AI provider and API key. Settings are saved permanently to `~/.ghost/ghost_config.json`.

Supported providers:
- **OpenRouter** (Recommended)
- **OmniRoute Gateway**
- **OpenAI**
- **Groq**
- **DeepSeek**
- **NVIDIA NIM**
- **Custom** (Any OpenAI-compatible endpoint)

You can also configure providers using environment variables:
```bash
export OPENROUTER_API_KEY="sk-or-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Permission Modes

Ghost provides four granular permission modes to balance autonomy and safety:

| Mode | Behavior |
|---|---|
| `plan` | Read-only inspection tools execute freely; mutating actions require confirmation. |
| `safe` | Confirmation is required for all mutating actions (file edits/writes, git commits, shell commands). |
| `normal` | Standard routine operations execute automatically; destructive actions (e.g., `rm -rf`, `git push --force`) require confirmation. |
| `autonomous` | Full autonomy; operations execute without interactive prompts. |

Use `/mode <mode>` inside the REPL or press `Shift+Tab` to cycle between modes.

## Slash Commands

Within the interactive REPL (`ghost>`), the following slash commands are available:

| Command | Aliases | Description |
|---|---|---|
| `/help` | `/h`, `/?` | Show the help table of available commands. |
| `/plan <goal>` | | Draft, critique, and execute a multi-step structured plan. |
| `/sessions` | | List and resume saved conversation sessions. |
| `/status` | `/git` | Show git status and active workspace state. |
| `/diff [path]` | | Show git diff for the workspace or specific path. |
| `/commit <msg>` | | Stage changes and commit with message. |
| `/push` | | Push commits to the remote repository. |
| `/undo` | | Revert last file edits or restore shadow git checkpoint. |
| `/checkpoint [label]` | | Create a workspace shadow checkpoint. |
| `/checkpoints` | | List all recorded shadow checkpoints. |
| `/restore` | | Restore workspace to most recent checkpoint. |
| `/rewind [n]` | | Rewind conversation history by n turns (default 1). |
| `/test` | | Run detected test suite (pytest/unittest/npm/yarn/pnpm). |
| `/scan` | | Scan project structure, languages, and codebase index. |
| `/jobs` | `/bg` | List and monitor running background processes. |
| `/tokens` | | Show current conversation token usage and session budget. |
| `/compact` | | Manually compact conversation context into summary. |
| `/clean` | | Clean up checkpoints, undo stack, and index caches. |
| `/export [path]` | | Export conversation transcript to Markdown. |
| `/init` | | Generate or refresh GHOST.md project memory conventions. |
| `/model [id]` | | Show or switch primary AI model. |
| `/mode [mode]` | | Show or switch permission mode (`plan`, `safe`, `normal`, `autonomous`). |
| `/doctor` | | Run system diagnostics and environment health check. |
| `/clear` | | Clear active conversation history. |
| `/exit` | `/quit`, `/q` | Exit Ghost CLI. |

## License

MIT License — see LICENSE for details.
