# Changelog

## [3.0.0] — 2026-10-28

### Breaking
- Python 3.10+ required (was unspecified)
- Package renamed from implicit to `ghost-cli` on PyPI / pip
- Permission mode `ask` → `normal`, `strict` → `safe` (auto-migrated from saved config)

### Added
- `pyproject.toml` — pip-installable via `pip install git+https://github.com/ThunderBoltX7/ghost`
- `ghost` entry point on PATH after install
- Auto shadow checkpoint at every turn start (enables `/undo` without `/checkpoint`)
- Loop detection: same tool call repeated 3× injects a nudge; 2 nudges stops the turn
- Step limit from config (`max_steps`, default 40; was hardcoded 15)
- Parallel read-only tool execution (ThreadPoolExecutor, up to 6 workers)
- Two-stage context compaction: microcompaction at 60% (no API call), LLM summary at 75%
- `rich.Live` streaming Markdown renderer with syntax-highlighted code blocks
- Tool result cards: compact per-call line with timing and success/fail icon
- Turn summary after editing turns (files changed, context %, /undo tip)
- prompt_toolkit live bottom toolbar (model, mode, context %, branch, level)
- Shift+Tab cycles permission modes (plan → normal → autonomous)
- `call_completion()` for health checks in /doctor
- `diagnose_error()`: categorized error panels with what/why/fix
- `render_diff_viewer()`: side-by-side at >120 cols, unified otherwise
- `SessionTokenTracker`: real token usage from API; `/tokens` command
- `repair_messages()`: called before every API request and on session load
- `get_git_branch()`: branch shown in system prompt, status bar, and banner
- `/export`, `/checkpoint`, `/checkpoints`, `/restore`, `/tokens`, `/compact`, `/clean` commands
- Session list sorted by current cwd first, with relative timestamps
- Environment block in system prompt (OS, shell, cwd, time, git context)
- Process tree kill on timeout/interrupt (cross-platform: killpg / taskkill)
- 1 MB output cap per stream to prevent memory exhaustion
- `--` before ripgrep pattern (fixes patterns starting with `-`)
- `markup=False, highlight=False` on raw streamed text (prevents Rich mangling brackets)
- `atexit` stats flush so XP is never lost
- GHOST.md project memory auto-loaded into system prompt (6k token cap)
- `prompt_toolkit` input with persistent history, slash autocomplete, and descriptions
- `shlex`-based compound command splitter in permissions (catches `&&`, `||`, `$(...)`, pipes)
- Four permission modes with correct semantics: plan / safe / normal / autonomous

### Fixed
- `json` import missing in git_manager.py (checkpoints silently failed)
- `run_test_fix_loop` crash (`from .agent import run_turn` didn't exist)
- `/model` crash (`HOLDER.presets` didn't exist)
- `/doctor` crash (`call_completion` didn't exist)
- Permission mode `safe` was less strict than `normal` (inverted logic)
- `compact_messages` could split tool_call/tool pairs, causing API 400 errors
- Ctrl+C during tool execution left dangling tool_call messages in history
- `console.print(delta.content)` interpreted `[brackets]` as Rich markup

### Infrastructure
- pytest test suite covering permissions, context, utils, git manager
- `ruff` and `mypy` configured in pyproject.toml
- `.gitignore` covering Python, virtual envs, Ghost data dirs
