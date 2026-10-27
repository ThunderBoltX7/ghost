"""OpenAI client wrappers, streaming completions, retries, and tool fallback for Ghost CLI."""

import hashlib
import json
import random
import time
import uuid
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Tuple

from rich.rule import Rule

from .config import HOLDER
from .ui import console, UI


def _is_retryable_error(e: Exception) -> bool:
    msg = str(e).lower()
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return any(k in msg for k in ("timeout", "connection", "rate limit", "429", "500", "502", "503", "504", "529"))


def stream_completion(messages, tool_definitions, _model_override: str = None, quiet: bool = False):
    from .context import repair_messages
    messages = repair_messages(messages)

    max_retries = 3
    base_delay = 2.0

    extra_params = {}
    base_url = str(HOLDER.cfg.get("base_url", "")).lower()
    if "nvidia.com" in base_url:
        extra_params["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        extra_params["temperature"] = 1
        extra_params["top_p"] = 0.95

    model = _model_override or HOLDER.model_name
    fallback_model = HOLDER.cfg.get("fallback_model")
    usage_data = None

    for attempt in range(1, max_retries + 2):
        is_fallback = attempt > max_retries and fallback_model
        current_model = fallback_model if is_fallback else model

        text_parts = []
        tool_call_parts = {}
        printed_header = False
        status = console.status("[cyan]thinking[/cyan]", spinner="dots") if not quiet else None
        if status:
            status.start()

        try:
            extra = dict(extra_params)
            try:
                stream = HOLDER.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    tools=tool_definitions if tool_definitions else None,
                    tool_choice="auto" if tool_definitions else None,
                    stream=True,
                    stream_options={"include_usage": True},
                    **extra,
                )
            except Exception as stream_opt_err:
                if "stream_options" in str(stream_opt_err).lower():
                    stream = HOLDER.client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        tools=tool_definitions if tool_definitions else None,
                        tool_choice="auto" if tool_definitions else None,
                        stream=True,
                        **extra,
                    )
                else:
                    raise

            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage_data = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    if not quiet:
                        if not printed_header:
                            if status:
                                status.stop()
                            console.print(Rule(style="grey50"))
                            label = f"[bold white]👻 ghost[/bold white] [dim]({current_model})[/dim]"
                            if is_fallback:
                                label += " [yellow](fallback)[/yellow]"
                            console.print(label)
                            printed_header = True
                        console.print(delta.content, end="", markup=False, highlight=False)
                    text_parts.append(delta.content)

                if delta.tool_calls:
                    if not printed_header and status:
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

            break

        except KeyboardInterrupt:
            if status:
                status.stop()
            if not quiet:
                console.print()
            raise
        except Exception as e:
            if status:
                status.stop()
            if not _is_retryable_error(e):
                raise

            if attempt <= max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                if not quiet:
                    console.print(f"[yellow]API error ({e}). Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})...[/yellow]")
                time.sleep(delay)
                text_parts = []
                tool_call_parts = {}
                continue
            elif fallback_model and not is_fallback:
                if not quiet:
                    console.print(f"[yellow]All retries failed. Trying fallback model: {fallback_model}...[/yellow]")
                continue
            else:
                raise
        finally:
            if status:
                status.stop()

    if text_parts and not quiet:
        console.print()
        console.print(Rule(style="grey50"))

    full_text = "".join(text_parts) or None
    tool_calls = None
    if tool_call_parts:
        tool_calls = [
            SimpleNamespace(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
            )
            for _, slot in sorted(tool_call_parts.items())
        ]
    elif full_text and tool_definitions:
        from .tools.registry import _parse_text_tool_calls
        full_text, fallback_calls = _parse_text_tool_calls(full_text)
        if fallback_calls:
            tool_calls = fallback_calls

    from .context import TOKEN_TRACKER, _total_tokens
    from .utils import _count_tokens
    if usage_data:
        in_tok = getattr(usage_data, "prompt_tokens", 0) or 0
        out_tok = getattr(usage_data, "completion_tokens", 0) or 0
        TOKEN_TRACKER.record(in_tok, out_tok)
    else:
        in_tok = _total_tokens(messages)
        out_tok = _count_tokens(full_text or "")
        TOKEN_TRACKER.record(in_tok, out_tok)

    return SimpleNamespace(content=full_text, tool_calls=tool_calls, usage=usage_data)


def call_completion(messages: List[Dict[str, Any]], _model_override: Optional[str] = None, max_tokens: int = 16) -> str:
    """Non-streaming minimal completion call for diagnostics and health checks."""
    from .context import repair_messages
    messages = repair_messages(messages)
    model = _model_override or HOLDER.model_name
    extra_params = {}
    base_url = str(HOLDER.cfg.get("base_url", "")).lower()
    if "nvidia.com" in base_url:
        extra_params["extra_body"] = {"chat_template_kwargs": {"thinking": False}}

    resp = HOLDER.client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=False,
        **extra_params,
    )
    if resp and resp.choices:
        return resp.choices[0].message.content or ""
    return ""


def _tool_call_hash(tc) -> str:
    try:
        args = json.loads(tc.function.arguments or "{}")
    except Exception:
        args = {"_raw": tc.function.arguments or ""}
    key = json.dumps({"n": tc.function.name, "a": args}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()
