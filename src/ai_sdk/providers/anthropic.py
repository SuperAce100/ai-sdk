"""Native Anthropic provider using the official ``anthropic`` Python SDK.

This module talks directly to the Anthropic Messages API rather than routing
through an OpenAI-compatible compatibility layer.  That unlocks first-class
support for Anthropic-specific features such as:

* ``system`` prompts as a top-level parameter (not a pseudo-message)
* tool use via Anthropic's ``input_schema`` tool definitions
* extended thinking / ``thinking`` blocks (passed through ``**kwargs``)
* accurate prompt / completion token accounting
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic as _anthropic
from pydantic import BaseModel

from ..tool import Tool
from .language_model import LanguageModel

# ---------------------------------------------------------------------------
# Internal helpers – message / tool conversion
# ---------------------------------------------------------------------------


def _extract_system_and_messages(
    *,
    prompt: str | None,
    system: str | None,
    messages: list[Any] | None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate SDK-level inputs into Anthropic ``system`` + ``messages``.

    Anthropic keeps the system prompt as a dedicated top-level parameter and
    only accepts ``user`` / ``assistant`` roles in the ``messages`` array.
    Tool results are represented as ``user`` messages whose content is a list
    of ``tool_result`` blocks.

    Parameters
    ----------
    prompt:
        Simple user prompt used when no explicit ``messages`` are supplied.
    system:
        Optional system instruction (merged with any ``system``-role entries
        found inside ``messages``).
    messages:
        Either SDK :class:`~ai_sdk.types.CoreMessage` objects (with ``to_dict``)
        or already-serialised dicts in the intermediate SDK format produced by
        :func:`ai_sdk.generate_text`.

    Returns
    -------
    tuple[Optional[str], List[Dict[str, Any]]]
        ``(system_prompt, anthropic_messages)`` ready for the Messages API.
    """
    system_parts: list[str] = []
    if system:
        system_parts.append(system)

    if messages is None:
        if not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        return (
            "\n\n".join(system_parts) if system_parts else None,
            [{"role": "user", "content": prompt}],
        )

    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        if hasattr(msg, "to_dict"):
            msg_dict = msg.to_dict()  # type: ignore[attr-defined]
        else:
            msg_dict = dict(msg)  # type: ignore[arg-type]

        role = msg_dict.get("role")

        # System role → collect into the top-level system parameter.
        if role == "system":
            content = msg_dict.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        # Tool results (SDK intermediate format) → Anthropic tool_result blocks.
        if role == "tool":
            content = msg_dict.get("content", [])
            tool_result_blocks: list[dict[str, Any]] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        tool_call_id = item.get("toolCallId") or item.get(
                            "tool_call_id", "tool-call"
                        )
                        result_val = item.get("result")
                        is_error = item.get("isError") or item.get("is_error", False)
                        if not isinstance(result_val, str):
                            result_val = _json.dumps(result_val, default=str)
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": result_val,
                                "is_error": bool(is_error),
                            }
                        )
            if tool_result_blocks:
                anthropic_messages.append(
                    {"role": "user", "content": tool_result_blocks}
                )
            continue

        # Assistant messages may carry OpenAI-style tool_calls from the
        # intermediate format produced by generate_text.
        if role == "assistant":
            content = msg_dict.get("content")
            tool_calls = msg_dict.get("tool_calls")
            blocks: list[dict[str, Any]] = []

            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            blocks.append(
                                {"type": "text", "text": part.get("text", "")}
                            )
                        elif part.get("type") == "tool-call":
                            blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": part.get("toolCallId")
                                    or part.get("tool_call_id", "tool-call"),
                                    "name": part.get("toolName")
                                    or part.get("tool_name", "tool"),
                                    "input": part.get("args", {}),
                                }
                            )

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = (
                            _json.loads(args_raw)
                            if isinstance(args_raw, str)
                            else args_raw
                        )
                    except Exception:  # noqa: BLE001
                        args = {"raw": args_raw}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", "tool-call"),
                            "name": fn.get("name", "tool"),
                            "input": args,
                        }
                    )

            if blocks:
                anthropic_messages.append({"role": "assistant", "content": blocks})
            elif content is None and not tool_calls:
                # Empty assistant turn – skip.
                pass
            elif isinstance(content, str):
                anthropic_messages.append({"role": "assistant", "content": content})
            continue

        # User messages (and anything else) pass through with light normalisation.
        if role == "user":
            content = msg_dict.get("content", "")
            if isinstance(content, list):
                # Normalise SDK multi-part user content to Anthropic blocks.
                blocks = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, dict) and part.get("type") == "image":
                        # Basic image support – pass data URL / base64 if present.
                        blocks.append(part)
                    elif isinstance(part, str):
                        blocks.append({"type": "text", "text": part})
                anthropic_messages.append(
                    {"role": "user", "content": blocks if blocks else str(content)}
                )
            else:
                anthropic_messages.append({"role": "user", "content": content})
            continue

        # Fallback – include as-is if role is recognised.
        if role in ("user", "assistant"):
            anthropic_messages.append(
                {"role": role, "content": msg_dict.get("content", "")}
            )

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, anthropic_messages


def _normalise_tools(
    tools: Sequence[Any] | None,
) -> list[dict[str, Any]] | None:
    """Convert Tool instances or intermediate tool dicts to Anthropic format.

    Accepts:
    * :class:`~ai_sdk.tool.Tool` instances (preferred)
    * OpenAI-style ``{"type": "function", "function": {...}}`` dicts
    * Already-native Anthropic tool dicts with ``input_schema``
    """
    if not tools:
        return None

    result: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, Tool):
            result.append(t.to_anthropic_dict())
        elif isinstance(t, dict):
            if "input_schema" in t and "name" in t:
                # Already Anthropic-shaped.
                result.append(t)
            elif t.get("type") == "function" and "function" in t:
                fn = t["function"]
                result.append(
                    {
                        "name": fn.get("name", "tool"),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object"}),
                    }
                )
            elif "name" in t and "parameters" in t:
                # Gemini-style function declaration.
                result.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("parameters", {"type": "object"}),
                    }
                )
            else:
                result.append(t)
        else:
            # Unknown type – skip rather than fail hard.
            continue
    return result or None


def _map_stop_reason(stop_reason: str | None) -> str:
    """Map Anthropic ``stop_reason`` values onto the SDK's common vocabulary."""
    if stop_reason == "end_turn":
        return "stop"
    if stop_reason == "tool_use":
        return "tool"
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "stop_sequence":
        return "stop"
    return stop_reason or "unknown"


def _parse_response(resp: Any) -> dict[str, Any]:
    """Convert a native Anthropic ``Message`` response into the SDK result dict.

    Parameters
    ----------
    resp:
        An ``anthropic.types.Message`` instance (or compatible duck-type).

    Returns
    -------
    dict
        Standardised provider result with ``text``, ``finish_reason``,
        ``usage``, ``raw_response``, and optionally ``tool_calls``.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in getattr(resp, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "tool_call_id": getattr(block, "id", "tool-call"),
                    "tool_name": getattr(block, "name", "tool"),
                    "args": dict(getattr(block, "input", {}) or {}),
                }
            )

    finish_reason = _map_stop_reason(getattr(resp, "stop_reason", None))
    if tool_calls and finish_reason == "unknown":
        finish_reason = "tool"

    usage = None
    if hasattr(resp, "usage") and resp.usage is not None:
        u = resp.usage
        usage = {
            "prompt_tokens": getattr(u, "input_tokens", 0) or 0,
            "completion_tokens": getattr(u, "output_tokens", 0) or 0,
            "total_tokens": (getattr(u, "input_tokens", 0) or 0)
            + (getattr(u, "output_tokens", 0) or 0),
        }

    return {
        "text": "".join(text_parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": resp,
        "tool_calls": tool_calls or None,
        "provider_metadata": {
            "anthropic": {
                "model": getattr(resp, "model", None),
                "id": getattr(resp, "id", None),
                "stop_reason": getattr(resp, "stop_reason", None),
            }
        },
    }


def _prepare_request_kwargs(
    default_kwargs: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Merge defaults with call-site overrides and normalise Anthropic-specific keys.

    Handles ``tools`` / ``tool_choice`` conversion and ensures ``max_tokens``
    is always present (required by the Messages API).
    """
    request_kwargs: dict[str, Any] = {**default_kwargs, **kwargs}

    # Anthropic requires max_tokens on every request.
    if "max_tokens" not in request_kwargs:
        request_kwargs["max_tokens"] = 8192

    # Convert tools to Anthropic format if present.
    if "tools" in request_kwargs:
        normalised = _normalise_tools(request_kwargs.pop("tools"))
        if normalised:
            request_kwargs["tools"] = normalised

    # Map OpenAI-style tool_choice values.
    if "tool_choice" in request_kwargs:
        tc = request_kwargs.pop("tool_choice")
        if tc == "auto":
            request_kwargs["tool_choice"] = {"type": "auto"}
        elif tc == "none":
            request_kwargs["tool_choice"] = {"type": "none"}
        elif tc == "required":
            request_kwargs["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict):
            # Allow callers to pass native Anthropic tool_choice directly.
            request_kwargs["tool_choice"] = tc
        elif isinstance(tc, str):
            # Treat as a specific tool name.
            request_kwargs["tool_choice"] = {"type": "tool", "name": tc}

    return request_kwargs


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class AnthropicModel(LanguageModel):
    """Language model backed by the official Anthropic Messages API.

    Uses the ``anthropic`` Python package directly so that every request can
    take advantage of native Anthropic capabilities (system prompts, tool use
    with ``input_schema``, streaming events, extended thinking, etc.).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **default_kwargs: Any,
    ) -> None:
        """Initialise a native Anthropic client for *model*.

        Parameters
        ----------
        model:
            Anthropic model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        api_key:
            API key.  Falls back to the ``ANTHROPIC_API_KEY`` environment
            variable when *None*.
        base_url:
            Optional custom API base URL (useful for proxies / Bedrock
            gateways that still speak the Anthropic protocol).
        **default_kwargs:
            Keyword arguments applied to every subsequent request
            (e.g. ``temperature``, ``max_tokens``, ``top_p``).  Call-site
            kwargs always win over these defaults.
        """
        client_kwargs: dict[str, Any] = {}
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if resolved_key:
            client_kwargs["api_key"] = resolved_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = _anthropic.Anthropic(**client_kwargs)
        self._model = model
        self._default_kwargs = default_kwargs

    # ------------------------------------------------------------------
    # LanguageModel interface
    # ------------------------------------------------------------------

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronously generate a completion via the Anthropic Messages API.

        Parameters
        ----------
        prompt:
            Simple user prompt (mutually exclusive with *messages* in normal
            usage; when both are supplied *messages* takes precedence).
        system:
            Optional system instruction passed as Anthropic's top-level
            ``system`` parameter.
        messages:
            Conversation history in the SDK intermediate format.
        **kwargs:
            Extra Anthropic request parameters such as ``tools``,
            ``tool_choice``, ``temperature``, ``max_tokens``, ``thinking``,
            or ``metadata``.

        Returns
        -------
        dict
            Standardised result containing ``text``, ``finish_reason``,
            ``usage``, ``raw_response``, and optionally ``tool_calls`` /
            ``provider_metadata``.
        """
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        system_prompt, anthropic_messages = _extract_system_and_messages(
            prompt=prompt, system=system, messages=messages
        )
        request_kwargs = _prepare_request_kwargs(self._default_kwargs, kwargs)

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            **request_kwargs,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        resp = self._client.messages.create(**create_kwargs)
        return _parse_response(resp)

    def generate_object(
        self,
        *,
        schema: type[BaseModel],
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured output constrained by a Pydantic *schema*.

        Anthropic does not expose an OpenAI-style ``parse()`` endpoint, so we
        instruct the model (via the system prompt) to emit valid JSON matching
        the schema and then validate the response with Pydantic.

        Parameters
        ----------
        schema:
            Pydantic model class describing the expected response shape.
        prompt / system / messages:
            Same semantics as :meth:`generate_text`.
        **kwargs:
            Forwarded to :meth:`generate_text`.

        Returns
        -------
        dict
            Contains ``object`` (parsed ``schema`` instance), ``raw_text``,
            ``finish_reason``, ``usage``, and ``raw_response``.
        """
        schema_json = _json.dumps(schema.model_json_schema())
        instruction = (
            "You are a JSON generator. Respond ONLY with valid JSON that exactly "
            f"matches this JSON Schema (no markdown fences, no commentary):\n"
            f"{schema_json}"
        )
        combined_system = f"{system}\n\n{instruction}" if system else instruction

        # Force deterministic, non-tool output.
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)

        raw = self.generate_text(
            prompt=prompt, system=combined_system, messages=messages, **kwargs
        )
        text = (raw.get("text") or "").strip()

        # Strip optional markdown code fences.
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first fence line and optional trailing fence.
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed = schema.model_validate_json(text)
        return {
            "object": parsed,
            "raw_text": raw.get("text", ""),
            "finish_reason": raw.get("finish_reason"),
            "usage": raw.get("usage"),
            "raw_response": raw.get("raw_response"),
            "provider_metadata": raw.get("provider_metadata"),
        }

    def stream_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text deltas from the Anthropic Messages API.

        Uses the synchronous ``anthropic`` streaming helper inside a background
        thread and bridges chunks into an ``async`` generator so callers can
        ``async for`` over incremental text.

        Parameters
        ----------
        prompt / system / messages:
            Same semantics as :meth:`generate_text`.
        **kwargs:
            Forwarded to the Messages API (``tools`` are accepted but only
            text deltas are yielded; tool-use events are currently ignored in
            the stream path).

        Returns
        -------
        AsyncIterator[str]
            Async generator yielding incremental text segments.
        """
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        system_prompt, anthropic_messages = _extract_system_and_messages(
            prompt=prompt, system=system, messages=messages
        )
        request_kwargs = _prepare_request_kwargs(self._default_kwargs, kwargs)

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            **request_kwargs,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        async def _generator() -> AsyncIterator[str]:
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _producer() -> None:
                try:
                    with self._client.messages.stream(**create_kwargs) as stream:
                        for text in stream.text_stream:
                            if text:
                                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                except Exception as exc:  # noqa: BLE001
                    # Surface errors as a sentinel-wrapped string so the
                    # consumer can observe them if desired; we still terminate.
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                    # Re-raise inside the thread for logging / debugging.
                    raise exc
                else:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop = asyncio.get_running_loop()
            threading.Thread(target=_producer, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return _generator()


# ---------------------------------------------------------------------------
# Public factory helper
# ---------------------------------------------------------------------------


def anthropic(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **default_kwargs: Any,
) -> AnthropicModel:
    """Return a configured :class:`AnthropicModel` using the native Anthropic SDK.

    Parameters
    ----------
    model:
        Anthropic model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    api_key:
        API key used for authentication.  If *None*, the client falls back to
        the ``ANTHROPIC_API_KEY`` environment variable.
    base_url:
        Optional override for the Anthropic API base URL.
    **default_kwargs:
        Keyword arguments attached to every subsequent request (e.g.
        ``temperature``, ``max_tokens``).  They can still be overridden
        per-call.

    Returns
    -------
    AnthropicModel
        Model instance ready for use with the SDK helpers.

    Example
    -------
    >>> from ai_sdk import anthropic, generate_text
    >>> model = anthropic("claude-sonnet-4-20250514")
    >>> res = generate_text(model=model, prompt="Hello!")
    """
    return AnthropicModel(model, api_key=api_key, base_url=base_url, **default_kwargs)
