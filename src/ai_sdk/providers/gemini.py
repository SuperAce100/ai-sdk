"""Native Google Gemini provider using the official ``google-genai`` SDK.

This module talks directly to the Gemini / Generative Language API rather than
routing through an OpenAI-compatible compatibility endpoint.  That unlocks
first-class support for Gemini-specific features such as:

* system instructions via ``GenerateContentConfig.system_instruction``
* function calling via Gemini ``Tool`` / ``FunctionDeclaration`` objects
* multimodal inputs (images, video, audio) passed as native ``Part`` objects
* accurate token usage via ``usage_metadata``
* safety settings, response schemas, and other Gemini-only knobs
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import threading
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from google import genai as _genai
from google.genai import types as _genai_types
from pydantic import BaseModel

from ..tool import Tool
from .language_model import LanguageModel

# ---------------------------------------------------------------------------
# Internal helpers – message / tool conversion
# ---------------------------------------------------------------------------


def _content_to_parts(content: Any) -> list[Any]:
    """Convert heterogeneous content into a list of Gemini ``Part`` values.

    Handles plain strings, SDK multi-part content lists, and already-native
    Gemini part objects.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [_genai_types.Part.from_text(text=content)]
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            if isinstance(item, str):
                parts.append(_genai_types.Part.from_text(text=item))
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(_genai_types.Part.from_text(text=item.get("text", "")))
                elif item.get("type") == "image":
                    # Best-effort: accept a data-URI or bytes under ``image``.
                    image_val = item.get("image")
                    mime = item.get("mimeType") or item.get("mime_type") or "image/png"
                    if isinstance(image_val, bytes):
                        parts.append(
                            _genai_types.Part.from_bytes(data=image_val, mime_type=mime)
                        )
                    elif isinstance(image_val, str):
                        parts.append(_genai_types.Part.from_text(text=image_val))
                else:
                    # Unknown structured part – serialise as text.
                    parts.append(
                        _genai_types.Part.from_text(text=_json.dumps(item, default=str))
                    )
            else:
                parts.append(_genai_types.Part.from_text(text=str(item)))
        return parts
    return [_genai_types.Part.from_text(text=str(content))]


def _extract_system_and_contents(
    *,
    prompt: str | None,
    system: str | None,
    messages: list[Any] | None,
) -> tuple[str | None, list[_genai_types.Content]]:
    """Translate SDK-level inputs into Gemini ``system_instruction`` + ``contents``.

    Gemini uses a dedicated ``system_instruction`` field (not a pseudo-message)
    and a ``contents`` list whose roles are only ``user`` / ``model``.  Tool
    results are represented as ``user`` turns containing ``function_response``
    parts; assistant tool calls become ``model`` turns with ``function_call``
    parts.

    Parameters
    ----------
    prompt:
        Simple user prompt used when no explicit ``messages`` are supplied.
    system:
        Optional system instruction (merged with any ``system``-role entries
        found inside ``messages``).
    messages:
        Either SDK CoreMessage objects (with ``to_dict``) or intermediate
        serialised dicts produced by :func:`ai_sdk.generate_text`.

    Returns
    -------
    tuple[Optional[str], List[Content]]
        ``(system_instruction, contents)`` ready for ``generate_content``.
    """
    system_parts: list[str] = []
    if system:
        system_parts.append(system)

    if messages is None:
        if not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        contents = [
            _genai_types.Content(
                role="user",
                parts=_content_to_parts(prompt),
            )
        ]
        return (
            "\n\n".join(system_parts) if system_parts else None,
            contents,
        )

    contents: list[_genai_types.Content] = []

    for msg in messages:
        if hasattr(msg, "to_dict"):
            msg_dict = msg.to_dict()  # type: ignore[attr-defined]
        else:
            msg_dict = dict(msg)  # type: ignore[arg-type]

        role = msg_dict.get("role")

        if role == "system":
            content = msg_dict.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue

        if role == "tool":
            # Tool results → user turn with function_response parts.
            content = msg_dict.get("content", [])
            parts: list[Any] = []
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    tool_name = item.get("toolName") or item.get("tool_name", "tool")
                    result_val = item.get("result")
                    # Gemini expects a dict response; try to parse JSON strings.
                    if isinstance(result_val, str):
                        try:
                            response_payload: Any = _json.loads(result_val)
                        except Exception:  # noqa: BLE001
                            response_payload = {"result": result_val}
                    elif isinstance(result_val, dict):
                        response_payload = result_val
                    else:
                        response_payload = {"result": result_val}
                    parts.append(
                        _genai_types.Part.from_function_response(
                            name=tool_name,
                            response=(
                                response_payload
                                if isinstance(response_payload, dict)
                                else {"result": response_payload}
                            ),
                        )
                    )
            if parts:
                contents.append(_genai_types.Content(role="user", parts=parts))
            continue

        if role == "assistant":
            content = msg_dict.get("content")
            tool_calls = msg_dict.get("tool_calls")
            parts = []

            if isinstance(content, str) and content:
                parts.append(_genai_types.Part.from_text(text=content))
            elif isinstance(content, list):
                parts.extend(_content_to_parts(content))

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
                    if not isinstance(args, dict):
                        args = {"value": args}
                    parts.append(
                        _genai_types.Part.from_function_call(
                            name=fn.get("name", "tool"),
                            args=args,
                        )
                    )

            if parts:
                contents.append(_genai_types.Content(role="model", parts=parts))
            continue

        if role == "user":
            parts = _content_to_parts(msg_dict.get("content", ""))
            if parts:
                contents.append(_genai_types.Content(role="user", parts=parts))
            continue

        # Fallback for unknown roles – treat as user text.
        parts = _content_to_parts(msg_dict.get("content", ""))
        if parts:
            contents.append(_genai_types.Content(role="user", parts=parts))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _normalise_tools(
    tools: Sequence[Any] | None,
) -> list[_genai_types.Tool] | None:
    """Convert Tool instances or intermediate tool dicts to Gemini ``Tool`` objects.

    Accepts:
    * :class:`~ai_sdk.tool.Tool` instances (preferred)
    * OpenAI-style ``{"type": "function", "function": {...}}`` dicts
    * Gemini-style function declaration dicts with ``name`` + ``parameters``
    * Already-constructed ``google.genai.types.Tool`` instances
    """
    if not tools:
        return None

    declarations: list[_genai_types.FunctionDeclaration] = []
    passthrough_tools: list[_genai_types.Tool] = []

    for t in tools:
        if isinstance(t, _genai_types.Tool):
            passthrough_tools.append(t)
            continue

        name: str
        description: str = ""
        parameters: dict[str, Any] | None = None

        if isinstance(t, Tool):
            name = t.name
            description = t.description
            parameters = t.parameters
        elif isinstance(t, dict):
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                name = fn.get("name", "tool")
                description = fn.get("description", "")
                parameters = fn.get("parameters")
            elif "name" in t:
                name = t["name"]
                description = t.get("description", "")
                # Anthropic uses input_schema; Gemini / intermediate use parameters.
                parameters = t.get("parameters") or t.get("input_schema")
            else:
                continue
        else:
            continue

        decl_kwargs: dict[str, Any] = {"name": name, "description": description}
        if parameters:
            decl_kwargs["parameters"] = parameters
        declarations.append(_genai_types.FunctionDeclaration(**decl_kwargs))

    if not declarations and not passthrough_tools:
        return None

    result: list[_genai_types.Tool] = list(passthrough_tools)
    if declarations:
        result.append(_genai_types.Tool(function_declarations=declarations))
    return result


def _map_finish_reason(reason: Any) -> str:
    """Map Gemini ``FinishReason`` enum / string values to the SDK vocabulary."""
    if reason is None:
        return "unknown"
    # Enum members expose ``.name``; strings pass through.
    name = getattr(reason, "name", None) or str(reason)
    name_upper = name.upper()
    if name_upper in ("STOP", "FINISH_REASON_STOP"):
        return "stop"
    if name_upper in ("MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"):
        return "length"
    if name_upper in (
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "FINISH_REASON_SAFETY",
    ):
        return "content-filter"
    if "TOOL" in name_upper or "FUNCTION" in name_upper:
        return "tool"
    return name.lower() if isinstance(name, str) else "unknown"


def _parse_response(resp: Any) -> dict[str, Any]:
    """Convert a Gemini ``GenerateContentResponse`` into the SDK result dict.

    Parameters
    ----------
    resp:
        A ``google.genai.types.GenerateContentResponse`` (or compatible).

    Returns
    -------
    dict
        Standardised provider result with ``text``, ``finish_reason``,
        ``usage``, ``raw_response``, and optionally ``tool_calls``.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    finish_reason = "unknown"

    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        finish_reason = _map_finish_reason(getattr(candidate, "finish_reason", None))
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            # Text part.
            text_val = getattr(part, "text", None)
            if text_val:
                text_parts.append(text_val)

            # Function call part.
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = dict(getattr(fc, "args", None) or {})
                tool_calls.append(
                    {
                        # Gemini doesn't always return a stable id; synthesise one.
                        "tool_call_id": getattr(fc, "id", None)
                        or f"call_{uuid.uuid4().hex[:12]}",
                        "tool_name": getattr(fc, "name", "tool"),
                        "args": args,
                    }
                )

    if tool_calls:
        finish_reason = "tool"

    # Token usage.
    usage = None
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
        completion_tokens = getattr(um, "candidates_token_count", 0) or 0
        total_tokens = getattr(um, "total_token_count", None)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # Fall back to the convenience ``.text`` accessor when parts were empty.
    text = "".join(text_parts)
    if not text:
        try:
            text = getattr(resp, "text", None) or ""
        except Exception:  # noqa: BLE001 – .text can raise on multi-part / blocked
            text = ""

    return {
        "text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": resp,
        "tool_calls": tool_calls or None,
        "provider_metadata": {
            "gemini": {
                "model_version": getattr(resp, "model_version", None),
                "response_id": getattr(resp, "response_id", None),
            }
        },
    }


def _build_config(
    *,
    system_instruction: str | None,
    default_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[_genai_types.GenerateContentConfig, dict[str, Any]]:
    """Build a ``GenerateContentConfig`` and return leftover non-config kwargs.

    Recognises common OpenAI / SDK kwargs (``temperature``, ``max_tokens``,
    ``top_p``, ``tools``, ``tool_choice``) and maps them onto Gemini's config
    object.  Unknown keys are returned as a separate dict so callers can decide
    what to do with them.
    """
    merged: dict[str, Any] = {**default_kwargs, **kwargs}
    config_kwargs: dict[str, Any] = {}

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    # Sampling parameters.
    if "temperature" in merged:
        config_kwargs["temperature"] = merged.pop("temperature")
    if "top_p" in merged:
        config_kwargs["top_p"] = merged.pop("top_p")
    if "top_k" in merged:
        config_kwargs["top_k"] = merged.pop("top_k")
    if "max_tokens" in merged:
        config_kwargs["max_output_tokens"] = merged.pop("max_tokens")
    if "max_output_tokens" in merged:
        config_kwargs["max_output_tokens"] = merged.pop("max_output_tokens")
    if "stop_sequences" in merged:
        config_kwargs["stop_sequences"] = merged.pop("stop_sequences")
    if "candidate_count" in merged:
        config_kwargs["candidate_count"] = merged.pop("candidate_count")
    if "seed" in merged:
        config_kwargs["seed"] = merged.pop("seed")
    if "response_mime_type" in merged:
        config_kwargs["response_mime_type"] = merged.pop("response_mime_type")
    if "response_schema" in merged:
        config_kwargs["response_schema"] = merged.pop("response_schema")
    if "safety_settings" in merged:
        config_kwargs["safety_settings"] = merged.pop("safety_settings")

    # Tools.
    if "tools" in merged:
        gemini_tools = _normalise_tools(merged.pop("tools"))
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

    # tool_choice / tool_config mapping.
    if "tool_choice" in merged:
        tc = merged.pop("tool_choice")
        if tc == "auto":
            config_kwargs["tool_config"] = _genai_types.ToolConfig(
                function_calling_config=_genai_types.FunctionCallingConfig(mode="AUTO")
            )
        elif tc == "none":
            config_kwargs["tool_config"] = _genai_types.ToolConfig(
                function_calling_config=_genai_types.FunctionCallingConfig(mode="NONE")
            )
        elif tc == "required":
            config_kwargs["tool_config"] = _genai_types.ToolConfig(
                function_calling_config=_genai_types.FunctionCallingConfig(mode="ANY")
            )
        elif isinstance(tc, str):
            # Specific function name.
            config_kwargs["tool_config"] = _genai_types.ToolConfig(
                function_calling_config=_genai_types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tc],
                )
            )
        elif isinstance(tc, dict) or isinstance(tc, _genai_types.ToolConfig):
            config_kwargs["tool_config"] = tc

    if "tool_config" in merged:
        config_kwargs["tool_config"] = merged.pop("tool_config")

    config = _genai_types.GenerateContentConfig(**config_kwargs)
    return config, merged


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class GeminiModel(LanguageModel):
    """Language model backed by the official Google Gemini / GenAI API.

    Uses the ``google-genai`` Python package directly so that every request can
    take advantage of native Gemini capabilities (system instructions, function
    calling, multimodal parts, safety settings, response schemas, etc.).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **default_kwargs: Any,
    ) -> None:
        """Initialise a native Gemini client for *model*.

        Parameters
        ----------
        model:
            Gemini model identifier (e.g. ``"gemini-2.0-flash"``,
            ``"gemini-2.5-flash"``).
        api_key:
            API key.  Falls back to ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
            environment variables when *None*.
        **default_kwargs:
            Keyword arguments applied to every subsequent request
            (e.g. ``temperature``, ``max_tokens``, ``top_p``).  Call-site
            kwargs always win over these defaults.
        """
        resolved_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        client_kwargs: dict[str, Any] = {}
        if resolved_key:
            client_kwargs["api_key"] = resolved_key

        self._client = _genai.Client(**client_kwargs)
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
        """Synchronously generate a completion via the Gemini generateContent API.

        Parameters
        ----------
        prompt:
            Simple user prompt (used when *messages* is not supplied).
        system:
            Optional system instruction passed via
            ``GenerateContentConfig.system_instruction``.
        messages:
            Conversation history in the SDK intermediate format.
        **kwargs:
            Extra Gemini request parameters such as ``tools``,
            ``tool_choice``, ``temperature``, ``max_tokens``,
            ``safety_settings``, or ``response_mime_type``.

        Returns
        -------
        dict
            Standardised result containing ``text``, ``finish_reason``,
            ``usage``, ``raw_response``, and optionally ``tool_calls`` /
            ``provider_metadata``.
        """
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        system_instruction, contents = _extract_system_and_contents(
            prompt=prompt, system=system, messages=messages
        )
        config, _extra = _build_config(
            system_instruction=system_instruction,
            default_kwargs=self._default_kwargs,
            kwargs=kwargs,
        )

        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
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
        """Generate structured output using Gemini's native JSON / schema mode.

        Sets ``response_mime_type`` to ``application/json`` and, when possible,
        supplies the Pydantic-derived JSON schema as ``response_schema`` so the
        model is constrained to emit valid structured data.

        Parameters
        ----------
        schema:
            Pydantic model class describing the expected response shape.
        prompt / system / messages:
            Same semantics as :meth:`generate_text`.
        **kwargs:
            Forwarded to :meth:`generate_text` (tools are stripped since
            structured output and tool use are mutually exclusive here).

        Returns
        -------
        dict
            Contains ``object`` (parsed ``schema`` instance), ``raw_text``,
            ``finish_reason``, ``usage``, and ``raw_response``.
        """
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)

        # Ask Gemini for strict JSON output conforming to the schema.
        kwargs.setdefault("response_mime_type", "application/json")
        try:
            kwargs.setdefault("response_schema", schema)
        except Exception:  # noqa: BLE001 – schema injection is best-effort
            pass

        # Also include a textual instruction as a safety net for models that
        # don't fully honour response_schema.
        schema_json = _json.dumps(schema.model_json_schema())
        instruction = (
            "Respond ONLY with valid JSON matching this schema (no markdown, "
            f"no commentary):\n{schema_json}"
        )
        combined_system = f"{system}\n\n{instruction}" if system else instruction

        raw = self.generate_text(
            prompt=prompt, system=combined_system, messages=messages, **kwargs
        )
        text = (raw.get("text") or "").strip()

        # Strip optional markdown code fences.
        if text.startswith("```"):
            lines = text.split("\n")
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
        """Stream text deltas from the Gemini generateContent stream API.

        Uses the synchronous ``generate_content_stream`` helper inside a
        background thread and bridges chunks into an ``async`` generator.

        Parameters
        ----------
        prompt / system / messages:
            Same semantics as :meth:`generate_text`.
        **kwargs:
            Forwarded to the generate content config builder.

        Returns
        -------
        AsyncIterator[str]
            Async generator yielding incremental text segments.
        """
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        system_instruction, contents = _extract_system_and_contents(
            prompt=prompt, system=system, messages=messages
        )
        config, _extra = _build_config(
            system_instruction=system_instruction,
            default_kwargs=self._default_kwargs,
            kwargs=kwargs,
        )

        async def _generator() -> AsyncIterator[str]:
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _producer() -> None:
                try:
                    stream = self._client.models.generate_content_stream(
                        model=self._model,
                        contents=contents,
                        config=config,
                    )
                    for chunk in stream:
                        text_val = None
                        try:
                            text_val = getattr(chunk, "text", None)
                        except Exception:  # noqa: BLE001
                            text_val = None
                        if not text_val:
                            # Fall back to inspecting candidate parts.
                            candidates = getattr(chunk, "candidates", None) or []
                            if candidates:
                                parts = (
                                    getattr(
                                        getattr(candidates[0], "content", None),
                                        "parts",
                                        None,
                                    )
                                    or []
                                )
                                texts = [
                                    getattr(p, "text", "") or ""
                                    for p in parts
                                    if getattr(p, "text", None)
                                ]
                                text_val = "".join(texts) or None
                        if text_val:
                            asyncio.run_coroutine_threadsafe(queue.put(text_val), loop)
                finally:
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


def gemini(
    model: str,
    *,
    api_key: str | None = None,
    **default_kwargs: Any,
) -> GeminiModel:
    """Return a configured :class:`GeminiModel` using the native Google GenAI SDK.

    Parameters
    ----------
    model:
        Gemini model identifier (e.g. ``"gemini-2.0-flash"``).
    api_key:
        API key used for authentication.  If *None*, the client falls back to
        ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` environment variables.
    **default_kwargs:
        Keyword arguments attached to every subsequent request (e.g.
        ``temperature``, ``max_tokens``).  They can still be overridden
        per-call.

    Returns
    -------
    GeminiModel
        Model instance ready for use with the SDK helpers.

    Example
    -------
    >>> from ai_sdk import gemini, generate_text
    >>> model = gemini("gemini-2.0-flash")
    >>> res = generate_text(model=model, prompt="Hello!")
    """
    # ``base_url`` was accepted by the previous OpenAI-compat implementation;
    # ignore it gracefully if callers still pass it.
    default_kwargs.pop("base_url", None)
    return GeminiModel(model, api_key=api_key, **default_kwargs)
