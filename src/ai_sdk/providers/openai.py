from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import openai as _openai

from .language_model import LanguageModel


class OpenAIModel(LanguageModel):
    """Implementation of :class:`~ai_sdk.providers.language_model.LanguageModel` for OpenAI Chat models."""

    def __init__(
        self, model: str, *, api_key: Optional[str] = None, **default_kwargs: Any
    ) -> None:
        # ``openai`` 1.x client prefers an *api_key* argument.  We keep the
        # client instance around so we can re-use TCP connections.
        self._client = _openai.OpenAI(api_key=api_key)
        self._model = model
        # default kwargs (temperature, top_p, etc.) that will be sent on every
        # invocation unless overridden by the caller.
        self._default_kwargs = default_kwargs

    # ---------------------------------------------------------------------
    # LanguageModel interface
    # ---------------------------------------------------------------------
    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronously generate a completion using the Chat Completions API."""
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        chat_messages = _build_chat_messages(
            prompt=prompt, system=system, messages=messages
        )

        # Merge default kwargs with call-site overrides.
        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=chat_messages,
            **request_kwargs,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "unknown"

        # ------------------------------------------------------------------
        # Extract *tool_calls* if present.  The OpenAI SDK exposes them on the
        # message object as ``tool_calls`` – each item contains ``id`` and a
        # nested ``function`` object with ``name`` + ``arguments``.
        # ------------------------------------------------------------------
        tool_calls = []
        if getattr(choice.message, "tool_calls", None):
            import json as _json

            for call in choice.message.tool_calls:  # type: ignore[attr-defined]
                try:
                    args_dict = _json.loads(call.function.arguments)
                except Exception:  # noqa: BLE001 – handle unparsable JSON
                    args_dict = {"raw": call.function.arguments}

                tool_calls.append(
                    {
                        "tool_call_id": call.id,
                        "tool_name": call.function.name,
                        "args": args_dict,
                    }
                )

            # Per the OpenAI spec, the *finish_reason* is set to ``tool_calls``
            # when the assistant returns function invocations.
            finish_reason = "tool"

        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": resp.usage.model_dump() if hasattr(resp, "usage") else None,
            "raw_response": resp,
            "tool_calls": tool_calls or None,
        }

    def stream_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream deltas from the Chat Completions API.

        This function returns an *async iterator* that yields the incremental
        text deltas as soon as they are received from OpenAI.  It purposefully
        hides all non-text events for a first implementation – callers that
        want the raw stream can always wrap this provider directly.
        """

        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        chat_messages = _build_chat_messages(
            prompt=prompt, system=system, messages=messages
        )
        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}

        import asyncio, threading

        async def _generator() -> AsyncIterator[str]:
            # ----------------------------------------------------------------
            # 1) Kick off the *blocking* OpenAI streaming request in a
            #    background thread.
            # ----------------------------------------------------------------
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            def _producer() -> None:
                try:
                    for chunk in self._client.chat.completions.create(
                        model=self._model,
                        messages=chat_messages,
                        stream=True,
                        **request_kwargs,
                    ):  # type: ignore[typing-arg-types]
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            asyncio.run_coroutine_threadsafe(queue.put(content), loop)
                finally:
                    # Signal that the stream is finished
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop = asyncio.get_running_loop()
            threading.Thread(target=_producer, daemon=True).start()

            # ----------------------------------------------------------------
            # 2) Yield items from the queue until a *None* sentinel is
            #    received.
            # ----------------------------------------------------------------
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return _generator()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _build_chat_messages(
    *,
    prompt: str | None,
    system: str | None,
    messages: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Translate the SDK's high-level arguments into OpenAI chat messages."""
    if messages is not None:
        # If explicit messages are provided, optionally prepend the system
        # message.
        chat_messages: List[Dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        return chat_messages

    # Fallback: emulate the *prompt* + optional system prompt API.
    chat_messages = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    if prompt:
        chat_messages.append({"role": "user", "content": prompt})
    return chat_messages


# ---------------------------------------------------------------------------
# Public factory helper – mirrors the TypeScript "openai()" helper.
# ---------------------------------------------------------------------------


def openai(
    model: str, *, api_key: Optional[str] = None, **default_kwargs: Any
) -> OpenAIModel:  # noqa: N802
    """Factory helper that returns an :class:`OpenAIModel` instance.

    Example
    -------
    >>> from ai_sdk import generate_text, openai
    >>> model = openai("gpt-4o-mini")
    >>> result = await generate_text(model=model, prompt="Hello!")
    """
    return OpenAIModel(model, api_key=api_key, **default_kwargs)
