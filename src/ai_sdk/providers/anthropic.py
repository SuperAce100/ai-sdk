from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import OpenAI  # type: ignore[import]


from .language_model import LanguageModel
from .openai import _build_chat_messages


class AnthropicModel(LanguageModel):
    """OpenAI SDK compatibility provider for Anthropic chat models."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY"),
        base_url: str = "https://api.anthropic.com/v1/",
        **default_kwargs: Any,
    ) -> None:
        # Use OpenAI SDK to talk to Anthropic endpoint
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._default_kwargs = default_kwargs

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a completion via the OpenAI SDK compatibility layer against Anthropic."""
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Build chat messages using OpenAI helper
        chat_messages = _build_chat_messages(
            prompt=prompt, system=system, messages=messages
        )

        # Merge default kwargs with call-site overrides
        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}
        # Anthropic API requires max_tokens; set a reasonable default if missing
        if "max_tokens" not in request_kwargs:
            request_kwargs["max_tokens"] = 8192

        # Call via OpenAI SDK
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=chat_messages,
            **request_kwargs,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "unknown"
        # Extract tool_calls if present
        tool_calls = []
        if getattr(choice.message, "tool_calls", None):
            import json as _json

            for call in choice.message.tool_calls:  # type: ignore[attr-defined]
                try:
                    args = _json.loads(call.function.arguments)
                except Exception:
                    args = {"raw": call.function.arguments}
                tool_calls.append(
                    {
                        "tool_call_id": call.id,
                        "tool_name": call.function.name,
                        "args": args,
                    }
                )
            finish_reason = "tool"

        # Usage if available
        usage = resp.usage.model_dump() if hasattr(resp, "usage") else None
        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage,
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
        """Stream deltas via OpenAI SDK compatibility."""
        if prompt is None and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Build messages and merge kwargs
        chat_messages = _build_chat_messages(
            prompt=prompt, system=system, messages=messages
        )
        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}

        import asyncio
        import threading

        async def _generator() -> AsyncIterator[str]:
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
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            loop = asyncio.get_running_loop()
            threading.Thread(target=_producer, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return _generator()


# Public factory helper


def anthropic(
    model: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://api.anthropic.com/v1/",
    **default_kwargs: Any,
) -> AnthropicModel:
    """Return a configured AnthropicModel instance using OpenAI SDK compatibility."""
    return AnthropicModel(model, api_key=api_key, base_url=base_url, **default_kwargs)
