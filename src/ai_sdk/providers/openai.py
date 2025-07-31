from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import openai as _openai

from .language_model import LanguageModel
from .embedding_model import EmbeddingModel

# ---------------------------------------------------------------------------
# Chat completion models
# ---------------------------------------------------------------------------


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

        import asyncio
        import threading

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
# Embedding model implementation
# ---------------------------------------------------------------------------


class OpenAIEmbeddingModel(EmbeddingModel):
    """Implementation of :class:`ai_sdk.providers.embedding_model.EmbeddingModel` for
    OpenAI embedding models (e.g. ``text-embedding-3-small``)."""

    # As of May 2024, the OpenAI *embeddings* endpoint accepts up to 2048 inputs
    # per request (may vary).  We expose this as a *conservative* default.
    max_batch_size: int | None = 2048

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        **default_kwargs: Any,
    ) -> None:
        self._client = _openai.OpenAI(api_key=api_key)
        self._model = model
        self._default_kwargs: Dict[str, Any] = default_kwargs
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    def embed_many(self, values: List[Any], **kwargs: Any) -> Dict[str, Any]:  # noqa: D401
        """OpenAI-specific implementation of :pyfunc:`EmbeddingModel.embed_many`.

        Parameters
        ----------
        values:
            List of values to embed.  The OpenAI embeddings endpoint expects a
            list of **strings** where each string represents a separate input
            (maximum length subject to the underlying model).
        **kwargs:
            Additional arguments forwarded to :pyfunc:`openai.resources.embeddings.Embeddings.create`.
            This can include e.g. ``user`` for request tracking or ``encoding_format``.

        Returns
        -------
        dict
            Mapping containing at least the keys ``values`` and ``embeddings`` as
            described by the parent class.  A flattened ``usage`` dict is
            included if the OpenAI response exposes a ``usage`` field.
        """
        if not values:
            raise ValueError("values must contain at least one item.")

        request_kwargs: Dict[str, Any] = {**self._default_kwargs, **kwargs}

        # Helper performing a single provider call.
        def _single_call(batch: List[Any]) -> Dict[str, Any]:
            resp = self._client.embeddings.create(  # type: ignore[attr-defined]
                model=self._model,
                input=batch,
                **request_kwargs,
            )
            embeddings_batch = [item.embedding for item in resp.data]  # type: ignore[attr-defined]
            usage = None
            if hasattr(resp, "usage"):
                usage = resp.usage.model_dump()
            return {
                "embeddings": embeddings_batch,
                "usage": usage,
                "raw_response": resp,
            }

        # Fast-path – no splitting required.
        if not self.max_batch_size or len(values) <= self.max_batch_size:
            call_res = _single_call(values)
            return {
                "values": values,
                **call_res,
            }

        # Otherwise, split into multiple requests.
        embeddings: List[List[float]] = []
        aggregated_tokens: int = 0
        for i in range(0, len(values), self.max_batch_size):
            sub_batch = values[i : i + self.max_batch_size]
            part = _single_call(sub_batch)
            embeddings.extend(part["embeddings"])
            if part.get("usage") and "total_tokens" in part["usage"]:
                aggregated_tokens += part["usage"]["total_tokens"]

        usage = {"total_tokens": aggregated_tokens} if aggregated_tokens else None
        return {
            "values": values,
            "embeddings": embeddings,
            "usage": usage,
            "raw_response": None,
        }


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
# Public factory helpers
# ---------------------------------------------------------------------------


def openai(
    model: str, *, api_key: Optional[str] = None, **default_kwargs: Any
) -> OpenAIModel:  # noqa: N802
    '''Return a configured :class:`OpenAIModel` instance.

    Parameters
    ----------
    model:
        Identifier of the OpenAI chat model (e.g. "gpt-4o-mini").
    api_key:
        API key used for authentication.  If *None*, the OpenAI client
        falls back to the ``OPENAI_API_KEY`` environment variable.
    **default_kwargs:
        Keyword arguments that will be attached to every subsequent
        request (for example ``temperature`` or ``user``).  They can still
        be overridden per-call.

    Returns
    -------
    OpenAIModel
        Model instance ready for use with the SDK helpers.

    Example
    -------
    >>> from ai_sdk import openai, generate_text
    >>> model = openai("gpt-4o-mini")
    >>> res = await generate_text(model=model, prompt="Hello!")
    '''
    return OpenAIModel(model, api_key=api_key, **default_kwargs)


def embedding(  # noqa: N802 – mimic TypeScript helper naming
    model: str,
    *,
    api_key: Optional[str] = None,
    **default_kwargs: Any,
) -> OpenAIEmbeddingModel:
    """Factory helper that returns an :class:`OpenAIEmbeddingModel` instance.

    Mirrors ``openai.embedding(...)`` semantics from the TS SDK while staying
    a simple function in Python.
    """

    return OpenAIEmbeddingModel(model, api_key=api_key, **default_kwargs)

# ---------------------------------------------------------
# Attach helper as attribute to the *openai* factory function
# to emulate the "openai.embedding(...)" TypeScript API in
#                              Python.
# ---------------------------------------------------------
setattr(openai, "embedding", embedding)  # type: ignore[attr-defined]
