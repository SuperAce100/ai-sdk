from __future__ import annotations

"""High-level helpers that mimic the Vercel AI SDK ``generateText`` and ``streamText`` APIs.

Only the subset required for a first Python port is implemented.  Additional
flags and features will be added in the future.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict

from .providers.language_model import LanguageModel
from .types import (
    AnyMessage,
    ReasoningDetail,
    Source,
    GeneratedFile,
    ToolCall,
    ToolResult,
    TokenUsage,
)  # type: ignore

__all__ = [
    "GenerateTextResult",
    "StreamTextResult",
    "generate_text",
    "stream_text",
]


# ---------------------------------------------------------------------------
# Public Result Objects – kept intentionally lightweight for now
# ---------------------------------------------------------------------------


class _Usage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class GenerateTextResult:
    """Return type for :func:`generate_text`."""

    text: str
    finish_reason: str | None = None
    usage: Optional[TokenUsage] = None
    reasoning: Optional[str] = None
    reasoning_details: Optional[List[ReasoningDetail]] = None
    sources: Optional[List[Source]] = None
    files: Optional[List[GeneratedFile]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    provider_metadata: Dict[str, Any] | None = None
    raw_response: Any | None = None


@dataclass(slots=True)
class StreamTextResult:
    """Return type for :func:`stream_text`.  Provides both the *stream* as well
    as helpers that resolve once the stream has been fully consumed.
    """

    text_stream: AsyncIterator[str]

    # Futures/promises – populated internally by ``_consume_stream``.
    _text_parts: List[str]

    # Extended metadata populated when stream ends (if available)
    finish_reason: str | None = None
    usage: Optional[TokenUsage] = None
    provider_metadata: Optional[Dict[str, Any]] = None

    async def text(self) -> str:  # noqa: D401  # keep parity with TS SDK
        """Return the **full** generated text once the stream has ended."""
        # Consume the stream lazily – only if the caller actually awaits it.
        if not self._text_parts:
            await self._consume_stream()
        return "".join(self._text_parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _consume_stream(self) -> None:
        async for part in self.text_stream:
            self._text_parts.append(part)


# ---------------------------------------------------------------------------
# External API
# ---------------------------------------------------------------------------


def generate_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: Optional[List[AnyMessage]] = None,
    **kwargs: Any,
) -> GenerateTextResult:
    """Generate *non-streaming* text via the given model.

    Parameters
    ----------
    model:
        Instance of a language model, e.g. returned by ``ai_sdk.openai()``.
    prompt / system / messages:
        Mirrors the Vercel AI SDK call signature.  Exactly one of *prompt* or
        *messages* **must** be supplied.
    **kwargs:
        Forwarded directly to the provider implementation (temperature, top_p,
        max_tokens, etc.).
    """
    serialised_messages: Optional[List[Dict[str, Any]]] = None
    if messages is not None:
        serialised_messages = [
            m.to_dict()  # type: ignore[attr-defined]
            for m in messages
        ]

    raw: Dict[str, Any] = model.generate_text(
        prompt=prompt,
        system=system,
        messages=serialised_messages,
        **kwargs,
    )

    return GenerateTextResult(
        text=raw.get("text", ""),
        finish_reason=raw.get("finish_reason"),
        usage=(
            TokenUsage(
                prompt_tokens=raw["usage"].get("prompt_tokens", 0),
                completion_tokens=raw["usage"].get("completion_tokens", 0),
                total_tokens=raw["usage"].get("total_tokens", 0),
            )
            if raw.get("usage")
            else None
        ),
        provider_metadata=raw.get("provider_metadata"),
        raw_response=raw.get("raw_response"),
        reasoning=raw.get("reasoning"),
        reasoning_details=[
            ReasoningDetail(**d) for d in raw.get("reasoning_details", [])
        ]
        if raw.get("reasoning_details")
        else None,
        sources=[Source(**s) for s in raw.get("sources", [])]
        if raw.get("sources")
        else None,
        files=[GeneratedFile(**f) for f in raw.get("files", [])]
        if raw.get("files")
        else None,
        tool_calls=[
            ToolCall(**tc)
            for tc in raw.get("tool_calls", [])
            if isinstance(tc, dict) and tc.get("tool_name")
        ]
        if raw.get("tool_calls")
        else None,
        tool_results=[
            ToolResult(**tr)
            for tr in raw.get("tool_results", [])
            if isinstance(tr, dict) and tr.get("tool_name")
        ]
        if raw.get("tool_results")
        else None,
    )


# ---------------------------------------------------------------------------
# StreamText implementation with callbacks
# ---------------------------------------------------------------------------


from typing import Callable, Awaitable, Coroutine
import asyncio


ChunkCallback = Callable[[str], Awaitable[None] | None]
ErrorCallback = Callable[[Exception], Awaitable[None] | None]
FinishCallback = Callable[[str], Awaitable[None] | None]


def stream_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: Optional[List[AnyMessage]] = None,
    on_chunk: ChunkCallback | None = None,
    on_error: ErrorCallback | None = None,
    on_finish: FinishCallback | None = None,
    # future args accepted via **kwargs for providerOptions etc.
    **kwargs: Any,
) -> StreamTextResult:
    """Stream text from the language model.

    The return value is *not* awaited – streaming starts immediately.  To obtain
    the final text you can either iterate over ``result.text_stream`` or
    ``await result.text()``.
    """

    serialised_messages: Optional[List[Dict[str, Any]]] = None
    if messages is not None:
        serialised_messages = [
            m.to_dict()  # type: ignore[attr-defined]
            for m in messages
        ]

    stream_iter = model.stream_text(
        prompt=prompt,
        system=system,
        messages=serialised_messages,
        **kwargs,
    )

    captured_parts: List[str] = []

    async def _capturing_wrapper() -> AsyncIterator[str]:
        try:
            async for chunk in stream_iter:
                captured_parts.append(chunk)
                if on_chunk:
                    maybe_cor = on_chunk(chunk)
                    if asyncio.iscoroutine(maybe_cor):
                        await maybe_cor
                yield chunk
        except Exception as exc:  # noqa: BLE001
            if on_error:
                maybe = on_error(exc)  # type: ignore[arg-type]
                if asyncio.iscoroutine(maybe):
                    await maybe
            raise
        finally:
            if on_finish:
                full_text = "".join(captured_parts)
                maybe = on_finish(full_text)
                if asyncio.iscoroutine(maybe):
                    await maybe

    return StreamTextResult(
        text_stream=_capturing_wrapper(), _text_parts=captured_parts
    )
