"""
High-level helpers mimicking Vercel AI SDK ``generateText`` / ``streamText``.

Only the subset required for a first Python port is implemented.  Additional
flags and features will be added in the future.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import (
    Any,
    TypedDict,
)

from ai_sdk.ui_stream import AnyUIStreamPart

from .providers.language_model import LanguageModel

# Optional tool support ------------------------------------------------------
from .tool import Tool  # re-export helper

# On-step result structure
from .types import (
    AnyMessage,
    GeneratedFile,
    OnStepFinishResult,
    ReasoningDetail,
    ResponseMetadata,
    Source,
    TokenUsage,
    ToolCall,
    ToolResult,
)  # type: ignore

# Callback type
OnStepCallback = Callable[[OnStepFinishResult], Any]

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
    usage: TokenUsage | None = None
    reasoning: str | None = None
    reasoning_details: list[ReasoningDetail] | None = None
    sources: list[Source] | None = None
    files: list[GeneratedFile] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    provider_metadata: dict[str, Any] | None = None
    raw_response: Any | None = None


@dataclass(slots=True)
class StreamTextResult:
    """Return type for :func:`stream_text`.  Provides both the *stream* as well
    as helpers that resolve once the stream has been fully consumed.
    """

    text_stream: AsyncIterator[str]

    # Futures/promises – populated internally by ``_consume_stream``.
    _text_parts: list[str]

    # Data stream (UI message stream) yielding structured parts
    full_stream: AsyncIterator[AnyUIStreamPart] | None = None

    # Extended metadata populated when stream ends (if available)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_metadata: dict[str, Any] | None = None

    async def text(self) -> str:  # noqa: D401  # keep parity with TS SDK
        """Return the **full** generated text once the stream has ended."""
        # Consume the stream lazily – only if the caller actually awaits it.
        if not self._text_parts:
            await self._consume_stream()
        return "".join(self._text_parts)

    # Alias to match TS naming on the Python object for convenience
    @property
    def fullStream(self) -> AsyncIterator[AnyUIStreamPart] | None:  # type: ignore[override]
        return self.full_stream

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _consume_stream(self) -> None:
        async for part in self.text_stream:
            self._text_parts.append(part)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_result(raw: dict[str, Any]) -> GenerateTextResult:  # noqa: D401
    """Translate a *raw* provider response into a typed result object."""

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
# Public generate_text helper (tool-aware)
# ---------------------------------------------------------------------------


def generate_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[AnyMessage] | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 8,
    on_step: OnStepCallback | None = None,
    **kwargs: Any,
) -> GenerateTextResult:
    """Generate text in a single, non-streaming request.

    This helper is synchronous and optionally supports iterative *tool
    calling* when ``tools`` are provided.

    Parameters
    ----------
    model:
        Instance of :class:`ai_sdk.providers.language_model.LanguageModel`
        that will be queried.
    prompt:
        User prompt (ignored when ``messages`` is supplied).
    system:
        Optional system prompt prepended to the conversation.
    messages:
        List of :class:`ai_sdk.types.AnyMessage` objects representing the
        conversation so far.
    tools:
        Sequence of :class:`ai_sdk.tool.Tool` instances that can be invoked
        by the model.  Leave empty to disable tool calling.
    max_steps:
        Maximum number of model ↔ tool iterations before aborting (only
        relevant when ``tools`` are supplied).
    on_step:
        Optional callback executed after every model response (including the
        final one).  Receives ``(step_index, GenerateTextResult)``.
    **kwargs:
        Additional keyword arguments forwarded verbatim to
        :pyfunc:`LanguageModel.generate_text`.

    Returns
    -------
    GenerateTextResult
        Dataclass with the following attributes:

        * ``text`` – final model output.
        * ``finish_reason`` – why the model stopped.
        * ``usage`` – provider-reported token usage (may be *None*).
        * ``tool_calls`` / ``tool_results`` – populated when tool calling
          was used.
        * ``provider_metadata`` / ``raw_response`` – provider specific
          information for advanced inspection.

    Raises
    ------
    ValueError
        If both ``prompt`` and ``messages`` are *None*.
    RuntimeError
        If ``max_steps`` is exceeded without the model returning a final
        answer (only applicable when tool calling is active).
    """

    # Fast-path – no tools provided → fall back to the original behaviour.
    if not tools:
        serialised_messages: list[dict[str, Any]] | None = None
        if messages is not None:
            serialised_messages = [
                m.to_dict()  # type: ignore[attr-defined]
                for m in messages
            ]

        raw = model.generate_text(
            prompt=prompt,
            system=system,
            messages=serialised_messages,
            **kwargs,
        )
        return _build_result(raw)

    # ---------------------------------------------------------------------
    # 1) Build the *initial* conversation array for provider calls.
    # ---------------------------------------------------------------------
    conversation: list[dict[str, Any]] = []

    if messages is not None:
        conversation = [m.to_dict() for m in messages]  # type: ignore[attr-defined]
        if system:
            conversation.insert(0, {"role": "system", "content": system})
    else:
        if system:
            conversation.append({"role": "system", "content": system})
        if prompt:
            conversation.append({"role": "user", "content": prompt})

    # Pass Tool instances directly; each provider converts them to its
    # own native tool schema (OpenAI function tools, Anthropic input_schema,
    # Gemini function declarations, etc.).
    tools_for_provider: list[Tool] = list(tools)

    # Keep track of *all* tool results for the final result aggregation.
    aggregated_tool_results: list[ToolResult] = []

    step_idx = 0
    while True:
        raw = model.generate_text(
            messages=conversation,
            tools=tools_for_provider,
            tool_choice="auto",
            **kwargs,
        )

        result = _build_result(raw)

        def _dispatch_step(step_type: str) -> None:  # local helper
            if not on_step:
                return

            meta = ResponseMetadata(
                id=getattr(raw.get("raw_response"), "id", None)
                if isinstance(raw.get("raw_response"), object)
                else None,
                model=raw.get("raw_response").model  # type: ignore[attr-defined]
                if raw.get("raw_response") is not None
                and hasattr(raw.get("raw_response"), "model")
                else None,
                timestamp=getattr(raw.get("raw_response"), "created", None),
                body=raw.get("raw_response"),
            )

            step_obj = OnStepFinishResult(
                step_type=step_type,  # type: ignore[arg-type]
                finish_reason=result.finish_reason,
                usage=result.usage,
                text=result.text,
                tool_calls=result.tool_calls,
                tool_results=result.tool_results,
                response=meta,
                is_continued=bool(result.tool_calls),
                provider_metadata=result.provider_metadata,
            )

            try:
                on_step(step_obj)
            except Exception:  # noqa: BLE001
                pass

        # Dispatch "initial" or "continue" step
        _dispatch_step("initial" if step_idx == 0 else "continue")

        # No tool calls → we are done.
        if not result.tool_calls:
            # Propagate *aggregated* tool results to the final object.
            if aggregated_tool_results and result.tool_results is None:
                result.tool_results = aggregated_tool_results
            return result

        # Guard rail: iteration limit.
        if step_idx >= max_steps:
            raise RuntimeError(
                f"Tool calling exceeded maxSteps={max_steps} without completion."
            )

        import json as _json

        # ------------------------------------------------------------------
        # 2) Run *all* tool calls returned by the model.
        # ------------------------------------------------------------------
        assistant_tool_calls = []
        for tc in result.tool_calls:
            assistant_tool_calls.append(
                {
                    "id": tc.tool_call_id or "tool-call",
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": _json.dumps(tc.args, default=str),
                    },
                }
            )

        conversation.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            }
        )

        for call in result.tool_calls:
            matching = next((t for t in tools if t.name == call.tool_name), None)

            if matching is None:
                # Unknown tool – record error result.
                tool_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    result={"error": "Unknown tool"},
                    is_error=True,
                )
            else:
                try:
                    # *Synchronous* execution for now – async handlers can be
                    # worked in later via ``asyncio.run`` if needed.
                    result_value = matching.handler(**call.args)
                    tool_result = ToolResult(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        result=result_value,
                        is_error=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    tool_result = ToolResult(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        result=str(exc),
                        is_error=True,
                    )

            aggregated_tool_results.append(tool_result)

            # Append the tool result as a CoreToolMessage to keep internal consistency
            from ai_sdk.types import CoreToolMessage, ToolResultPart  # type: ignore

            conversation.append(
                CoreToolMessage(
                    content=[
                        ToolResultPart(
                            tool_call_id=tool_result.tool_call_id or "tool-call",
                            tool_name=tool_result.tool_name or "tool",
                            result=_json.dumps(tool_result.result, default=str),
                            is_error=tool_result.is_error,
                        )
                    ]
                ).to_dict()
            )

        # Callback for the *tool-result* step.
        _dispatch_step("tool-result")

        # Go for another round…
        step_idx += 1


# ---------------------------------------------------------------------------
# StreamText implementation with callbacks
# ---------------------------------------------------------------------------


ChunkCallback = Callable[[str], Awaitable[None] | None]
ErrorCallback = Callable[[Exception], Awaitable[None] | None]
FinishCallback = Callable[[str], Awaitable[None] | None]


def stream_text(
    *,
    model: LanguageModel,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[AnyMessage] | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 8,
    on_step: OnStepCallback | None = None,
    on_chunk: ChunkCallback | None = None,
    on_error: ErrorCallback | None = None,
    on_finish: FinishCallback | None = None,
    # future args accepted via **kwargs for providerOptions etc.
    **kwargs: Any,
) -> StreamTextResult:
    """Asynchronously stream text from the language model.

    This helper returns a :class:`StreamTextResult` whose ``text_stream``
    attribute is an async iterator yielding text *deltas* as soon as they
    arrive from the provider.

    Parameters
    ----------
    model, prompt, system, messages:
        Same as :func:`generate_text`.
    tools, max_steps, on_step:
        Identical semantics to :func:`generate_text`.  When *tools* are
        supplied, streaming falls back to :func:`generate_text` internally
        and emits the final answer as a single chunk.
    on_chunk:
        Callback executed for every delta chunk.
    on_error:
        Callback executed if an exception happens while streaming.
    on_finish:
        Callback executed once the stream completes. Receives the full
        text as its single argument.
    **kwargs:
        Additional arguments forwarded to
        :pyfunc:`LanguageModel.stream_text`.

    Returns
    -------
    StreamTextResult
        Object containing the async ``text_stream`` as well as helpers
        like ``await result.text()`` and metadata fields (``finish_reason``,
        ``usage``, ``provider_metadata``).
    """

    # If tool calling is requested, we currently fall back to the *blocking*
    # implementation under the hood and expose the full text as a *single*
    # delta.  A future revision may introduce proper interleaved streaming
    # including intermediate tool results.

    if tools:
        # Delegate to the synchronous helper and wrap the result.
        final_res = generate_text(
            model=model,
            prompt=prompt,
            system=system,
            messages=messages,
            tools=tools,
            max_steps=max_steps,
            on_step=on_step,
            **kwargs,
        )

        async def _single_yield() -> AsyncIterator[str]:
            yield final_res.text

        # Build a minimal UI data stream that wraps the single final text
        async def _single_full_stream() -> AsyncIterator[AnyUIStreamPart]:
            from ai_sdk.ui_stream import (
                FinishStepPart,
                StartStepPart,
                ToolInputAvailablePart,
                ToolInputStartPart,
                ToolOutputAvailablePart,
                UIFinishMessagePart,
                UIStreamStartPart,
                UITextDeltaPart,
                UITextEndPart,
                UITextStartPart,
            )

            text_id = f"msg_{uuid.uuid4().hex}"
            yield UIStreamStartPart()
            # If tools were involved we can stitch a basic step around it
            if final_res.tool_results or final_res.tool_calls:
                yield StartStepPart()
            if final_res.tool_calls:
                # Emit inputs as available; we do not currently stream tool input deltas
                for call in final_res.tool_calls or []:
                    yield ToolInputStartPart(
                        tool_call_id=call.tool_call_id or "tool-call",
                        tool_name=call.tool_name or "tool",
                    )
                for call in final_res.tool_calls or []:
                    yield ToolInputAvailablePart(
                        tool_call_id=call.tool_call_id or "tool-call",
                        tool_name=call.tool_name or "tool",
                        input=call.args,
                    )

            # Final assistant text as one block
            yield UITextStartPart(id=text_id)
            if final_res.text:
                yield UITextDeltaPart(id=text_id, delta=final_res.text)
            yield UITextEndPart(id=text_id)

            # Emit tool outputs if present
            if final_res.tool_results:
                for tr in final_res.tool_results:
                    yield ToolOutputAvailablePart(
                        tool_call_id=tr.tool_call_id or "tool-call",
                        output=tr.result,
                    )
            if final_res.tool_results or final_res.tool_calls:
                yield FinishStepPart()
            yield UIFinishMessagePart()

        return StreamTextResult(
            text_stream=_single_yield(),
            _text_parts=[final_res.text],
            finish_reason=final_res.finish_reason,
            usage=final_res.usage,
            provider_metadata=final_res.provider_metadata,
            full_stream=_single_full_stream(),
        )

    # ------------------------------------------------------------------
    # Default streaming path (no tool support)
    # ------------------------------------------------------------------

    serialised_messages: list[dict[str, Any]] | None = None
    if messages is not None:
        serialised_messages = [
            m.to_dict()  # type: ignore[attr-defined]
            for m in messages
        ]

    # Use a single provider stream and fan-out to both text_stream and full_stream
    provider_stream = model.stream_text(
        prompt=prompt,
        system=system,
        messages=serialised_messages,
        **kwargs,
    )

    captured_parts: list[str] = []
    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    data_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _producer() -> None:
        try:
            async for chunk in provider_stream:
                captured_parts.append(chunk)
                # Forward callbacks for text deltas
                if on_chunk:
                    maybe_cor = on_chunk(chunk)
                    if asyncio.iscoroutine(maybe_cor):
                        await maybe_cor
                # Fan-out to both queues
                await text_queue.put(chunk)
                await data_queue.put(chunk)
        except Exception as exc:  # noqa: BLE001
            # Send error to data stream and propagate
            await data_queue.put(None)
            if on_error:
                maybe = on_error(exc)  # type: ignore[arg-type]
                if asyncio.iscoroutine(maybe):
                    await maybe
            raise
        finally:
            # Signal termination to both consumers
            await text_queue.put(None)
            await data_queue.put(None)
            if on_finish:
                full_text = "".join(captured_parts)
                maybe = on_finish(full_text)
                if asyncio.iscoroutine(maybe):
                    await maybe

    # Start producer in background
    asyncio.create_task(_producer())

    async def _text_consumer() -> AsyncIterator[str]:
        while True:
            item = await text_queue.get()
            if item is None:
                break
            yield item

    async def _full_stream_consumer() -> AsyncIterator[AnyUIStreamPart]:
        from ai_sdk.ui_stream import (
            UIFinishMessagePart,
            UIStreamStartPart,
            UITextDeltaPart,
            UITextEndPart,
            UITextStartPart,
        )

        text_id = f"msg_{uuid.uuid4().hex}"
        yield UIStreamStartPart()
        yield UITextStartPart(id=text_id)

        while True:
            item = await data_queue.get()
            if item is None:
                break
            yield UITextDeltaPart(id=text_id, delta=item)

        yield UITextEndPart(id=text_id)
        yield UIFinishMessagePart()

    return StreamTextResult(
        text_stream=_text_consumer(),
        _text_parts=captured_parts,
        full_stream=_full_stream_consumer(),
    )
