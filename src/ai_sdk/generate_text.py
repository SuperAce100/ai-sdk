"""
High-level helpers that mimic the Vercel AI SDK ``generateText`` and ``streamText`` APIs.

Only the subset required for a first Python port is implemented.  Additional
flags and features will be added in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    TypedDict,
    Callable,
    Awaitable,
)

import asyncio

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

# Optional tool support ------------------------------------------------------

from .tool import Tool  # re-export helper

# On-step result structure
from .types import OnStepFinishResult, ResponseMetadata

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
# Internal helpers
# ---------------------------------------------------------------------------


def _build_result(raw: Dict[str, Any]) -> GenerateTextResult:  # noqa: D401
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
    messages: Optional[List[AnyMessage]] = None,
    tools: Optional[List[Tool]] = None,
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
        serialised_messages: Optional[List[Dict[str, Any]]] = None
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
    conversation: List[Dict[str, Any]] = []

    if messages is not None:
        conversation = [m.to_dict() for m in messages]  # type: ignore[attr-defined]
        if system:
            conversation.insert(0, {"role": "system", "content": system})
    else:
        if system:
            conversation.append({"role": "system", "content": system})
        if prompt:
            conversation.append({"role": "user", "content": prompt})

    # Pre-compute the JSON schema for the provider.
    tools_schema = [t.to_openai_dict() for t in tools]

    # Keep track of *all* tool results for the final result aggregation.
    aggregated_tool_results: List[ToolResult] = []

    step_idx = 0
    while True:
        raw = model.generate_text(
            messages=conversation,
            tools=tools_schema,
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
                )
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
    messages: Optional[List[AnyMessage]] = None,
    tools: Optional[List[Tool]] = None,
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

        return StreamTextResult(
            text_stream=_single_yield(),
            _text_parts=[final_res.text],
            finish_reason=final_res.finish_reason,
            usage=final_res.usage,
            provider_metadata=final_res.provider_metadata,
        )

    # ------------------------------------------------------------------
    # Default streaming path (no tool support)
    # ------------------------------------------------------------------

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
