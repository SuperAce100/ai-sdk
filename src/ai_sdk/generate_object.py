"""
High-level helpers that mirror *generate_text* / *stream_text* but return a
Pydantic model instance following a caller-supplied schema.

Implementation strategy – provider-agnostic:
1. We delegate to the *existing* provider.generate_text / .stream_text methods.
2. The returned text **must** be valid JSON that conforms to the supplied
   Pydantic *schema* – the caller is responsible for crafting their prompt /
   messages to ensure this.  We still run a best-effort extraction if the model
   wraps the JSON in Markdown fences or prefixes it with prose.
"""

from __future__ import annotations
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Callable,
)
import json as _json
import re

from pydantic import BaseModel, ValidationError

from .providers.language_model import LanguageModel
from .types import TokenUsage, AnyMessage


# ---------------------------------------------------------------------------
# Generic typing helpers
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def _build_schema_instruction(schema: Type[BaseModel]) -> str:
    """Return a terse system prompt guiding the model to emit the required JSON."""

    def _simplify(ann: Any) -> str:  # noqa: ANN401 – generic typing here
        if hasattr(ann, "__name__"):
            return ann.__name__
        return str(ann)

    field_parts = [
        f"'{name}': {_simplify(field.annotation)}"
        for name, field in schema.model_fields.items()
    ]
    joined = ", ".join(field_parts)
    return (
        "You are a JSON generator. Respond ONLY with valid JSON that exactly matches "
        f"this schema: {{ {joined} }}. Do not add any additional keys or explanatory text."
    )


# ---------------------------------------------------------------------------
# Result containers – kept intentionally lightweight (slots=True)
# ---------------------------------------------------------------------------


class GenerateObjectResult(Generic[T]):
    __slots__ = (
        "object",
        "finish_reason",
        "usage",
        "provider_metadata",
        "raw_response",
        "raw_text",
    )

    def __init__(
        self,
        *,
        object: T,
        finish_reason: str | None,
        usage: Optional[TokenUsage],
        provider_metadata: Dict[str, Any] | None,
        raw_response: Any | None,
        raw_text: str,
    ) -> None:
        self.object = object
        self.finish_reason = finish_reason
        self.usage = usage
        self.provider_metadata = provider_metadata
        self.raw_response = raw_response
        self.raw_text = raw_text  # original model output for debugging

    def model_dump(self) -> Dict[str, Any]:  # convenience passthrough
        return self.object.model_dump()


class StreamObjectResult(Generic[T]):
    """Return type for *stream_object* – exposes an async iterator as well as a
    helper that returns the full parsed object once the stream ends.
    """

    __slots__ = (
        "object_stream",
        "_text_parts",
        "finish_reason",
        "usage",
        "provider_metadata",
    )

    def __init__(
        self,
        *,
        object_stream: AsyncIterator[str],
        text_parts: List[str],
        finish_reason: str | None = None,
        usage: Optional[TokenUsage] = None,
        provider_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.object_stream = object_stream
        self._text_parts = text_parts
        self.finish_reason = finish_reason
        self.usage = usage
        self.provider_metadata = provider_metadata

    async def object(self, schema: Type[T]) -> T:  # noqa: D401
        """Return the parsed object once the stream has finished."""

        if not self._text_parts:
            await self._consume_stream()
        full_text = "".join(self._text_parts)
        return _parse_to_schema(full_text, schema)

    # ------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------

    async def _consume_stream(self) -> None:
        async for part in self.object_stream:
            self._text_parts.append(part)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def generate_object(
    *,
    model: LanguageModel,
    schema: Type[T],
    prompt: str | None = None,
    system: str | None = None,
    messages: Optional[List[AnyMessage]] = None,
    **kwargs: Any,
) -> GenerateObjectResult[T]:
    """Generate an object that conforms to *schema* in a single request.

    The helper instructs the language model to emit JSON and validates the
    result against the supplied Pydantic *schema*.

    Parameters
    ----------
    model:
        Language model used for generation.
    schema:
        Pydantic :class:`pydantic.BaseModel` (or compatible) subclass that
        defines the expected output format.
    prompt, system, messages:
        See :func:`ai_sdk.generate_text`.
    **kwargs:
        Forwarded to :pyfunc:`LanguageModel.generate_text`.

    Returns
    -------
    GenerateObjectResult[`T`]
        Wrapper exposing the parsed ``object`` as well as ``raw_text`` and
        standard metadata fields.

    Raises
    ------
    pydantic.ValidationError
        If the model output cannot be parsed into *schema*.
    """

    serialised_messages: Optional[List[Dict[str, Any]]] = None
    if messages is not None:
        serialised_messages = [m.to_dict() for m in messages]  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # 1) Prefer provider-native structured outputs if available.
    #    Fallback to the JSON-parsing approach for providers that do not yet
    #    implement *generate_object* (or if something goes wrong).
    # ------------------------------------------------------------------

    try:
        raw = model.generate_object(  # type: ignore[attr-defined]
            schema=schema,
            prompt=prompt,
            system=system,
            messages=serialised_messages,
            **kwargs,
        )
        obj = raw.get("object")  # type: ignore[assignment]
        if obj is None:
            raise ValueError("Provider did not return 'object' key.")
        text = raw.get("raw_text", "")
    except Exception:
        # ------------------------------------------------------------------
        # Fallback: rely on text generation + manual JSON parsing.
        # Inject a concise system prompt snippet describing the expected schema.
        # ------------------------------------------------------------------
        schema_instruction = _build_schema_instruction(schema)

        if serialised_messages is not None:
            # Prepend as an additional system message.
            serialised_messages = [
                {"role": "system", "content": schema_instruction}
            ] + serialised_messages
            augmented_system = None
        else:
            # Use the *system* field – prepend our instruction.
            if system:
                augmented_system = f"{schema_instruction}\n{system}"
            else:
                augmented_system = schema_instruction

        raw = model.generate_text(
            prompt=prompt,
            system=augmented_system if "augmented_system" in locals() else None,
            messages=serialised_messages,
            **kwargs,
        )
        text = raw.get("text", "")
        obj = _parse_to_schema(text, schema)

    usage = None
    if raw.get("usage"):
        usage = TokenUsage(
            prompt_tokens=raw["usage"].get("prompt_tokens", 0),
            completion_tokens=raw["usage"].get("completion_tokens", 0),
            total_tokens=raw["usage"].get("total_tokens", 0),
        )

    return GenerateObjectResult(
        object=obj,
        finish_reason=raw.get("finish_reason"),
        usage=usage,
        provider_metadata=raw.get("provider_metadata"),
        raw_response=raw.get("raw_response"),
        raw_text=text,
    )


def stream_object(
    *,
    model: LanguageModel,
    schema: Type[T],
    prompt: str | None = None,
    system: str | None = None,
    messages: Optional[List[AnyMessage]] = None,
    on_chunk: Optional[Callable[[str], Any]] = None,
    on_partial: Optional[Callable[[T], Any]] = None,
    **kwargs: Any,
) -> StreamObjectResult[T]:
    """Stream a structured object that conforms to *schema*.

    The underlying model is streamed via
    :pyfunc:`LanguageModel.stream_text`.  Partial deltas are forwarded
    through *on_chunk* while the helper attempts to incrementally parse
    the accumulated text, invoking *on_partial* whenever a valid partial
    object can be produced.

    Parameters
    ----------
    model, schema, prompt, system, messages:
        Same as :func:`generate_object`.
    on_chunk:
        Callback receiving every raw text delta.
    on_partial:
        Callback receiving partial objects of type *T* that were
        successfully parsed.
    **kwargs:
        Extra parameters forwarded to the provider.

    Returns
    -------
    StreamObjectResult[`T`]
        See class docs for the exposed helpers & metadata.
    """

    serialised_messages: Optional[List[Dict[str, Any]]] = None
    if messages is not None:
        serialised_messages = [m.to_dict() for m in messages]  # type: ignore[attr-defined]

    stream = model.stream_text(
        prompt=prompt,
        system=system,
        messages=serialised_messages,
        **kwargs,
    )

    collected: List[str] = []

    async def _forward() -> AsyncIterator[str]:
        async for delta in stream:
            if on_chunk:
                try:
                    on_chunk(delta)
                except Exception:  # noqa: BLE001
                    pass
            collected.append(delta)
            current_text = "".join(collected)
            if on_partial:
                partial_obj = _parse_partial_to_schema(current_text, schema)
                if partial_obj is not None:
                    try:
                        on_partial(partial_obj)
                    except Exception:
                        pass
            yield delta

    return StreamObjectResult(
        object_stream=_forward(),
        text_parts=collected,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_json_block(text: str) -> str:
    """Return the *first* JSON object contained in *text*.

    1. If the entire string is valid JSON, return it as-is.
    2. Otherwise, attempt to locate the first top-level ``{...}`` block.
    """

    text = text.strip()

    # Fast path – the whole string is JSON.
    try:
        _json.loads(text)
        return text
    except Exception:  # noqa: BLE001
        pass

    # Markdown code fences (```json ... ```)
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            _json.loads(candidate)
            return candidate
        except Exception:  # noqa: BLE001
            pass

    # Braces search – very naive but good enough for first implementation.
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            _json.loads(candidate)
            return candidate
        except Exception:  # noqa: BLE001
            pass

    raise ValueError("Could not find a valid JSON object in model output.")


def _parse_partial_to_schema(text: str, schema: Type[T]) -> Optional[T]:
    """Return *schema* instance if possible, else None without raising.

    This is lenient: missing required fields are ignored; only keys present in
    the JSON blob are injected via ``model_construct`` (no validation)."""

    try:
        json_str = _extract_json_block(text)
    except ValueError:
        return None
    try:
        # Try full validation first – this succeeds once the object is complete.
        return schema.model_validate_json(json_str)
    except ValidationError:
        pass  # fall through to lenient mode

    try:
        data = _json.loads(json_str)
        if not isinstance(data, dict):
            return None
        filtered = {k: v for k, v in data.items() if k in schema.model_fields}
        if not filtered:
            return None
        return schema.model_construct(**filtered)
    except Exception:
        return None


def _parse_to_schema(text: str, schema: Type[T]) -> T:
    """Validate + parse *text* (JSON string) into *schema* instance."""

    json_str = _extract_json_block(text)
    try:
        return schema.model_validate_json(json_str)
    except ValidationError as exc:
        raise ValueError(
            "Model output does not conform to the expected schema"
        ) from exc
