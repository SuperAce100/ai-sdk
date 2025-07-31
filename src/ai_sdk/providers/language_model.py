from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any

from pydantic import BaseModel


class LanguageModel(ABC):
    """Abstract base class for all language model providers.

    A *language model* encapsulates a specific model on a provider (e.g. the
    GPT-4o model on OpenAI).  Each concrete implementation must expose a
    synchronous *generate_text* convenience as well as a *stream_text*
    generator that yields incremental text deltas.  The exact signature is
    intentionally kept very loose to remain provider-agnostic – callers should
    go through the top-level ``ai_sdk.generate_text`` / ``ai_sdk.stream_text``
    helpers instead of invoking the provider directly.
    """

    @abstractmethod
    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return the full completion for the given input.

        The return value **must** at minimum contain a ``text`` key holding the
        generated string as well as a ``finish_reason`` key.  Concrete
        implementations are free to add additional provider-specific fields,
        e.g. ``usage`` or ``raw_response``.
        """

    @abstractmethod
    def stream_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they are produced by the model."""

    # ------------------------------------------------------------------
    # NEW: Native structured output helper
    # ------------------------------------------------------------------

    def generate_object(
        self,
        *,
        schema: type[BaseModel],
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return structured output parsed into *schema*.

        The concrete implementation should leverage the provider's native
        structured output capabilities (if any) and **must** return at least an
        ``object`` key holding the parsed ``schema`` instance.  Additional
        provider-specific fields like ``usage`` or ``raw_response`` can be
        included as desired.
        """
        raise NotImplementedError
