"""High-level helpers that mirror the Vercel AI SDK ``embed`` / ``embedMany``
API.

They provide a *provider-agnostic* façade over concrete
:class:`ai_sdk.providers.embedding_model.EmbeddingModel` implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from .providers.embedding_model import EmbeddingModel

__all__ = [
    "EmbeddingTokenUsage",
    "EmbedManyResult",
    "EmbedResult",
    "embed_many",
    "embed",
]


# ---------------------------------------------------------------------------
# Lightweight Pydantic-compatible usage container (TypeScript parity)
# ---------------------------------------------------------------------------


class EmbeddingTokenUsage(TypedDict, total=False):
    tokens: int


# ---------------------------------------------------------------------------
# Public result containers – kept intentionally lightweight
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EmbedManyResult:
    values: List[Any]
    embeddings: List[List[float]]
    usage: Optional[EmbeddingTokenUsage] = None
    provider_metadata: Optional[Dict[str, Any]] = None
    raw_response: Any | None = None


@dataclass(slots=True)
class EmbedResult:
    value: Any
    embedding: List[float]
    usage: Optional[EmbeddingTokenUsage] = None
    provider_metadata: Optional[Dict[str, Any]] = None
    raw_response: Any | None = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def embed_many(
    *,
    model: EmbeddingModel,
    values: Sequence[Any],
    max_retries: int = 2,
    **kwargs: Any,
) -> EmbedManyResult:  # noqa: D401 – maintain parity with TS naming
    """Embed *values* using the given *model*.

    The helper automatically splits *values* into multiple batches if the
    provider exposes a ``max_batch_size`` limit.
    """

    if not values:
        raise ValueError("values must contain at least one item.")

    # Determine batch size (if any)
    batch_size = getattr(model, "max_batch_size", None)

    # Helper performing a *single* embed_many call with retry logic.
    def _call_with_retries(batch: List[Any]) -> Dict[str, Any]:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= max_retries:
            try:
                return model.embed_many(batch, **kwargs)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempt += 1
                if attempt > max_retries:
                    raise
        # mypy believes we might fall through – we cannot.
        raise RuntimeError("unreachable") from last_exc  # pragma: no cover

    # Fast-path – no batching required.
    if not batch_size or len(values) <= batch_size:
        raw = _call_with_retries(list(values))
        return EmbedManyResult(
            values=list(values),
            embeddings=raw["embeddings"],
            usage=raw.get("usage"),
            provider_metadata=raw.get("provider_metadata"),
            raw_response=raw.get("raw_response"),
        )

    # Split into multiple batches.
    embeddings: List[List[float]] = []
    total_tokens: int = 0
    for i in range(0, len(values), batch_size):  # type: ignore[arg-type]
        sub_batch = list(values)[i : i + batch_size]
        part = _call_with_retries(sub_batch)
        embeddings.extend(part["embeddings"])
        if part.get("usage") and "total_tokens" in part["usage"]:
            total_tokens += part["usage"]["total_tokens"]  # type: ignore[index]

    usage: Optional[EmbeddingTokenUsage] = None
    if total_tokens:
        usage = EmbeddingTokenUsage(tokens=total_tokens)  # type: ignore[call-arg]

    return EmbedManyResult(
        values=list(values),
        embeddings=embeddings,
        usage=usage,
    )


def embed(
    *,
    model: EmbeddingModel,
    value: Any,
    **kwargs: Any,
) -> EmbedResult:  # noqa: D401 – maintain parity with TS naming
    """Embed a *single* value – thin wrapper around :func:`embed_many`."""

    res_many = embed_many(model=model, values=[value], **kwargs)
    return EmbedResult(
        value=value,
        embedding=res_many.embeddings[0],
        usage=res_many.usage,
        provider_metadata=res_many.provider_metadata,
        raw_response=res_many.raw_response,
    )
