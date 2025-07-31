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
    "cosine_similarity",
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
# Utility helpers
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:  # noqa: D401
    """Return the *cosine similarity* between two vectors.

    Parameters
    ----------
    vec_a, vec_b:
        Numeric sequences (e.g. ``list`` or ``numpy.ndarray``) of equal length.

    Returns
    -------
    float
        Cosine similarity in the range ``[-1.0, 1.0]`` where ``1.0`` means the
        vectors point in the same direction and ``0.0`` indicates orthogonality.

    Raises
    ------
    ValueError
        If the vectors have different lengths or at least one vector is the
        all-zero vector (magnitude ``0``).
    """

    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length.")

    import math

    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Vectors must not be zero-vectors.")

    return dot / (norm_a * norm_b)

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
    """Embed a sequence of *values* using the given embedding *model*.

    Parameters
    ----------
    model:
        Instance of :class:`ai_sdk.providers.embedding_model.EmbeddingModel` that
        will be used to embed the input values.
    values:
        Iterable of items to embed.  The element type depends on the concrete
        model – most text models expect ``str`` but multimodal models may also
        accept images, audio, …
    max_retries:
        Maximum number of times to retry a failed provider request before giving
        up (defaults to ``2``).  Set to ``0`` to disable retries entirely.
    **kwargs:
        Passed verbatim to the provider's :pyfunc:`EmbeddingModel.embed_many`
        implementation to expose provider-specific features (e.g. ``headers``).

    Returns
    -------
    EmbedManyResult
        Lightweight dataclass with the following attributes:

        * ``values`` – original values in the same order as passed in.
        * ``embeddings`` – list of embedding vectors (``List[List[float]]``)
          aligned with ``values``.
        * ``usage`` – optional :class:`EmbeddingTokenUsage` holding total token
          count if the provider reports usage.
        * ``provider_metadata`` – provider-specific response metadata (may be
          ``None``).
        * ``raw_response`` – raw provider response object(s) for advanced
          inspection (may be ``None``).

    Notes
    -----
    The helper automatically splits *values* into multiple batches if the
    provider exposes a ``max_batch_size`` attribute.
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
    """Embed a *single* value using the given embedding *model*.

    This is a convenience wrapper around :func:`embed_many` that returns the
    first (and only) embedding in a dedicated :class:`EmbedResult` container.

    Parameters
    ----------
    model:
        Instance of :class:`ai_sdk.providers.embedding_model.EmbeddingModel`.
    value:
        The value to embed (commonly a ``str``).  The expected type depends on
        the provider.
    **kwargs:
        Additional keyword arguments forwarded to :func:`embed_many`.

    Returns
    -------
    EmbedResult
        Dataclass containing the original *value*, its *embedding* (vector of
        floats), optional *usage* information and the *raw_response* object from
        the provider.
    """

    res_many = embed_many(model=model, values=[value], **kwargs)
    return EmbedResult(
        value=value,
        embedding=res_many.embeddings[0],
        usage=res_many.usage,
        provider_metadata=res_many.provider_metadata,
        raw_response=res_many.raw_response,
    )
