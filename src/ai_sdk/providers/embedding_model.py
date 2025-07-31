"""Abstract base class for *embedding* model providers.

An *embedding model* takes one or multiple *values* (typically strings, but
some providers also support other modalities) and returns the corresponding
vector embeddings.

The public surface mirrors the TypeScript Vercel AI SDK as closely as possible
while keeping the interface intentionally lightweight:

* ``embed_many`` **must** return a *dict* containing at minimum the following
  keys:

  - ``embeddings``: ``List[List[float]]`` – embedding vectors in exactly the
    same order as the provided ``values``.
  - ``usage`` *(optional)*: provider-specific token usage information.
  - ``raw_response`` *(optional)*: the original provider response for advanced
    inspection / debugging.

* Implementations **can** expose a ``max_batch_size`` attribute that signals the
  maximum number of values that can be embedded in a single network request.
  The high-level :pyfunc:`ai_sdk.embed_many` helper will automatically split
  larger inputs into multiple batches respecting this limit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EmbeddingModel(ABC):
    """Abstract base class for embedding providers."""

    # Providers can optionally set this class attribute to indicate the
    # maximum number of *values* accepted in a single call.
    max_batch_size: Optional[int] = None

    # ---------------------------------------------------------------------
    # Public interface – intentionally kept minimal
    # ---------------------------------------------------------------------
    @abstractmethod
    def embed_many(self, values: List[Any], **kwargs: Any) -> Dict[str, Any]:  # noqa: D401
        """Embed *values* and return a provider-specific result dictionary.

        Implementations **must** preserve the order of *values* – the *i*-th
        embedding in the returned list **must** correspond to ``values[i]``.

        Parameters
        ----------
        values:
            List of items to embed.  Their type depends on the provider (most
            commonly ``str`` for text models).
        **kwargs:
            Provider-specific keyword arguments (e.g. ``headers`` for custom HTTP
            headers).  Concrete implementations should document supported keys.

        Returns
        -------
        Dict[str, Any]
            A mapping with at minimum the following keys:

            ``embeddings`` : ``List[List[float]]``
                Embedding vectors aligned with *values*.
            ``usage`` : *Any*, optional
                Provider-specific token usage / billing information.
            ``raw_response`` : *Any*, optional
                Original SDK / HTTP response object for advanced inspection.
        """
