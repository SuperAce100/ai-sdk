from __future__ import annotations

"""Public entry-point for the *Python* port of Vercel's AI SDK.

Only a fraction of the original surface is implemented right now – namely the
``generate_text`` and ``stream_text`` helpers together with the OpenAI
provider.  The goal is to mirror the *ergonomics* of the TypeScript version so
that existing examples translate 1-to-1.
"""

from .generate_text import generate_text, stream_text  # noqa: F401  (re-export)
from .providers.openai import openai  # noqa: F401

__all__ = [
    "generate_text",
    "stream_text",
    "openai",
]
