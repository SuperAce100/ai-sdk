from __future__ import annotations
from .generate_text import generate_text, stream_text
from .embed import embed_many, embed, cosine_similarity
from .generate_object import generate_object, stream_object
from .providers.openai import openai
from .tool import tool, Tool
from .providers.anthropic import anthropic

"""Public entry-point for the *Python* port of Vercel's AI SDK.

Only a fraction of the original surface is implemented right now – namely the
``generate_text`` and ``stream_text`` helpers together with the OpenAI
provider.  The goal is to mirror the *ergonomics* of the TypeScript version so
that existing examples translate 1-to-1.
"""


__all__ = [
    "generate_text",
    "stream_text",
    "generate_object",
    "stream_object",
    "embed_many",
    "embed",
    "cosine_similarity",
    "openai",
    "anthropic",
    "tool",
    "Tool",
]
