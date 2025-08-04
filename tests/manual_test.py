Sw# ---------------------------------------------------------------------------
# Manual demo script – extended with *tool calling* showcase
# ---------------------------------------------------------------------------

import asyncio
import os
from typing import List, Optional

# Optional dependency – only used to load .env during local development.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ModuleNotFoundError:
    pass  # run fine without python-dotenv

from ai_sdk import (
    generate_text,
    stream_text,
    generate_object,
    stream_object,
    openai,
    tool,
    embed_many,
    cosine_similarity,
    anthropic,
)
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("AI_SDK_TEST_MODEL", "claude-3-5-haiku-latest")


# ---------------------------------------------------------------------------
# Tool definition with Pydantic model – doubles an integer
# ---------------------------------------------------------------------------


class DoubleParams(BaseModel):
    x: int = Field(description="Integer to double")


def _double_exec(x: int) -> int:  # noqa: D401
    print(f"Double called with {x}")
    return x * 2


double_tool = tool(
    name="double",
    description="Double the given integer.",
    parameters=DoubleParams,
    execute=_double_exec,
)


async def demo_generate_prompt(model):
    print("\n-- Prompt-only generate_text --")
    res = generate_text(model=model, prompt="Say hello from the Python AI SDK port.")
    print("Text:", res.text)
    print("Usage:", res.usage)


async def demo_generate_messages(model):
    print("\n-- Message-based generate_text --")
    messages = [
        CoreSystemMessage(content="You are a helpful assistant."),
        CoreUserMessage(content=[TextPart(text="Respond with the single word 'ack'.")]),
    ]
    res = generate_text(model=model, messages=messages)
    print("Text:", res.text)


async def demo_stream(model):
    print("\n-- Streaming example --")
    result = stream_text(model=model, prompt="Tell a short Python joke.")
    collected = []
    async for delta in result.text_stream:
        print(delta, end="", flush=True)
        collected.append(delta)
    print()
    full = await result.text()
    print("Full:", full)
    assert full == "".join(collected)


async def demo_tool_call(model):
    print("\n-- Tool calling example --")

    # Track iteration steps via callback for demonstration purposes.
    step_types = []

    def on_step(info):
        step_types.append(info.step_type)

    res = generate_text(
        model=model,
        prompt="Please double 7 using the tool.",
        tools=[double_tool],
        on_step=on_step,
    )

    print("Assistant response:", res.text)
    print("Tool steps executed:", step_types)


async def demo_tool_call_streaming(model):
    print("\n-- Tool calling example --")

    result = stream_text(
        model=model, prompt="Please double 7 using the tool.", tools=[double_tool]
    )
    collected = []
    async for delta in result.text_stream:
        print(delta, end="", flush=True)
        collected.append(delta)

    full = await result.text()
    print("Full:", full)
    assert full == "".join(collected)


class RandomNumberDetails(BaseModel):
    number: int
    is_even: bool
    factors: List[int]
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Object generation demos (complex schema)
# ---------------------------------------------------------------------------


async def demo_generate_object(model):
    print("\n-- Generate object example --")
    prompt = (
        'Respond with JSON like {"number": 57, "is_even": false, '
        '"factors": [1, 3, 19, 57], "description": "57 is an odd number. Its factors are 1, 3, 19, and 57."} (no markdown).'
    )
    res = generate_object(model=model, schema=RandomNumberDetails, prompt=prompt)
    print("Object:", res.object)


async def demo_stream_object(model):
    print("\n-- Stream object example --")
    prompt = (
        'Respond with JSON like {"number": 42, "is_even": true, '
        '"factors": [1, 2, 3, 6, 7, 14, 21, 42], "description": "42 is an even number. Its factors are 1, 2, 3, 6, 7, 14, 21, and 42."} (no markdown).'
    )

    def on_partial(obj):
        print("Partial:", obj)

    result = stream_object(
        model=model, schema=RandomNumberDetails, prompt=prompt, on_partial=on_partial
    )
    async for delta in result.object_stream:
        pass

    obj = await result.object(RandomNumberDetails)
    print("\nObject:", obj)


async def demo_embed(_):
    """Demonstrate embedding generation + cosine similarity."""
    print("\n-- Embedding example --")

    # Use a separate *embedding* model (text-embedding-3-small by default)
    EMBED_MODEL_ID = os.getenv("AI_SDK_EMBED_MODEL", "text-embedding-3-small")
    embedding_model = openai.embedding(EMBED_MODEL_ID)  # type: ignore

    values = [
        "cat",
        "dog",
        "aerospace engineer",
        "astronaut",
    ]

    res = embed_many(model=embedding_model, values=values)
    print("Embeddings lengths:", [len(e) for e in res.embeddings])

    sim = cosine_similarity(res.embeddings[0], res.embeddings[1])
    print("Cosine similarity of first two:", sim)
    sim2 = cosine_similarity(res.embeddings[0], res.embeddings[2])
    print("Cosine similarity of first and third:", sim2)
    sim3 = cosine_similarity(res.embeddings[2], res.embeddings[3])
    print("Cosine similarity of third and fourth:", sim3)


async def main():
    # model = openai(MODEL_ID)
    model = anthropic(MODEL_ID, api_key=os.getenv("ANTHROPIC_API_KEY"))
    await demo_generate_prompt(model)
    await demo_generate_messages(model)
    await demo_stream(model)
    await demo_tool_call(model)
    await demo_tool_call_streaming(model)
    await demo_generate_object(model)
    await demo_stream_object(model)
    await demo_embed(model)


if __name__ == "__main__":
    asyncio.run(main())
