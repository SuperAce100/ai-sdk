import os

import pytest
from dotenv import load_dotenv

from ai_sdk import generate_text, stream_text, generate_object, stream_object, openai
from ai_sdk.types import (
    CoreSystemMessage,
    CoreUserMessage,
    TextPart,
)
from pydantic import BaseModel

load_dotenv()


# Skip all tests if OPENAI_API_KEY is missing to avoid network failures in CI
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)

MODEL_ID = os.getenv("AI_SDK_TEST_MODEL", "gpt-3.5-turbo")
model = openai(MODEL_ID)


@pytest.mark.asyncio
async def test_generate_text_simple():
    """Basic prompt-only generation returns non-empty text and usage."""
    res = generate_text(model=model, prompt="Hello! Respond with the word 'hi'.")
    assert "hi".lower() in res.text.lower()
    # Usage dataclass populated
    assert res.usage is not None
    assert res.usage.total_tokens >= 1


@pytest.mark.asyncio
async def test_generate_text_with_messages():
    """Generation using typed Core*Message list."""
    messages = [
        CoreSystemMessage(content="You are a polite assistant."),
        CoreUserMessage(content=[TextPart(text="Say the word 'yes'.")]),
    ]
    res = generate_text(model=model, messages=messages, temperature=0)
    assert "yes" in res.text.lower()


@pytest.mark.asyncio
async def test_stream_text_iterable():
    """stream_text yields multiple deltas and assembles full text correctly."""
    result = stream_text(model=model, prompt="Repeat the word foo five times.")
    collected = []
    async for delta in result.text_stream:
        collected.append(delta)
    full_text = await result.text()
    # ensure concatenation of deltas equals final text
    assert full_text == "".join(collected)
    assert full_text.lower().count("foo") >= 5


# ---------------------------------------------------------------------------
# New object-generation tests
# ---------------------------------------------------------------------------


class AckSchema(BaseModel):
    ack: str


@pytest.mark.asyncio
async def test_generate_object_simple():
    """generate_object returns parsed Pydantic model instance."""

    res = generate_object(
        model=model,
        schema=AckSchema,
        prompt='Respond with JSON {"ack": "yes"} and nothing else.',
    )
    assert isinstance(res.object, AckSchema)
    assert res.object.ack.lower() == "yes"
    # Ensure raw_text keeps original string for debugging
    assert "ack" in res.raw_text.lower()


@pytest.mark.asyncio
async def test_stream_object_iterable():
    """stream_object yields chunks and returns full object on completion."""

    result = stream_object(
        model=model,
        schema=AckSchema,
        prompt='Respond with JSON {"ack": "foo"} and nothing else.',
    )
    collected = []
    async for delta in result.object_stream:
        collected.append(delta)

    obj = await result.object(AckSchema)
    assert isinstance(obj, AckSchema)
    assert obj.ack.lower() == "foo"
    # At least some chunks should have been streamed
    assert collected
