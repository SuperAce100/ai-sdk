import asyncio
import os

import pytest
from dotenv import load_dotenv

from ai_sdk import generate_text, stream_text, openai
from ai_sdk.types import (
    CoreSystemMessage,
    CoreUserMessage,
    TextPart,
)

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
    res = generate_text(model=model, messages=messages)
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
