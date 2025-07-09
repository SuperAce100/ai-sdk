import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from ai_sdk import generate_text, stream_text, openai
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart

MODEL_ID = os.getenv("AI_SDK_TEST_MODEL", "gpt-3.5-turbo")


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


async def main():
    model = openai(MODEL_ID)
    await demo_generate_prompt(model)
    await demo_generate_messages(model)
    await demo_stream(model)


if __name__ == "__main__":
    asyncio.run(main())
