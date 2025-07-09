# ---------------------------------------------------------------------------
# Manual demo script – extended with *tool calling* showcase
# ---------------------------------------------------------------------------

import asyncio
import os

# Optional dependency – only used to load .env during local development.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ModuleNotFoundError:
    pass  # run fine without python-dotenv

from ai_sdk import generate_text, stream_text, openai, tool
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart

MODEL_ID = os.getenv("AI_SDK_TEST_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Tool definition – doubles an integer
# ---------------------------------------------------------------------------


def _double_exec(x: int) -> int:  # noqa: D401
    print(f"Double called with {x}")
    return x * 2


double_tool = tool(
    name="double",
    description="Double the given integer.",
    parameters={
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    },
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


async def main():
    model = openai(MODEL_ID)
    await demo_generate_prompt(model)
    await demo_generate_messages(model)
    await demo_stream(model)
    await demo_tool_call(model)
    await demo_tool_call_streaming(model)


if __name__ == "__main__":
    asyncio.run(main())
