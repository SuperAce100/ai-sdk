import asyncio
import json

from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env

from ai_sdk import generate_text, stream_text, openai


async def main() -> None:
    model = openai("gpt-4.1-mini")

    # ------------------------------------------------------------------
    # 1. Non-streaming generation
    # ------------------------------------------------------------------
    res = generate_text(model=model, prompt="Say hello from the Python AI SDK port.")
    print("\n=== generate_text result ===\n", res.text)
    print(res)

    # ------------------------------------------------------------------
    # 2. Streaming generation
    # ------------------------------------------------------------------
    print("\n=== stream_text deltas ===")
    stream_res = stream_text(model=model, prompt="Tell me a short Python joke.")
    async for delta in stream_res.text_stream:
        print(delta, end="", flush=True)

    full_text = await stream_res.text()
    print("\n\n=== stream_text full ===\n", full_text)


if __name__ == "__main__":
    asyncio.run(main())
