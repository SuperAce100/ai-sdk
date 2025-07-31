#!/usr/bin/env python3
"""
stream_text_example.py: CLI for asynchronous streaming completions with ai-sdk.
Usage: stream_text_example.py --prompt "Your prompt" [--provider openai|anthropic] [--model MODEL_ID]
"""

import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
from ai_sdk import openai, anthropic, stream_text


async def main():
    parser = argparse.ArgumentParser(
        description="Streaming completion CLI using ai-sdk."
    )
    parser.add_argument(
        "--prompt", required=True, help="The prompt to stream to the model."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider to use.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AI_SDK_MODEL", "gpt-4o-mini"),
        help="Model ID to use.",
    )
    parser.add_argument("--system", default=None, help="Optional system message.")
    args = parser.parse_args()

    client = openai if args.provider == "openai" else anthropic
    api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    model = client(args.model, api_key=os.getenv(api_key_env))

    print("Streaming output:", end=" ")
    stream = stream_text(model=model, prompt=args.prompt, system=args.system)
    async for chunk in stream.text_stream:
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
