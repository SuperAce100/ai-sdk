#!/usr/bin/env python3
"""
generate_text_example.py: CLI for text completions with ai-sdk.
Usage: generate_text_example.py --prompt "Your prompt" [--provider openai|anthropic] [--model MODEL_ID]
"""

import os
from dotenv import load_dotenv

import argparse
from ai_sdk import openai, anthropic, generate_text

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Text completion CLI using ai-sdk.")
    parser.add_argument(
        "--prompt", required=True, help="The prompt to send to the model."
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

    result = generate_text(model=model, prompt=args.prompt, system=args.system)
    print(result.text)
    if result.usage:
        print(f"Usage: {result.usage}")


if __name__ == "__main__":
    main()
