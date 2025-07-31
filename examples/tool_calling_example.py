#!/usr/bin/env python3
"""
tool_calling_example.py: CLI for tool/function calling with ai-sdk.
Usage: tool_calling_example.py --a INT --b INT [--provider openai|anthropic] [--model MODEL_ID]
"""

import os
from dotenv import load_dotenv

load_dotenv()

import argparse
from ai_sdk import openai, anthropic, generate_text, tool


def add_exec(a: int, b: int) -> int:
    return a + b


add_tool = tool(
    name="add",
    description="Add two integers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
    execute=add_exec,
)


def main():
    parser = argparse.ArgumentParser(description="Addition tool CLI using ai-sdk.")
    parser.add_argument("--a", type=int, required=True, help="First integer.")
    parser.add_argument("--b", type=int, required=True, help="Second integer.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default=os.getenv("AI_SDK_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    client = openai if args.provider == "openai" else anthropic
    api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    model = client(args.model, api_key=os.getenv(api_key_env))

    prompt = f"Use the 'add' tool to compute the sum of {args.a} and {args.b}."
    res = generate_text(model=model, prompt=prompt, tools=[add_tool])
    print(res.text)


if __name__ == "__main__":
    main()
