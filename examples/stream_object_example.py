#!/usr/bin/env python3
"""
stream_object_example.py: CLI for streaming structured objects with ai-sdk.
Usage: stream_object_example.py --username USERNAME [--provider openai|anthropic] [--model MODEL_ID]
"""

import os
from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
from pydantic import BaseModel
from ai_sdk import openai, anthropic, stream_object


class UserProfile(BaseModel):
    username: str
    followers: int
    bio: str


async def main():
    parser = argparse.ArgumentParser(
        description="UserProfile streaming CLI using ai-sdk."
    )
    parser.add_argument(
        "--username", required=True, help="Username to fetch profile for."
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default=os.getenv("AI_SDK_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    client = openai if args.provider == "openai" else anthropic
    api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    model = client(args.model, api_key=os.getenv(api_key_env))

    prompt = f"Provide profile JSON for user '{args.username}' with fields username, followers, bio."
    result = stream_object(model=model, schema=UserProfile, prompt=prompt)
    async for _ in result.object_stream:
        pass
    profile = await result.object(UserProfile)
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
