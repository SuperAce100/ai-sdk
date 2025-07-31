#!/usr/bin/env python3
"""
generate_object_example.py: CLI for structured object generation with ai-sdk.
Usage: generate_object_example.py --city CITY [--provider openai|anthropic] [--model MODEL_ID]
"""

import os
from dotenv import load_dotenv

load_dotenv()

import argparse
from pydantic import BaseModel
from ai_sdk import openai, anthropic, generate_object


class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    description: str


def main():
    parser = argparse.ArgumentParser(
        description="Weather report JSON generation CLI using ai-sdk."
    )
    parser.add_argument(
        "--city", required=True, help="City to generate weather report for."
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default=os.getenv("AI_SDK_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    client = openai if args.provider == "openai" else anthropic
    api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    model = client(args.model, api_key=os.getenv(api_key_env))

    prompt = f"Provide a weather report for the city '{args.city}' in JSON with keys city, temperature_c, description."
    result = generate_object(model=model, schema=WeatherReport, prompt=prompt)
    print(result.object.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
