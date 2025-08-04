#!/usr/bin/env python3
"""
tool_calling_example.py: CLI for tool/function calling with ai-sdk.
Usage: tool_calling_example.py --a INT --b INT [--provider openai|anthropic] [--model MODEL_ID] [--use-pydantic]
"""

import os
from dotenv import load_dotenv

load_dotenv()

import argparse
from pydantic import BaseModel, Field
from typing import Optional, List
from ai_sdk import openai, anthropic, generate_text, tool


# ---------------------------------------------------------------------------
# Tool definitions with both JSON schema and Pydantic models
# ---------------------------------------------------------------------------

# JSON Schema approach (legacy)
def add_exec(a: int, b: int) -> int:
    return a + b


add_tool_json = tool(
    name="add",
    description="Add two integers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
    execute=add_exec,
)


# Pydantic model approach (recommended)
class AddParams(BaseModel):
    a: int = Field(description="First integer to add")
    b: int = Field(description="Second integer to add")


def add_exec_pydantic(a: int, b: int) -> int:
    return a + b


add_tool_pydantic = tool(
    name="add_pydantic",
    description="Add two integers using Pydantic validation.",
    parameters=AddParams,
    execute=add_exec_pydantic,
)


# Complex Pydantic model example
class CalculatorParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Mathematical operation", pattern="^[+\\-*/]$")


def calculator_exec(a: float, b: float, operation: str) -> float:
    """Perform basic mathematical operations."""
    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


calculator_tool = tool(
    name="calculator",
    description="Perform basic mathematical operations",
    parameters=CalculatorParams,
    execute=calculator_exec,
)


# User profile example with validation
class UserProfileParams(BaseModel):
    name: str = Field(description="User's full name", min_length=1, max_length=100)
    age: int = Field(description="User's age", ge=0, le=120)
    email: Optional[str] = Field(default=None, description="User's email address")
    interests: List[str] = Field(default_factory=list, description="User's interests")
    is_active: bool = Field(default=True, description="Whether the user is active")


def create_user_profile_exec(
    name: str, 
    age: int, 
    email: Optional[str] = None, 
    interests: List[str] = None, 
    is_active: bool = True
) -> dict:
    """Create a new user profile."""
    return {
        "id": f"user_{hash(name) % 10000}",
        "name": name,
        "age": age,
        "email": email,
        "interests": interests or [],
        "is_active": is_active,
        "created_at": "2024-01-01T00:00:00Z"
    }


user_profile_tool = tool(
    name="create_user_profile",
    description="Create a new user profile with validation",
    parameters=UserProfileParams,
    execute=create_user_profile_exec,
)


def main():
    parser = argparse.ArgumentParser(description="Tool calling CLI using ai-sdk.")
    parser.add_argument("--a", type=int, required=True, help="First integer.")
    parser.add_argument("--b", type=int, required=True, help="Second integer.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default=os.getenv("AI_SDK_MODEL", "gpt-4o-mini"))
    parser.add_argument("--use-pydantic", action="store_true", help="Use Pydantic model instead of JSON schema")
    parser.add_argument("--demo", choices=["add", "calculator", "user-profile"], default="add", 
                       help="Demo to run")
    args = parser.parse_args()

    client = openai if args.provider == "openai" else anthropic
    api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    model = client(args.model, api_key=os.getenv(api_key_env))

    # Choose tool based on demo type and Pydantic preference
    if args.demo == "add":
        if args.use_pydantic:
            tool_to_use = add_tool_pydantic
            prompt = f"Use the 'add_pydantic' tool to compute the sum of {args.a} and {args.b}."
        else:
            tool_to_use = add_tool_json
            prompt = f"Use the 'add' tool to compute the sum of {args.a} and {args.b}."
    elif args.demo == "calculator":
        tool_to_use = calculator_tool
        prompt = f"Use the calculator tool to multiply {args.a} and {args.b}."
    elif args.demo == "user-profile":
        tool_to_use = user_profile_tool
        prompt = f"Create a user profile for Alice, age 30, with interests in Python and AI."
    else:
        raise ValueError(f"Unknown demo: {args.demo}")

    print(f"Running {args.demo} demo with {'Pydantic' if args.use_pydantic else 'JSON schema'} approach...")
    print(f"Provider: {args.provider}, Model: {args.model}")
    print(f"Prompt: {prompt}")
    print("-" * 50)

    res = generate_text(model=model, prompt=prompt, tools=[tool_to_use])
    print("Response:", res.text)

    # Show tool information
    print("\nTool Information:")
    print(f"  Name: {tool_to_use.name}")
    print(f"  Description: {tool_to_use.description}")
    print(f"  Uses Pydantic: {tool_to_use._pydantic_model is not None}")
    if tool_to_use._pydantic_model:
        print(f"  Pydantic Model: {tool_to_use._pydantic_model.__name__}")
    print(f"  JSON Schema: {tool_to_use.parameters}")


if __name__ == "__main__":
    main()
