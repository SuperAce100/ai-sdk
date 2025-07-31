# ai-sdk

Python SDK for unified access to large language models, embeddings, and function/tool calling.

## Installation

Install via UV (Python package manager):

```bash
uv add ai-sdk
```

Or with pip:

```bash
pip install ai-sdk
```

## Quickstart Examples

```python
import asyncio
from ai_sdk import (
    generate_text,
    stream_text,
    generate_object,
    stream_object,
    openai,
    anthropic,
    tool,
    embed_many,
    cosine_similarity,
)
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart
from pydantic import BaseModel

# 1. Synchronous text completion (prompt)
model = openai("gpt-4o-mini", api_key="YOUR_OPENAI_KEY")
res = generate_text(model=model, prompt="Hello, world!")
print(res.text)

# 2. Chat-based completion (messages)
messages = [
    CoreSystemMessage(content="You are a helpful assistant."),
    CoreUserMessage(content=[TextPart(text="Respond with 'ack'.")]),
]
res = generate_text(model=model, messages=messages)
print(res.text)

# 3. Streaming completion (async)
async def demo_stream():
    stream = stream_text(model=model, prompt="Tell me a joke.")
    async for chunk in stream.text_stream:
        print(chunk, end="")
    print()

asyncio.run(demo_stream())

# 4. Structured object output
class Person(BaseModel):
    name: str
    age: int

res = generate_object(
    model=model,
    schema=Person,
    prompt="Respond with JSON: {'name':'Alice','age':30}"  # no markdown
)
print(res.object)

# 5. Streaming structured output (async)
async def demo_stream_object():
    stream_obj = stream_object(
        model=model,
        schema=Person,
        prompt="Respond with JSON: {'name':'Bob','age':25}"  # no markdown
    )
    async for _ in stream_obj.object_stream:
        pass
    person = await stream_obj.object(Person)
    print(person)

asyncio.run(demo_stream_object())

# 6. Embeddings + cosine similarity
embed_model = openai.embedding("text-embedding-3-small")
values = ["cat", "dog"]
emb = embed_many(model=embed_model, values=values)
print([len(v) for v in emb.embeddings])
print(cosine_similarity(emb.embeddings[0], emb.embeddings[1]))

# 7. Tool/function calling
# Define a simple 'double' tool
def double_exec(x: int) -> int:
    return x * 2

double_tool = tool(
    name="double",
    description="Double an integer.",
    parameters={
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    },
    execute=double_exec,
)
res = generate_text(
    model=model,
    prompt="Please double 7 using the tool.",
    tools=[double_tool],
)
print(res.text)
```
