# The Vercel AI SDK comes to Python

A pure Python re-implementation of Vercel's popular AI SDK for TypeScript. Zero-configuration functions that work consistently across providers with first-class streaming, tool-calling, and structured output support.

## Why another SDK?

Python is the defacto language for AI. However, to actually get started with AI, you'll need to 1. use a bloated external framework and install a bunch of dependencies, or 2. use an incredibly confusing API client (to simply call an LLM, you need `client.chat.completions.create(**kwargs).result.choices[0].message.content`).

### Features

- **Zero-configuration functions** that work consistently across providers
- **First-class streaming** & **tool-calling** support
- **Strong Pydantic types** throughout - you know exactly what you're getting
- **Strict structured-output** generation and streaming via Pydantic models
- **Provider-agnostic embeddings** with built-in batching & retry logic
- **Tiny dependency footprint** - no bloated external frameworks

## Installation

Install via UV (Python package manager):

```bas
uv add ai-sdk-python
```

Or with pip:

```bash
pip install ai-sdk-python
```

That's it - no extra build steps or config files.

## Quick Start

Get started in just a few lines of code.

### Basic Text Generation

```python
from ai_sdk import generate_text, openai

model = openai("gpt-4o-mini")
res = generate_text(model=model, prompt="Tell me a haiku about Python")
print(res.text)
```

### Streaming Text

```python
import asyncio
from ai_sdk import stream_text, openai

async def main():
    model = openai("gpt-4o-mini")
    stream_res = stream_text(model=model, prompt="Write a short story")

    async for chunk in stream_res.text_stream:
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Structured Output

```python
from ai_sdk import generate_object, openai
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

model = openai("gpt-4o-mini")
res = generate_object(
    model=model,
    schema=Person,
    prompt="Create a person named Alice, age 30"
)
print(res.object)  # Person(name='Alice', age=30)
```

### Embeddings & Similarity

```python
from ai_sdk import embed_many, cosine_similarity, openai

model = openai.embedding("text-embedding-3-small")
texts = ["The cat sat on the mat.", "A dog was lying on the rug."]
result = embed_many(model=model, values=texts)

similarity = cosine_similarity(result.embeddings[0], result.embeddings[1])
print(f"Similarity: {similarity:.3f}")
```

### Tool Calling

```python
from ai_sdk import tool, generate_text, openai
from pydantic import BaseModel, Field

# Using Pydantic models (recommended)
class AddParams(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

@tool(
    name="add",
    description="Add two numbers.",
    parameters=AddParams
)
def add(a: float, b: float) -> float:
    return a + b

model = openai("gpt-4o-mini")
res = generate_text(
    model=model,
    prompt="What is 21 + 21?",
    tools=[add],
)
print(res.text)  # "The result is 42."
```

## Core Functions

### Text Generation

- **`generate_text`** - Synchronous text generation with rich metadata
- **`stream_text`** - Asynchronous streaming with real-time callbacks

### Object Generation

- **`generate_object`** - Structured output with Pydantic validation
- **`stream_object`** - Streaming structured output with partial updates

### Embeddings

- **`embed`** - Single-value embedding helper
- **`embed_many`** - Batch embedding with automatic batching
- **`cosine_similarity`** - Semantic similarity calculation

### Tools

- **`tool`** - Define LLM-callable functions with Pydantic models or JSON schema

## Advanced Examples

### Chat-based Completion

```python
from ai_sdk import generate_text, openai
from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart

model = openai("gpt-4o-mini")
messages = [
    CoreSystemMessage(content="You are a helpful assistant."),
    CoreUserMessage(content=[TextPart(text="Respond with 'ack'.")]),
]
res = generate_text(model=model, messages=messages)
print(res.text)
```

### Streaming Structured Output

```python
import asyncio
from ai_sdk import stream_object, openai
from pydantic import BaseModel

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]

async def main():
    model = openai("gpt-4o-mini")
    result = stream_object(
        model=model,
        schema=Recipe,
        prompt="Create a recipe for chocolate chip cookies"
    )

    async for chunk in result.object_stream:
        print(chunk, end="", flush=True)

    recipe = await result.object()
    print(f"\n\nComplete recipe: {recipe}")

asyncio.run(main())
```

### Semantic Search

```python
from ai_sdk import embed_many, cosine_similarity, openai

model = openai.embedding("text-embedding-3-small")

# Knowledge base
documents = [
    "Python is a programming language.",
    "Machine learning involves training models on data.",
    "Databases store and retrieve information."
]

# Search query
query = "How do I learn to code?"

# Embed everything
all_texts = [query] + documents
result = embed_many(model=model, values=all_texts)

query_embedding = result.embeddings[0]
doc_embeddings = result.embeddings[1:]

# Find most similar document
similarities = []
for i, doc_embedding in enumerate(doc_embeddings):
    sim = cosine_similarity(query_embedding, doc_embedding)
    similarities.append((sim, documents[i]))

# Get top result
top_result = max(similarities, key=lambda x: x[0])
print(f"Most relevant: {top_result[1]}")
```

### Complex Tool Example

```python
from ai_sdk import tool, generate_text, openai
import requests

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "New York": "72°F, Sunny",
        "London": "55°F, Rainy",
        "Tokyo": "68°F, Cloudy"
    }
    return weather_data.get(city, "Weather data not available")

weather_tool = tool(
    name="get_weather",
    description="Get current weather information for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name to get weather for"
            }
        },
        "required": ["city"]
    },
    execute=get_weather
)

model = openai("gpt-4o-mini")
res = generate_text(
    model=model,
    prompt="What's the weather like in New York?",
    tools=[weather_tool],
)
print(res.text)
```

## Provider Support

The SDK is provider-agnostic. Currently supported:

- **OpenAI** - GPT models, embeddings, function calling
- **Anthropic** - Claude models

```python
from ai_sdk import generate_text, openai, anthropic

# OpenAI
openai_model = openai("gpt-4o-mini")
res1 = generate_text(model=openai_model, prompt="Hello")

# Anthropic
anthropic_model = anthropic("claude-3-sonnet-20240229")
res2 = generate_text(model=anthropic_model, prompt="Hello")
```

## Key Benefits

### 1. **Zero Configuration**

No complex setup - just import and use:

```python
from ai_sdk import generate_text, openai
res = generate_text(model=openai("gpt-4o-mini"), prompt="Hello!")
```

### 2. **Provider Agnostic**

Swap providers without changing code:

```python
# Works with any provider
model = openai("gpt-4o-mini")  # or anthropic("claude-3-sonnet-20240229")
res = generate_text(model=model, prompt="Hello!")
```

### 3. **Strong Typing**

Full Pydantic integration for type safety:

```python
from pydantic import BaseModel
from ai_sdk import generate_object

class User(BaseModel):
    name: str
    age: int

res = generate_object(model=model, schema=User, prompt="Create a user")
user = res.object
```

### 4. **Built-in Streaming**

Real-time text generation:

```python
async for chunk in stream_text(model=model, prompt="Tell a story").text_stream:
    print(chunk, end="", flush=True)
```

### 5. **Automatic Tool Calling**

Define tools once, use everywhere:

```python
add = tool(name="add", description="Add numbers",
           parameters={...}, execute=lambda x, y: x + y)

res = generate_text(model=model, prompt="What's 2+2?", tools=[add])
```

## Examples

Check out the [examples directory](examples/) for complete working examples:

- `generate_text_example.py` - Basic text generation
- `stream_text_example.py` - Streaming text generation
- `generate_object_example.py` - Structured output generation
- `stream_object_example.py` - Streaming structured output
- `embeddings_example.py` - Embedding and similarity
- `tool_calling_example.py` - Tool calling with Pydantic models and JSON schema

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.
