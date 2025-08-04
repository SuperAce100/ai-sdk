# Contributing to AI SDK Python

Thank you for your interest in contributing to AI SDK Python! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) for package management
- [Ty](https://github.com/astral-sh/ty) for type checking
- [Ruff](https://docs.astral.sh/ruff/) for linting and formatting

### Quick Start

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/your-username/ai-sdk.git
   cd ai-sdk
   ```

2. **Set up the development environment**

   ```bash
   uv sync --dev
   ```

3. **Install pre-commit hooks**

   ```bash
   uv run pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   uv run pytest
   ```

## Development Setup

### Environment Setup

1. **Create a virtual environment**

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   uv sync --dev
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### API Keys for Testing

You'll need API keys for testing different providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Project Structure

```
ai-sdk/
├── src/ai_sdk/           # Main package source
│   ├── __init__.py       # Package initialization
│   ├── embed.py          # Embedding functionality
│   ├── generate_object.py # Structured output generation
│   ├── generate_text.py  # Text generation
│   ├── providers/        # Provider implementations
│   │   ├── anthropic.py  # Anthropic provider
│   │   ├── openai.py     # OpenAI provider
│   │   ├── language_model.py # Base language model
│   │   └── embedding_model.py # Base embedding model
│   ├── tool.py           # Tool calling functionality
│   └── types.py          # Type definitions
├── tests/                # Test suite
│   ├── test_ai_sdk.py    # Core functionality tests
│   ├── test_embed.py     # Embedding tests
│   ├── test_generate_object_dummy.py # Object generation tests
│   └── test_tool_calling.py # Tool calling tests
├── examples/             # Usage examples
├── docs/                 # Documentation
└── pyproject.toml        # Project configuration
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `isort` configuration
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

### Type Checking

We use [Ty](https://github.com/astral-sh/ty) for type checking:

```bash
# Run type checker
uv run ty check src/
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/ai_sdk

# Run specific test file
uv run pytest tests/test_generate_text.py

# Run tests with verbose output
uv run pytest -v
```

### Writing Tests

1. **Test Structure**

   - Place tests in the `tests/` directory
   - Use descriptive test function names
   - Group related tests in classes

2. **Test Examples**

   ```python
   import pytest
   from ai_sdk import generate_text, openai

   class TestGenerateText:
       def test_basic_text_generation(self):
           model = openai("gpt-4o-mini")
           result = generate_text(model=model, prompt="Hello")
           assert result.text is not None
           assert len(result.text) > 0

       @pytest.mark.asyncio
       async def test_streaming_text(self):
           model = openai("gpt-4o-mini")
           stream = stream_text(model=model, prompt="Hello")
           async for chunk in stream.text_stream:
               assert chunk is not None
   ```

3. **Mocking External APIs**

   ```python
   import pytest
   from unittest.mock import patch

   @patch('ai_sdk.providers.openai.OpenAIClient')
   def test_openai_integration(mock_client):
       # Mock the OpenAI client
       mock_client.return_value.chat.completions.create.return_value = MockResponse()
       # Test your code
   ```

### Test Coverage

We aim for high test coverage:

```bash
# Generate coverage report
uv run pytest --cov=src/ai_sdk --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

### Code Documentation

1. **Docstrings**

   - Use Google-style docstrings
   - Include type hints
   - Provide usage examples

   ```python
   def generate_text(
       model: LanguageModel,
       prompt: str,
       system: Optional[str] = None,
       **kwargs
   ) -> TextGenerationResult:
       """Generate text using the specified language model.

       Args:
           model: The language model to use for generation.
           prompt: The input prompt for text generation.
           system: Optional system message to set context.
           **kwargs: Additional parameters to pass to the model.

       Returns:
           TextGenerationResult: The generated text and metadata.

       Raises:
           ValueError: If the model is not properly configured.
           APIError: If the API request fails.

       Example:
           >>> from ai_sdk import generate_text, openai
           >>> model = openai("gpt-4o-mini")
           >>> result = generate_text(model=model, prompt="Hello, world!")
           >>> print(result.text)
           Hello, world!
       """
   ```

2. **Type Hints**
   - Use type hints for all public functions
   - Import types from `typing` module
   - Use `Optional` for nullable parameters

### Documentation Structure

The documentation is built with [Mintlify](https://mintlify.com/):

```
docs/
├── index.mdx              # Homepage
├── sdk/                   # SDK documentation
│   ├── introduction.mdx   # Getting started
│   ├── concepts.mdx       # Core concepts
│   ├── generate_text.mdx  # Text generation
│   ├── generate_object.mdx # Object generation
│   ├── embed.mdx          # Embeddings
│   ├── tool.mdx           # Tool calling
│   └── providers/         # Provider-specific docs
│       ├── openai.mdx     # OpenAI provider
│       └── anthropic.mdx  # Anthropic provider
└── examples/              # Example documentation
    ├── basic-text.mdx     # Basic text generation
    ├── streaming.mdx      # Streaming examples
    └── structured-output.mdx # Structured output
```

### Writing Documentation

1. **Structure**

   - Use clear, descriptive titles
   - Include code examples
   - Provide step-by-step guides

2. **Code Examples**

   ````markdown
   ```python
   from ai_sdk import generate_text, openai

   model = openai("gpt-4o-mini")
   result = generate_text(model=model, prompt="Hello, world!")
   print(result.text)
   ```
   ````

   ```

   ```

3. **Components**
   - Use Mintlify components for better UX
   - Include tips, warnings, and notes
   - Add interactive code snippets

## Tool Development

### Creating New Tools

When adding new tools to the SDK, follow these guidelines:

1. **Use Pydantic Models (Recommended)**

   - Define parameter schemas using Pydantic models
   - Include field descriptions and validation constraints
   - Provide clear, descriptive field names

2. **Tool Structure**

   ```python
   from pydantic import BaseModel, Field
   from ai_sdk import tool

   class MyToolParams(BaseModel):
       input: str = Field(description="Input parameter")
       option: bool = Field(default=False, description="Optional flag")

   @tool(
       name="my_tool",
       description="Clear description of what the tool does",
       parameters=MyToolParams
   )
   def my_tool(input: str, option: bool = False) -> str:
       # Tool implementation
       return f"Processed: {input}"
   ```

3. **Validation and Error Handling**

   - Use Pydantic validation constraints (e.g., `ge`, `le`, `min_length`)
   - Handle errors gracefully with meaningful messages
   - Test edge cases and invalid inputs

4. **Testing Tools**

   - Test both valid and invalid inputs
   - Verify Pydantic model validation
   - Test tool execution and return values
   - Mock external dependencies

5. **Documentation**
   - Update tool documentation with examples
   - Include parameter descriptions
   - Show both Pydantic and JSON schema approaches

### Tool Best Practices

- **Clear Descriptions**: Provide descriptive field and tool descriptions
- **Type Safety**: Use Pydantic models for automatic validation
- **Error Handling**: Gracefully handle validation and runtime errors
- **Testing**: Comprehensive test coverage for all tool functionality
- **Documentation**: Clear examples and usage patterns

## Submitting Changes

### Workflow

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Run quality checks**

   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run ty check src/
   uv run pytest
   ```

4. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

**Examples:**

```
feat: add support for Claude 3.5 Sonnet
fix: resolve token counting issue in streaming
docs: update OpenAI provider documentation
test: add comprehensive embedding tests
```

### Pull Request Guidelines

1. **Title**: Clear, descriptive title
2. **Description**: Explain what and why, not how
3. **Checklist**: Include a checklist of completed tasks
4. **Tests**: Ensure all tests pass
5. **Documentation**: Update relevant documentation

**Example PR Description:**

```markdown
## Description

Adds support for Claude 3.5 Sonnet model in the Anthropic provider.

## Changes

- Add `claude-3.5-sonnet` model identifier
- Update model documentation with pricing info
- Add tests for new model

## Checklist

- [x] Code follows project style guidelines
- [x] Tests added for new functionality
- [x] Documentation updated
- [x] All tests pass
- [x] Type checking passes

## Related Issues

Closes #123
```

## Code Review Process

### Review Guidelines

1. **Code Quality**

   - Follows project standards
   - Proper error handling
   - Good test coverage

2. **Documentation**

   - Clear docstrings
   - Updated README/docs
   - Good commit messages

3. **Testing**
   - Tests for new functionality
   - No breaking changes
   - All tests pass

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No breaking changes
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version**

   ```bash
   # Update pyproject.toml version
   # Update CHANGELOG.md
   ```

2. **Create release branch**

   ```bash
   git checkout -b release/v1.2.0
   ```

3. **Run release checks**

   ```bash
   uv run pytest
   uv run ruff check .
   uv run ty check src/
   ```

4. **Build and publish**

   ```bash
   uv build
   uv publish
   ```

5. **Create GitHub release**
   - Tag the release
   - Add release notes
   - Attach built artifacts

## Getting Help

### Communication Channels

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Discord**: Join our community Discord server

### Resources

- [Python Documentation](https://docs.python.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License

By contributing to AI SDK Python, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AI SDK Python! 🚀
