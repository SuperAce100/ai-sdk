"""
Lightweight *tool* helper mirroring the AI SDK TypeScript implementation.

A *Tool* couples a JSON schema (name, description, parameters) with a Python
handler function.  The :func:`tool` decorator behaves similar to the JavaScript
version – it takes the manifest as its first call and then expects a function
that implements the tool logic::

    @tool({
        "name": "double",
        "description": "Double the given integer.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "number"}},
            "required": ["x"],
        },
    })
    def double(x: int) -> int:  # noqa: D401 – simple demo
        return x * 2

    # Or using Pydantic models for better type safety:
    from pydantic import BaseModel
    
    class DoubleParams(BaseModel):
        x: int
    
    @tool(name="double", description="Double the given integer.", parameters=DoubleParams)
    def double(x: int) -> int:
        return x * 2

The resulting :class:`Tool` instance can be passed to
:func:`ai_sdk.generate_text` / :func:`ai_sdk.stream_text` via the *tools*
argument to enable iterative tool calling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Union, Type
import inspect

from pydantic import BaseModel, create_model

HandlerFn = Callable[..., Union[Any, Awaitable[Any]]]


def _pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to JSON schema format."""
    schema = model.model_json_schema()
    
    # Ensure we have the required structure for OpenAI function calling
    if "properties" not in schema:
        schema["properties"] = {}
    if "required" not in schema:
        schema["required"] = []
    
    return schema


@dataclass(slots=True)
class Tool:  # noqa: D101 – simple value object
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: HandlerFn = field(repr=False)
    _pydantic_model: Type[BaseModel] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Helper utilities used by provider adapters
    # ------------------------------------------------------------------

    def to_openai_dict(self) -> Dict[str, Any]:
        """Return the OpenAI Chat Completions *tools* representation."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, **kwargs: Any) -> Any:  # noqa: D401 – mirrors JS SDK
        """Invoke the wrapped handler with **kwargs, *awaiting* if necessary."""
        # Validate inputs against Pydantic model if available
        if self._pydantic_model is not None:
            validated_data = self._pydantic_model(**kwargs)
            kwargs = validated_data.model_dump()
        
        result = self.handler(**kwargs)
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result


# ---------------------------------------------------------------------------
# Public factory – mirrors functional style of TS SDK *tool()* helper
# ---------------------------------------------------------------------------


def tool(
    *, 
    name: str, 
    description: str, 
    parameters: Dict[str, Any] | Type[BaseModel], 
    execute: HandlerFn
) -> "Tool":  # noqa: D401
    '''Create a :class:`ai_sdk.tool.Tool` from a Python callable.

    Parameters
    ----------
    name:
        Unique identifier that the model will use to reference the tool.
    description:
        Human-readable sentence describing the tool's purpose.
    parameters:
        Either a JSON-Schema dict describing the accepted arguments as required by the
        OpenAI *function calling* specification, or a Pydantic model class that will
        be automatically converted to JSON schema.
    execute:
        Python callable implementing the tool logic.  Can be synchronous
        or ``async``.

    Returns
    -------
    Tool
        Configured tool instance ready to be supplied via the *tools*
        argument of :func:`ai_sdk.generate_text` / :func:`ai_sdk.stream_text`.

    Examples
    --------
    Using JSON schema directly:
    
    >>> @tool(
    ...     name="double",
    ...     description="Double the given integer.",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {"x": {"type": "number"}},
    ...         "required": ["x"],
    ...     }
    ... )
    ... def double(x: int) -> int:
    ...     return x * 2

    Using Pydantic model for better type safety:
    
    >>> from pydantic import BaseModel
    >>> 
    >>> class DoubleParams(BaseModel):
    ...     x: int
    ... 
    >>> @tool(
    ...     name="double",
    ...     description="Double the given integer.",
    ...     parameters=DoubleParams
    ... )
    ... def double(x: int) -> int:
    ...     return x * 2
    '''

    if not all([name, description, parameters, execute]):
        raise ValueError(
            "'name', 'description', 'parameters', and 'execute' are required"
        )

    # Handle Pydantic model vs JSON schema
    pydantic_model = None
    if isinstance(parameters, type) and issubclass(parameters, BaseModel):
        pydantic_model = parameters
        parameters_dict = _pydantic_to_json_schema(parameters)
    elif isinstance(parameters, dict):
        parameters_dict = parameters
    else:
        raise ValueError(
            "parameters must be either a JSON schema dict or a Pydantic model class"
        )

    return Tool(
        name=name, 
        description=description, 
        parameters=parameters_dict, 
        handler=execute,
        _pydantic_model=pydantic_model
    )
