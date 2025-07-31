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

The resulting :class:`Tool` instance can be passed to
:func:`ai_sdk.generate_text` / :func:`ai_sdk.stream_text` via the *tools*
argument to enable iterative tool calling.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Union
import inspect

HandlerFn = Callable[..., Union[Any, Awaitable[Any]]]


@dataclass(slots=True)
class Tool:  # noqa: D101 – simple value object
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: HandlerFn = field(repr=False)

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
        result = self.handler(**kwargs)
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result


# ---------------------------------------------------------------------------
# Public factory – mirrors functional style of TS SDK *tool()* helper
# ---------------------------------------------------------------------------


def tool(
    *, name: str, description: str, parameters: Dict[str, Any], execute: HandlerFn
) -> "Tool":  # noqa: D401
    '''Create a :class:`ai_sdk.tool.Tool` from a Python callable.

    Parameters
    ----------
    name:
        Unique identifier that the model will use to reference the tool.
    description:
        Human-readable sentence describing the tool’s purpose.
    parameters:
        JSON-Schema describing the accepted arguments as required by the
        OpenAI *function calling* specification.
    execute:
        Python callable implementing the tool logic.  Can be synchronous
        or ``async``.

    Returns
    -------
    Tool
        Configured tool instance ready to be supplied via the *tools*
        argument of :func:`ai_sdk.generate_text` / :func:`ai_sdk.stream_text`.
    '''

    if not all([name, description, parameters, execute]):
        raise ValueError(
            "'name', 'description', 'parameters', and 'execute' are required"
        )

    return Tool(
        name=name, description=description, parameters=parameters, handler=execute
    )
